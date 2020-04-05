# Get access to parent directory
import os, sys

sys.path.append(os.path.dirname(os.getcwd()))
import tensorflow as tf
from utils.utils import load_image_batch, load_image
from variables import *
import time


def get_optimizer():
    return tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
                                    name='Adam')


def log2(x):
    # Nor sure yet whether needed
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator


def log10(x):
    # Nor sure yet whether needed
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def neg_log_likelihood(real_idx, pred_prob_dist):
    """
        Computed the respective negative log-likelihood for each batch element.
        :param real_idx: tensor of correct word token indices to be predicted per batch element
        :param pred_prob_dist: For each batch element probability distribution over entire vocab giving probability for
                               selecting each of the available words in the vocab next
        :return: tensor of negative log-likelihood per batch element
    """
    print('###############################################')
    print('Real:\t\t', real_idx)
    print('pred max:\t', tf.math.reduce_max(pred_prob_dist, axis=1))
    print('pred mean:\t', tf.math.reduce_mean(pred_prob_dist, axis=1))
    print('pred min:\t', tf.math.reduce_min(pred_prob_dist, axis=1))
    print('pred:', pred_prob_dist)
    # Construct list of enumerated indices to retrieve predicted probs of correct classes/words
    batch_idx = [[i, x] for i, x in enumerate(real_idx)]
    # Extract probabilities for correct words
    likelihood = tf.gather_nd(pred_prob_dist, batch_idx)
    # Compute & return negative log-likelihood
    nll = -tf.math.log(likelihood)
    print('NLL:\t\t', nll)
    print('###############################################')
    return nll


def get_loss_object():
    #"""
    #    Info: https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy :
    #          "By default, we assume that y_pred encodes a probability distribution." -- from_logits=False
    #    :return:  Loss function
    #"""
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    #return neg_log_likelihood


def loss_function(real, pred):
    """
        Computes average loss over batch. For each batch element, only non-padded mis-predicted words count.

        :param real: Ints, indicating for each batch element correct word index for next word in caption.
        :param pred: For each batch element, probability distribution over entire vocab, indicating prob of each word to
                     be appended to caption next (as predicted).
        :return: Average loss over batch.
    """
    loss_object = get_loss_object()
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    print('LOSS UNMASKED:', loss_)
    mask = tf.cast(mask, dtype=loss_.dtype)
    print('MASK:', mask)
    loss_ *= mask
    print('LOSS MASKED:', loss_)
    return tf.reduce_mean(loss_)


#@tf.function
def train_step(img_batch, targets, decoder, attention_module, encoder, tokenizer, optimizer, train_flag):
    #print('Train step......')
    loss = 0
    # Initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=targets.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * targets.shape[0], 1)
    #print('Dec input:', dec_input)
    #print('Before expansion:', [tokenizer.word_index['<start>']] * targets.shape[0])
    #print('TS - Shape image batch:', img_batch.shape)
    #print('TS - Shape targets:', targets.shape)

    # Prediction step
    with tf.GradientTape() as tape:
        features = encoder(img_batch)
        print('Features:', features)
        #print('TS - Shape features:', features.shape)
        # Repeat, appending caption by one word at a time
        #print('TS - Shape targets:', targets.shape)
        #print('Going to construct captions...')
        for i in range(1, targets.shape[1]):
            # Passing the features through the attention module and decoder
            context_vector, attention_weights = attention_module(features, hidden)

            #print('TS - Shape context vector:', context_vector.shape)
            predictions, hidden = decoder(dec_input, hidden, context_vector)
            print('Predictions:', predictions)

            loss += loss_function(targets[:, i], predictions)
            print('LOSSSSSSSS::::::::::::::::::', loss)
            # Using teacher forcing
            dec_input = tf.expand_dims(targets[:, i], 1)

            # Save unecessary forward-passes if all captions are done
            if tf.math.reduce_sum(dec_input, axis=0) == 0:
                break

            print('Iteration:', i)
            print('Targets:', targets)
            print('Decoder input:', dec_input)
            #print('Decoder input:\n', dec_input)

    total_loss = (loss / float(targets.shape[1]))
    #print('Constructing captions done.')
    # Update step
    if train_flag:
        trainable_variables = encoder.trainable_variables + attention_module.trainable_variables + \
                              decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


def training(train_ds_meta, valid_ds_meta, tokenizer, encoder, attention_module, decoder):

    #print('Shape train-ds:', train_ds_meta)
    #print('Shape valid-ds:', valid_ds_meta)
    #print('Valid dataset:')
    #for element in valid_ds_meta:
    #    print(element)
    #print('End valid dataset.')

    num_train_examples = len(list(train_ds_meta))
    num_steps_train = num_train_examples // BATCH_SIZE

    num_val_examples = len(list(valid_ds_meta))
    num_steps_val = num_val_examples // BATCH_SIZE

    # Get Optimizer and loss Object
    optimizer = get_optimizer()
    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               attention=attention_module,
                               decoder=decoder,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint)

    loss_plot_train = []
    loss_plot_val = []
    min_validation_loss = 1000
    check_patience = 0

    # Shuffle data
    train_ds_meta = train_ds_meta.shuffle(buffer_size=num_train_examples,
                                          reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    valid_ds_meta = valid_ds_meta.shuffle(buffer_size=num_train_examples,
                                          reshuffle_each_iteration=False).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    # Start the training
    for epoch in range(start_epoch, EPOCHS):
        #print('EPOCH:', epoch)
        start = time.time()
        total_loss_train = 0
        total_loss_val = 0

        # TRAINING LOOP
        #print('##### TRAINING #####')
        #for img_paths, targets in train_ds_meta.__iter__():
        for (batch, (img_paths, targets)) in enumerate(train_ds_meta):
            #print('Epoch:', epoch, 'Batch:', batch)
            #print(img_paths)
            # Read in images from paths
            img_batch = load_image_batch(img_paths)
            #print('t - Shape image batch:', img_batch.shape)
            #print('t - Shape targets:', targets.shape)
            # Perform training on one image
            batch_loss, t_loss = train_step(img_batch, targets, decoder, attention_module, encoder, tokenizer,
                                            optimizer, 1)  # 1 - weights trainable
            total_loss_train += t_loss
        #print('##### END TRAINING #####')
        loss_plot_val.append(total_loss_train / num_steps_train)
        print('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                            total_loss_train / num_steps_train))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

        #print('##### VALIDATION #####')
        # VALIDATION LOOP
        for (batch, (img_paths, targets)) in enumerate(valid_ds_meta):
            #print('Batch:', batch)
            img_batch = load_image_batch(img_paths)
            #print('Shape image batch:', img_batch.shape)
            #print('Shape targets:', targets.shape)
            batch_loss, t_loss = train_step(img_batch, targets, decoder, attention_module, encoder, tokenizer,
                                            optimizer, 0)  # 0 - weights not trainable
            total_loss_val += t_loss

        val_loss = total_loss_val / num_steps_val
        #print('##### END VALIDATION #####')
        loss_plot_val.append(val_loss)
        print('Epoch {} Validation Loss {:.6f}\n'.format(epoch + 1,
                                                         val_loss))
        if val_loss < min_validation_loss:
            min_validation_loss = val_loss
            check_patience = 0
        else:
            check_patience = check_patience + 1
        if check_patience > Patience:
            break
    return loss_plot_train, loss_plot_val
