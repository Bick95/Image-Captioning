# Get access to parent directory
import os, sys

sys.path.append(os.path.dirname(os.getcwd()))
import tensorflow as tf
from utils.utils import load_image_batch, load_image
from variables import BATCH_SIZE, EPOCHS, loss_function_choice, Patience, learning_rate
import time


def get_optimizer():
    return tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
                                    name='Adam')


def neg_log_likelihood(real_idx, pred_prob_dist):
    """
        Computed the respective negative log-likelihood for each batch element.
        :param real_idx: tensor of correct word token indices to be predicted per batch element
        :param pred_prob_dist: For each batch element, probability distribution over entire vocab giving probability for
                               selecting each of the available words in the vocab next
        :return: tensor of negative log-likelihood per batch element
    """
    # Construct list of enumerated ground-truth indices to retrieve predicted probs of correct classes/words
    batch_idx = [[tf.constant(i), x] for i, x in enumerate(real_idx)]
    # Extract probabilities for correct words (per batch-element)
    likelihood = tf.gather_nd(pred_prob_dist, batch_idx)
    likelihood = tf.add(likelihood, tf.constant([0.000000001]*likelihood.shape[0]))  # Avoid infinity loss in case of prob == 0.
    print('Likelihoods:', likelihood)
    #print('Division by:', tf.math.log(tf.constant(10, dtype=likelihood.dtype)))
    # Compute & return negative log10-likelihood per batch element
    nll = -tf.math.log(likelihood) / tf.math.log(tf.constant(10, dtype=likelihood.dtype))
    return nll


def get_loss_object():
    """
        :return:  Loss function
    """
    if loss_function_choice == 0:
        #    Info: https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy :
        #          "By default, we assume that y_pred encodes a probability distribution." -- from_logits=False
        #          reduction='none': Don't reduce from batch size to scalar average, but keep batch-elements separate
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

    elif loss_function_choice == 1:
        return neg_log_likelihood

    else:
        raise NotImplementedError('Requested Loss function not available.')


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
    #print('\n\nMask1:', mask)
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    #print('Mask2:', mask)
    #print('Unmasked loss:', loss_)
    loss_ *= mask

    predictions = tf.math.argmax(pred, axis=1)
    print('Predictions:\t', predictions)
    print('Real:\t\t', real)
    #print('LOSS MASKED:', loss_)
    mean_loss = tf.reduce_mean(loss_)
    #print('Mean loss:', mean_loss)

    return mean_loss


#@tf.function
def train_step(img_batch, targets, decoder, attention_module, encoder, tokenizer, optimizer, train_flag):
    loss = 0
    # Initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=targets.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * targets.shape[0], 1)

    # Prediction step
    with tf.GradientTape() as tape:
        # Clear gradients in case of remaining ones from eval pass
        tape.reset()

        features = encoder(img_batch)
        # Repeat, appending caption by one word at a time
        for i in range(1, targets.shape[1]):
            #print('Iteration:', i)
            # Passing the features through the attention module and decoder
            context_vector, attention_weights = attention_module(features, hidden)

            predictions, hidden = decoder(dec_input, hidden, context_vector)

            loss_addition = loss_function(targets[:, i], predictions)
            loss += loss_addition
            #print('Loss-addition:', loss_addition, 'Loss after:', loss)

            if train_flag:
                # Using teacher forcing during training
                dec_input = tf.expand_dims(targets[:, i], 1)
            else:
                # Use predictions of previous word produced by network per batch element during eval
                dec_input = tf.expand_dims(tf.math.argmax(predictions, axis=1), 1)

            # Save unnecessary forward-passes if all captions are done
            if tf.math.reduce_sum(targets[:, i], axis=0) == 0:
                break

    print('Batch Loss:', loss.numpy())
    total_loss = loss  # loss == average loss over minibatch     #outtake: (loss / float(targets.shape[1]))

    # Update step
    if train_flag:
        trainable_variables = encoder.trainable_variables + attention_module.trainable_variables + \
                              decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


def training(train_ds_meta, valid_ds_meta, tokenizer, encoder, attention_module, decoder):

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
        print('##### ##### #####\n', '##### EPOCH: #####\n', epoch, '\n##### ##### #####')
        start = time.time()
        total_loss_train = 0
        total_loss_val = 0

        # TRAINING LOOP
        for (batch, (img_paths, targets)) in enumerate(train_ds_meta):
            print('##### Epoch: #####', epoch, 'Batch:', batch)
            # Read in images from paths
            img_batch = load_image_batch(img_paths)
            # Perform training on one image
            batch_loss, t_loss = train_step(img_batch, targets, decoder, attention_module, encoder, tokenizer,
                                            optimizer, 1)  # 1 - weights trainable & teacher forcing
            total_loss_train += t_loss
        loss_plot_train.append(total_loss_train / num_steps_train)
        print('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                            total_loss_train / num_steps_train))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

        # VALIDATION LOOP
        for (batch, (img_paths, targets)) in enumerate(valid_ds_meta):
            img_batch = load_image_batch(img_paths)
            batch_loss, t_loss = train_step(img_batch, targets, decoder, attention_module, encoder, tokenizer,
                                            optimizer, 0)  # 0 - weights not trainable & no teacher forcing
            total_loss_val += t_loss

        val_loss = total_loss_val / num_steps_val
        loss_plot_val.append(val_loss)
        print('Epoch {} Validation Loss {:.6f}\n'.format(epoch + 1, val_loss))

        if epoch % 1 == 0:
            ckpt_manager.save()

        if val_loss < min_validation_loss:
            min_validation_loss = val_loss
            check_patience = 0
        else:
            check_patience = check_patience + 1
        if check_patience > Patience:
            break

    return loss_plot_train, loss_plot_val, encoder, attention_module, decoder
