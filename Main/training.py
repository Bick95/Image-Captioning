# Get access to parent directory
import os, sys

sys.path.append(os.path.dirname(os.getcwd()))
import tensorflow as tf
from utils.utils import load_image_batch, load_image
from variables import BATCH_SIZE, EPOCHS, loss_function_choice, Patience, learning_rate, ckpt_frequency, \
                      attention_mode, SOFT_ATTENTION, HARD_ATTENTION
import time
import numpy as np


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
    likelihood = tf.add(likelihood, tf.constant([0.0000001]*likelihood.shape[0]))  # Avoid infinity loss in case of prob == 0.
    print('Likelihoods:', likelihood)
    # Compute & return negative log10-likelihood per batch element
    return -tf.math.log(likelihood)


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
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    predictions = tf.math.argmax(pred, axis=1)
    print('Predictions:\t', predictions)
    print('Real:\t\t', real)
    mean_loss = tf.reduce_mean(loss_)

    return mean_loss


#@tf.function
def train_step(img_batch, targets, decoder, attention_module, encoder, tokenizer, optimizer, train_flag):
    loss, reg_loss, data_loss = 0., 0., 0.
    # Initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=targets.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * targets.shape[0], 1)
    # For HardAttention:
    batch_likelihoods = []

    # Prediction step
    with tf.GradientTape() as tape:
        # Clear gradients in case of remaining ones from eval pass
        tape.reset()

        features = encoder(img_batch)
        # Repeat, appending caption by one word at a time
        for i in range(1, targets.shape[1]):
            #print('Iteration:', i)

            if attention_mode == SOFT_ATTENTION:
                # Passing the features through the attention module and decoder
                context_vector, attention_weights = attention_module(features, hidden)

                predictions, hidden = decoder(dec_input, hidden, context_vector)

                data_loss += loss_function(targets[:, i], predictions)

            elif attention_mode == HARD_ATTENTION:
                # Passing the features through the attention module and decoder
                context_vector, attention_weights, attention_location = attention_module(features, hidden)

                predictions, hidden = decoder(dec_input, hidden, context_vector)

                mean_loss, gt_likelihood = attention_module.loss(targets[:, i], predictions, attention_weights, attention_location)

                data_loss += mean_loss
                batch_likelihoods.append(gt_likelihood)

            reg_loss += (tf.math.reduce_sum(encoder.losses) +
                         tf.math.reduce_sum(attention_module.losses) +
                         tf.math.reduce_sum(decoder.losses))

            loss = data_loss + reg_loss  # Must be inside with-scope, otherwise trainable variables will not be found

            if train_flag:
                # Using teacher forcing during training
                dec_input = tf.expand_dims(targets[:, i], 1)
            else:
                # Use predictions of previous word produced by network per batch element during eval
                dec_input = tf.expand_dims(tf.math.argmax(predictions, axis=1), 1)

            # Save unnecessary forward-passes if all captions are done
            if tf.math.reduce_sum(targets[:, i], axis=0) == 0:
                break

    print('Data loss:', data_loss.numpy())
    print('Regu loss:', reg_loss.numpy())
    print('Combined loss:', loss.numpy())

    avg_data_loss = data_loss.numpy() / float(i)  # loss == average loss over max len of seen minibatcn

    # Update step
    if train_flag:
        trainable_variables = encoder.trainable_variables + attention_module.trainable_variables + \
                              decoder.trainable_variables
        if attention_mode == SOFT_ATTENTION:
            gradients = tape.gradient(loss, trainable_variables)
        elif attention_mode == HARD_ATTENTION:
            gradients = tape.gradient(loss / i, trainable_variables)
        else:
            gradients = None
        optimizer.apply_gradients(zip(gradients, trainable_variables))

    if attention_mode == HARD_ATTENTION:
        attention_module.update(batch_likelihoods, i)

    return data_loss, avg_data_loss, reg_loss, loss


def training(train_ds_meta, valid_ds_meta, tokenizer, encoder, attention_module, decoder, model_folder):

    num_train_examples = len(list(train_ds_meta))
    num_val_examples   = len(list(valid_ds_meta))

    # Get Optimizer
    optimizer = get_optimizer()

    # Get automatic checkpoint saver
    checkpoint_path = model_folder + "checkpoints/train"
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               attention=attention_module,
                               decoder=decoder,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        # Restoring the latest checkpoint in checkpoint_path if existent
        ckpt.restore(ckpt_manager.latest_checkpoint)

    # Extensive documentation of loss development
    train_total_total_data_loss, train_total_total_avg_data_loss, train_total_total_reg_loss, train_total_total_loss = \
        np.zeros([EPOCHS]), np.zeros([EPOCHS]), np.zeros([EPOCHS]), np.zeros([EPOCHS])
    train_avg_total_data_loss, train_avg_total_avg_data_loss, train_avg_total_reg_loss, train_avg_total_loss = \
        np.zeros([EPOCHS]), np.zeros([EPOCHS]), np.zeros([EPOCHS]), np.zeros([EPOCHS])

    eval_total_total_data_loss, eval_total_total_avg_data_loss, eval_total_total_reg_loss, eval_total_total_loss = \
        np.zeros([EPOCHS]), np.zeros([EPOCHS]), np.zeros([EPOCHS]), np.zeros([EPOCHS])
    eval_avg_total_data_loss, eval_avg_total_avg_data_loss, eval_avg_total_reg_loss, eval_avg_total_loss = \
        np.zeros([EPOCHS]), np.zeros([EPOCHS]), np.zeros([EPOCHS]), np.zeros([EPOCHS])

    min_validation_loss = float('inf')
    check_patience = 0

    # Shuffle data
    train_ds_meta = train_ds_meta.shuffle(buffer_size=num_train_examples,
                                          reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    valid_ds_meta = valid_ds_meta.shuffle(buffer_size=num_val_examples,
                                          reshuffle_each_iteration=False).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    # Start the training
    for epoch in range(start_epoch, EPOCHS):
        print('##### ##### #####\n', '##### EPOCH: #####\n', epoch, '\n##### ##### #####')
        start = time.time()
        total_data_loss = 0.
        total_avg_data_loss = 0.
        total_reg_loss = 0.
        total_loss = 0.

        # TRAINING LOOP
        for (batch, (img_paths, targets)) in enumerate(train_ds_meta):
            print('##### Epoch: #####', epoch, 'Batch:', batch)
            # Read in images from paths
            img_batch = load_image_batch(img_paths)
            # Perform training on one image
            data_loss, avg_data_loss, reg_loss, loss = train_step(img_batch, targets, decoder,
                                                                  attention_module, encoder, tokenizer,
                                                                  optimizer, 1)  # 1 - weights trainable & teacher forcing

            total_data_loss += data_loss
            total_avg_data_loss += avg_data_loss
            total_reg_loss += reg_loss
            total_loss += loss

        num_batches = float(batch + 1)

        train_total_total_data_loss[epoch] = total_data_loss
        train_total_total_avg_data_loss[epoch] = total_avg_data_loss
        train_total_total_reg_loss[epoch] = total_reg_loss
        train_total_total_loss[epoch] = total_loss

        train_avg_total_data_loss[epoch] = total_data_loss / num_batches
        train_avg_total_avg_data_loss[epoch] = total_avg_data_loss / num_batches
        train_avg_total_reg_loss[epoch] = total_reg_loss / num_batches
        train_avg_total_loss[epoch] = total_loss / num_batches

        print('Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

        # VALIDATION LOOP
        total_data_loss = 0.
        total_avg_data_loss = 0.
        total_reg_loss = 0.
        total_loss = 0.

        for (batch, (img_paths, targets)) in enumerate(valid_ds_meta):
            img_batch = load_image_batch(img_paths)
            data_loss, avg_data_loss, reg_loss, loss = train_step(img_batch, targets, decoder, attention_module,
                                                                          encoder, tokenizer, optimizer, 0)  # 0 - weights not trainable & no teacher forcing
            total_data_loss += data_loss
            total_avg_data_loss += avg_data_loss
            total_reg_loss += reg_loss
            total_loss += loss

        num_batches = float(batch + 1)

        # Documentation of evaluation stats
        eval_total_total_data_loss[epoch] = total_data_loss
        eval_total_total_avg_data_loss[epoch] = total_avg_data_loss
        eval_total_total_reg_loss[epoch] = total_reg_loss
        eval_total_total_loss[epoch] = total_loss

        eval_avg_total_data_loss[epoch] = total_data_loss / num_batches
        eval_avg_total_avg_data_loss[epoch] = total_avg_data_loss / num_batches
        eval_avg_total_reg_loss[epoch] = total_reg_loss / num_batches
        eval_avg_total_loss[epoch] = total_loss / num_batches
        print('Epoch {} Validation Loss {:.6f}\n'.format(epoch + 1, total_loss))

        # GENERATE CHECKPOINTS
        if epoch % ckpt_frequency == 0:
            ckpt_manager.save()

        # EARLY STOPPING IMPLEMENTATION
        if (total_loss / num_batches) < min_validation_loss:
            min_validation_loss = total_loss / num_batches
            check_patience = 0
        else:
            check_patience = check_patience + 1
        if check_patience > Patience:
            break

    return train_total_total_data_loss, train_total_total_avg_data_loss, \
           train_total_total_reg_loss, train_total_total_loss, \
           train_avg_total_data_loss, train_avg_total_avg_data_loss, \
           train_avg_total_reg_loss, train_avg_total_loss, \
           eval_total_total_data_loss, eval_total_total_avg_data_loss, \
           eval_total_total_reg_loss, eval_total_total_loss, \
           eval_avg_total_data_loss, eval_avg_total_avg_data_loss, \
           eval_avg_total_reg_loss, eval_avg_total_loss, \
           encoder, attention_module, decoder
