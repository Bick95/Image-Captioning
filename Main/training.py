# Get access to parent directory
import os, sys

sys.path.append(os.path.dirname(os.getcwd()))
import tensorflow as tf
from utils.utils import load_image_batch, load_image
from variables import *
import time


def get_optimizer():
    return tf.keras.optimizers.Adam()


def get_loss_object():
    return tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')


def loss_function(real, pred):
    loss_object = get_loss_object()
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


@tf.function
def train_step(img_batch, targets, decoder, attention_module, encoder, tokenizer, optimizer, train_flag):
    loss = 0
    # Initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=targets.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * targets.shape[0], 1)

    print('Shape image batch:', img_batch.shape)
    print('Shape targets:', targets.shape)

    # Prediction step
    with tf.GradientTape() as tape:
        features = encoder(img_batch)
        print('Shape features:', features.shape)
        # Repeat, appending caption by one word at a time
        for i in range(1, targets.shape[1]):
            # Passing the features through the attention module and decoder
            context_vector, attention_weights = attention_module(features, hidden)
            print('Shape context vector:', context_vector.shape)
            predictions, hidden = decoder(dec_input, context_vector)  # FIXME: prediction of caption not as in paper

            loss += loss_function(targets[:, i], predictions)
            # Using teacher forcing
            dec_input = tf.expand_dims(targets[:, i], 1)

    total_loss = (loss / int(targets.shape[1]))

    # Update step
    if train_flag:
        trainable_variables = encoder.trainable_variables + decoder.trainable_variables + \
                              attention_module.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


def training(train_ds_meta, valid_ds_meta, tokenizer, encoder, attention_module, decoder):

    num_train_examples = len(list(train_ds_meta))
    num_steps = num_train_examples // BATCH_SIZE

    #num_val_examples = len(list(val_data))
    #num_steps_val = num_val_examples // BATCH_SIZE

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

    # Start the training
    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss_train = 0
        total_loss_val = 0

        # TRAINING LOOP
        train_ds_meta = train_ds_meta.shuffle(num_train_examples).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
        for (batch, (img_paths, targets)) in enumerate(train_ds_meta):
            # Read in images from paths
            img_batch = load_image_batch(img_paths)
            print(img_batch.shape)
            # Perform training on one image
            batch_loss, t_loss = train_step(img_batch, targets, decoder, attention_module, encoder, tokenizer,
                                            optimizer, 1)  # 1 - weights trainable
            total_loss_train += t_loss

        loss_plot_val.append(total_loss_train / num_steps_train)
        print('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                            total_loss_train / num_steps_train))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

        # VALIDATION LOOP
        for (batch, (img_batch, targets)) in enumerate(valid_ds_meta):
            img_batch = load_image_batch(img_paths)
            batch_loss, t_loss = train_step(img_batch, targets, decoder, attention_module, encoder, tokenizer,
                                            optimizer, 0)  # 0 - weights not trainable
            total_loss_val += t_loss

        val_loss = total_loss_val / num_steps_val
        loss_plot_val.append(val_loss)
        print('Epoch {} Validation Loss {:.6f}\n'.format(epoch + 1,
                                                         val_loss))
        if (val_loss < min_validation_loss):
            min_validation_loss = val_loss
            check_patience = 0
        else:
            check_patience = check_patience + 1
        if (check_patience > Patience):
            break
    return loss_plot_train, loss_plot_val
