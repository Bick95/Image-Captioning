# Get access to parent directory
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
import tensorflow as tf
from utils.utils import *
from variables import *
import time

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

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
def train_step(img_tensor, target, decoder, encoder, tokenizer, optimizer, flag):
  loss = 0
  # initializing the hidden state for each batch
  # because the captions are not related from image to image
  hidden = decoder.reset_state(batch_size=target.shape[0])
  dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)
  with tf.GradientTape() as tape:
      features = encoder(img_tensor)
      for i in range(1, target.shape[1]):
          # passing the features through the decoder
          predictions, hidden, _ = decoder(dec_input, features, hidden)
          loss += loss_function(target[:, i], predictions)
          # using teacher forcing
          dec_input = tf.expand_dims(target[:, i], 1)
  total_loss = (loss / int(target.shape[1]))
  if flag:
        trainable_variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
  return loss, total_loss


def training(encoder, decoder, train_data, val_data, tokenizer):
    num_train_examples = len(list(train_data))
    num_steps_train = num_train_examples // BATCH_SIZE

    num_val_examples = len(list(val_data))
    num_steps_val = num_val_examples // BATCH_SIZE

    #Get the data ready for training
    # Use map to load the numpy files in parallel for train data
    train_data = train_data.map(lambda item1, item2: tf.numpy_function(
        map_func, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch the training data
    train_data = train_data.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Use map to load the numpy files in parallel for val data
    val_data = val_data.map(lambda item1, item2: tf.numpy_function(
        map_func, [item1, item2], [tf.float32, tf.int32]),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch the Validation data
    val_data = val_data.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    val_data = val_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    #Get Optimizer and loss Object
    optimizer = get_optimizer()
    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint)

    loss_plot_train = []
    loss_plot_val   = []
    min_validation_loss = 1000
    check_patience      = 0
    #Start the training
    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss_train = 0
        total_loss_val   = 0

        #TRAINING LOOP
        for (batch, (img_tensor, target)) in enumerate(train_data):
            batch_loss, t_loss = train_step(img_tensor, target, decoder, encoder, tokenizer, optimizer, 1) #1 - weights trainable
            total_loss_train += t_loss

        loss_plot_val.append(total_loss_train / num_steps_train)
        print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                             total_loss_train / num_steps_train))
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

        #VALIDATION LOOP
        for (batch, (img_tensor, target)) in enumerate(val_data):
            batch_loss, t_loss = train_step(img_tensor, target, decoder, encoder, tokenizer, optimizer, 0) #0 - weights not trainable
            total_loss_val += t_loss

        val_loss = total_loss_val/num_steps_val
        loss_plot_val.append(val_loss)
        print ('Epoch {} Validation Loss {:.6f}\n'.format(epoch + 1,
                                                        val_loss))
        if (val_loss< min_validation_loss):
            min_validation_loss = val_loss
            check_patience = 0
        else:
            check_patience = check_patience + 1
        if(check_patience > Patience):
            break



    return loss_plot_train, loss_plot_val











