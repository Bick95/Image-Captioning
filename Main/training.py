# Get access to parent directory
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
import tensorflow as tf
# Import own modules
from attention.modules import SoftAttention, HardAttention
from Decoder.rnn_decoder import *
from Encoder.encoder import *
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
def train_step(img_tensor, target, decoder, encoder, tokenizer, optimizer):
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
  trainable_variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, trainable_variables)
  optimizer.apply_gradients(zip(gradients, trainable_variables))
  return loss, total_loss

def training(img_name_train, img_name_val, cap_train, cap_val,tokenizer):
    num_steps = len(img_name_train + img_name_val) // BATCH_SIZE
    #Encoding Decoding Architecture
    encoder = InceptionEncoder(embedding_dim)
    decoder = RNNDecoder(embedding_dim, units, vocab_size)
    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
    #Get the data ready for training
    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
        map_func, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Shuffle and batch
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
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
    loss_plot = []
    #Start the training
    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0
        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(img_tensor, target, decoder, encoder, tokenizer, optimizer)
            total_loss += t_loss
            # if batch % 100 == 0:
            #     print ('Epoch {} Batch {} Loss {:.4f}'.format(
            #         epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)
        if epoch % 5 == 0:
            ckpt_manager.save()
        print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                             total_loss / num_steps))
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    return decoder, encoder











