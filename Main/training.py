# Get access to parent directory
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
import tensorflow as tf
# Import own modules
#from attention.modules import SoftAttention, HardAttention
#from Decoder.decoder import *
#from Encoder.encoder import *
from utils.utils import load_image_batch
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
def train_step(img_tensor, target, decoder, attention, encoder, tokenizer, optimizer):
  loss = 0
  # initializing the hidden state for each batch
  # because the captions are not related between images
  hidden = decoder.reset_state(batch_size=target.shape[0])
  dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)
  with tf.GradientTape() as tape:
      features = encoder(img_tensor)
      for i in range(1, target.shape[1]):
          # defining attention as a separate model
          context_vector, attention_weights = attention(features, hidden)
          # passing the context vector through the decoder
          predictions, hidden = decoder(dec_input, context_vector)  # TODO: Double-check working of decoder. Doing the right thing in terms of predictions???
          loss += loss_function(target[:, i], predictions)
          # using teacher forcing
          dec_input = tf.expand_dims(target[:, i], 1)
  total_loss = (loss / int(target.shape[1]))
  trainable_variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, trainable_variables)
  optimizer.apply_gradients(zip(gradients, trainable_variables))
  return loss, total_loss

def training(train_ds_meta, valid_ds_meta, tokenizer, encoder, attention, decoder):

    num_train_examples = len(list(train_ds_meta))
    num_steps = num_train_examples // BATCH_SIZE

    #Get Optimizer and loss Object
    optimizer = get_optimizer()
    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               attention=attention,
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
        train_ds_meta = train_ds_meta.shuffle(num_train_examples).batch(BATCH_SIZE)
        for (batch, (img_paths, target)) in enumerate(train_ds_meta):
            # Read in images from paths
            img_tensor = load_image_batch(img_paths)
            # Perform training on one image
            batch_loss, t_loss = train_step(img_tensor, target, decoder, attention, encoder, tokenizer, optimizer)
            total_loss += t_loss
            # if batch % 100 == 0:
            #     print ('Epoch {} Batch {} Loss {:.4f}'.format(
            #         epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)
        if epoch % 5 == 0:
            ckpt_manager.save()

        # TODO: make persistent - save list
        print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                             total_loss / num_steps))
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    #return decoder, encoder











