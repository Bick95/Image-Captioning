import tensorflow as tf
from variables import *

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    return img

def load_image_batch(image_paths):
    batch = tf.zeros([1, 299, 299, 3], tf.float32) # Placeholder
    for idx, path in enumerate(image_paths):
        batch = tf.concat((batch, [load_image(path) / 255.]), axis=0)
    return batch[1:] # Remove first empty element






