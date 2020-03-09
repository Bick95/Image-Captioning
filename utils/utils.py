import tensorflow as tf
from variables import *

def _load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    return img

def load_image_batch(image_paths):
    batch = tf.zeros([BATCH_SIZE, 299, 299, 3], tf.float32)
    for idx, path in enumerate(image_paths):
        batch[idx, :, :, :] = _load_image(path)
    return batch





