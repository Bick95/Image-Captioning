# Get access to parent directory
import os, sys

sys.path.append(os.path.dirname(os.getcwd()))

# Imports
import tensorflow as tf


class InceptionEncoder(tf.keras.Model):
    def __init__(self, embedding_dim, conv_trainable=False):
        super(InceptionEncoder, self).__init__()

        ## Convolutional NN (Inception-based)
        self.features_extract_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        self.features_extract_model.trainable = conv_trainable


    def call(self, img_batch):
        features = self.features_extract_model(img_batch)
        features = tf.reshape(features, (img_batch.shape[0], features.shape[1] * features.shape[2], features.shape[3]))
        return features
