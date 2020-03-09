# Get access to parent directory
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

# Imports
import tensorflow as tf


class InceptionEncoder(tf.keras.Model):
    def __init__(self, embedding_dim, trainable=False):
        super(InceptionEncoder, self).__init__()

        image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                        weights='imagenet')
        image_model.trainable = trainable
        self.inception = image_model

    def call(self, x):
        features = self.inception(x)
        # if len(features.shape) == 4:
        features = tf.reshape(features, (1, features.shape[1] * features.shape[2], features.shape[3]))
        return features
