# Get access to parent directory
import os, sys

sys.path.append(os.path.dirname(os.getcwd()))

# Imports
import tensorflow as tf


class InceptionEncoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(InceptionEncoder, self).__init__()

        ## Convolutional NN (Inception-based)
        self.features_extract_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        self.features_extract_model.trainable = False

        # Test
        image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                        weights='imagenet')
        new_input = image_model.input
        hidden_layer = image_model.layers[-1].output
        image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
        self.features_extract_model = image_features_extract_model

        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, img_batch):
        print('Inception module start')
        print(img_batch.shape)
        features = self.features_extract_model(img_batch)
        print(features.shape)
        features = self.fc(features)
        features = tf.nn.relu(features)
        print(features.shape)
        print('Inception module start')
        return features
