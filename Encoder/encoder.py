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
        print('Shape batch:', img_batch.shape)
        features = self.features_extract_model(img_batch)
        print('Shape extracted features:', features.shape)
        features = tf.reshape(features, (img_batch.shape[0], features.shape[1] * features.shape[2], features.shape[3]))
        print('Shape features after reshape:', features.shape)
        features = self.fc(features)
        features = tf.nn.relu(features)
        print('Shape features:', features.shape)
        print('Inception module start')
        return features
