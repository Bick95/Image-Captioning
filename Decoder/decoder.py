# Get access to parent directory
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
# Imports
import tensorflow as tf
from Attention.modules import *
from Main.variables import attention_mode, SOFT_ATTENTION, HARD_ATTENTION

class RNNDecoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNNDecoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        if attention_mode == HARD_ATTENTION:
            self.attention = HardAttention(units)
        elif attention_mode == SOFT_ATTENTION:
            self.attention = SoftAttention(units)
        else:
            raise NotImplementedError('Requested attention module doesn\'t exist!')

        self.attention_weights = None
        self.attention_location = None
        self.gt_likelihoods = []

    def call(self, x, features, hidden, train_flag):
        if attention_mode == HARD_ATTENTION:
            context_vector, attention_weights, self.attention_location = self.attention(features, hidden, train_flag)
            self.attention_weights = attention_weights  # Book-keeping
        else:
            context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        #x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)
        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))
        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        # Produce probability distribution over vocab
        x = tf.nn.softmax(x, axis=1)  # size=(batch_size, vocab_length+1)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        self.attention_weights = None
        self.attention_location = None
        self.gt_likelihoods = []
        return tf.zeros((batch_size, self.units))

