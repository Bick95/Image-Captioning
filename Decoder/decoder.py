# Get access to parent directory
import os, sys
import tensorflow as tf

sys.path.append(os.path.dirname(os.getcwd()))


class RNNDecoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNNDecoder, self).__init__()
        self.units = units  # Hidden state size

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)


    def call(self, prev_word, prev_hidden, context_vector):
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(prev_word)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, new_hidden = self.gru(x, initial_state=prev_hidden)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        # Produce probability distribution over vocab
        x = tf.nn.softmax(x, axis=1)            # size=(batch_size, vocab_length+1)
        
        return x, new_hidden


    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
