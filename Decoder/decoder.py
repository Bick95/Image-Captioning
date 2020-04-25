# Get access to parent directory
import os, sys
import tensorflow as tf

sys.path.append(os.path.dirname(os.getcwd()))


class RNNDecoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNNDecoder, self).__init__()
        self.units = units  # Hidden state size

        #self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        #self.gru = tf.keras.layers.GRU(self.units,
        #                               return_sequences=True,
        #                               return_state=True,
        #                               recurrent_initializer='glorot_uniform')

        self.gru = tf.keras.layers.GRU( self.units,
                                        activation='tanh',                      # Default
                                        recurrent_activation='sigmoid',         # Default
                                        use_bias=True,                          # Default
                                        kernel_initializer='glorot_uniform',    # Default
                                        recurrent_initializer='glorot_uniform',
                                        bias_initializer='glorot_uniform',               # Default
                                        kernel_regularizer=tf.keras.regularizers.l2(l=0.01),                # Default
                                        recurrent_regularizer=tf.keras.regularizers.l2(l=0.01),             # Default
                                        bias_regularizer=None,                  # Default
                                        activity_regularizer=tf.keras.regularizers.l2(l=0.01),              # Default
                                        kernel_constraint=None,                 # Default
                                        recurrent_constraint=None,              # Default
                                        bias_constraint=None,                   # Default
                                        dropout=0.0,                            # Default
                                        recurrent_dropout=0.0,                  # Default
                                        implementation=2,                       # Default
                                        return_sequences=False,
                                        return_state=True,  # Hidden state
                                        go_backwards=False,                     # Default
                                        stateful=False,                         # Default
                                        unroll=False,                           # Default
                                        time_major=False,                       # Default
                                        reset_after=False
                                        )

        self.transform_Lo = tf.keras.layers.Dense(vocab_size, kernel_regularizer=tf.keras.regularizers.l2(0.2))
        self.embedding_Lh = tf.keras.layers.Dense(embedding_dim, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.embedding_Lz = tf.keras.layers.Dense(embedding_dim, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.embedding_E = tf.keras.layers.Dense(embedding_dim, input_shape=[vocab_size])

    def call(self, batch_prev_word, batch_prev_hidden, batch_context_vector):

        ey = self.embedding_E(batch_prev_word)  # size=(batch_size, embedding_dim)

        # Compare to Eqn (1) from 'Show, Attend, and Tell'
        x = tf.concat([tf.expand_dims(batch_context_vector, 1),
                       tf.expand_dims(ey, 1)],
                      axis=-1)  # size=(batch_size, 1, (sum of lengths...))

        # Passing the concatenated vector to the GRU
        output, new_hidden = self.gru(x, initial_state=batch_prev_hidden)  # batch_prev_hidden: size=(batch_size, self.units)

        #print('Shape new hidden state: ', new_hidden)

        # Compare to Eqn. (7) from 'Show, Attend, and Tell'
        lh = self.embedding_Lh(new_hidden)      # size=(batch_size, embedding_dim)

        lz = self.embedding_Lz(batch_context_vector)    # size=(batch_size, embedding_dim)

        x = ey + lh + lz                        # size=(batch_size, embedding_dim)

        x = self.transform_Lo(x)                # size=(batch_size, vocab_length+1)

        # Print statements to check whether softmax axis is applied correctly.
        #print('SUM TEST: ', tf.math.reduce_sum(x, axis=1))
        x = tf.nn.softmax(x, axis=1)            # size=(batch_size, vocab_length+1)
        #print('SUM TEST: ', tf.math.reduce_sum(x, axis=1))
        return x, new_hidden

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
