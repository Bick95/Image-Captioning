# Get access to parent directory
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

# Imports
import tensorflow as tf


class SoftAttention(tf.keras.Model):
    def __init__(self, units):
        """
            units:      number of internal units per layer
        """
        super(SoftAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        """
            features:   features observed from image
            hidden:     hidden state of the decoder network (RNN) from previous iteration
        """
        # print("INSIDE ATTENTION MODULE")
        # print("Features are in Attention",features.shape)
        # print("hidden ",hidden.shape)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        # print("Hidden with time axis ", hidden_with_time_axis.shape)
        # score shape == (batch_size, 64, hidden_size)

        features_W1 = self.W1(features)
        # print("Features W1",features_W1.shape)
        hidden_with_time_axis_W2 = self.W2(hidden_with_time_axis)
        # print("Hidden with time axis W2",hidden_with_time_axis_W2.shape)

        sum_check = features_W1+hidden_with_time_axis_W2
        # print("Sum check",sum_check.shape)
        score= tf.nn.tanh(sum_check)
        #score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        # print("Score is ",score.shape)
        # attention_weights shape == (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        # print("Attention weights ",attention_weights.shape)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        # print("Context vector",context_vector.shape)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        # print("Context Vector after reduce sum",context_vector.shape)
        return context_vector, attention_weights

class HardAttention(tf.keras.Model):
    
    # TODO 1: Define custom loss function?
    # TODO 2: Include running average b_k
    # TODO 3: Add entropy H[s]
    
    def __init__(self, units):
        """
            units:      number of internal units per layer
        """
        super(HardAttention, self).__init__()
        
        self.feature_weights = tf.keras.layers.Dense(units)
        self.hidden_weights = tf.keras.layers.Dense(units)
        self.attention_weights = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        """
            features:   features observed from image, output of encoder,   shape: (batch_size, num_features, embedding_dim) 
            hidden:     hidden state of the decoder network (RNN) from previous iteration, shape: (batch_size, hidden_size)
        """

        # hidden_expanded, shape: (batch_size, 1, hidden_size)
        hidden_expanded = tf.expand_dims(hidden, 1)

        # Calculate unnormalized Attention weights;
        # unnormal_attent_weights, shape: (batch_size, num_features, hidden_size)
        unnormal_attent_weights = tf.nn.tanh(self.feature_weights(features) + self.hidden_weights(hidden_expanded))
        
        # Normalize Attention weights to turn them into a probability-distribution;
        # attention_weights_alpha, shape: (batch_size, num_features, 1)
        attention_weights_alpha = tf.nn.softmax(self.attention_weights(unnormal_attent_weights), axis=1)
        
        # Select index of feature to attend, i.e. Attention location
        # attention_location_s, shape = scalar = ();
        if tf.squeeze(tf.argmax(tensorflow_probability.distributions.Multinomial(total_count=1., probs=[0.5,0.5]))) == 0:
            # With 50% chance, set the sampled Attention location s to its expected value alpha
            attention_location_s = tf.squeeze(tf.argmax(attention_weights_alpha, axis=-1))
            
        else:
            # Select feature based on stochastic sampling from Multinoulli (categorical) distribution with probabilities attention_weights_alpha
            one_hot_selection = tensorflow_probability.distributions.Multinomial(total_count=1., probs=attention_weights_alpha)
            attention_location_s = tf.squeeze(tf.argmax(one_hot_selection, axis=-1))
        
        
        # Construct context vector by selecting stochastically chosen feature to pay Attention to;
        # context_vector_z, shape after selection of feature: (batch_size, embedding_dim)
        context_vector_z = features[attention_location_s,:]

        return context_vector_z, attention_weights_alpha
    

