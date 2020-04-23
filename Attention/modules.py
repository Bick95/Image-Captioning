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

    def __init__(self, units, max_caption_len):
        """
            units:      number of internal units per layer
        """
        super(HardAttention, self).__init__()

        print('\nGoing to use HardAttention model!\nMake sure to use train_mode==False where required when predicting!\n')

        self.feature_weights = tf.keras.layers.Dense(units)
        self.hidden_weights = tf.keras.layers.Dense(units)
        self.scoring_weights = tf.keras.layers.Dense(1)
        self.b = 1.
        self.lambda_r = self.lambda_h = 0.4
        self.caption_len = max_caption_len

    def loss(self, word_label_indices, batch_decoder_output, batch_attention_output):
        """
            :param word_label_indices: Indices of ground-truth work tokens to be predicted in current iteration
            :param batch_decoder_output: Probability of generating most probable word given feature vector and
                                         location selected by attention module (per batch element).
                                         == Softmax output of decoder
            :param batch_attention_output: Probgability of selecting most probable feature location to attend, as
                                           predicted by attention module (per batch element).
                                           == Softmax output of attention module
            :return: Average loss taken over entire batch; Computed for prediction of a single word in the caption
                     prediction sequence
        """

        ## Compute log-likelihood of predicting a word (per batch element)
        # Construct list of enumerated ground-truth indices to retrieve predicted probs of correct classes/words
        batch_idx = [[tf.constant(i), x] for i, x in enumerate(word_label_indices)]
        # Extract probabilities for correct words (per batch-element)
        likelihood = tf.gather_nd(batch_decoder_output, batch_idx)
        likelihood = tf.add(likelihood, tf.constant(
            [0.000000001] * likelihood.shape[0]))  # Avoid infinity loss in case of prob == 0.
        print('Likelihoods:', likelihood)
        # Compute log10-likelihood per batch element
        ll_decoder = tf.math.log(likelihood) / tf.math.log(tf.constant(10, dtype=likelihood.dtype))

        ## Compute log-likelihood of predicting a feature location (per batch element)
        # For each batch element, select highest probability value (i.e. for selected image location)
        likelihood = tf.math.reduce_max(batch_decoder_output, axis=1)
        likelihood = tf.add(likelihood, tf.constant(
            [0.000000001] * likelihood.shape[0]))  # Avoid infinity loss in case of prob == 0.
        print('Likelihoods locations:', likelihood)
        # Compute log10-likelihood per batch element
        ll_attention = tf.math.log(likelihood) / tf.math.log(tf.constant(10, dtype=likelihood.dtype))

        # All element-wise applications
        scale = tf.subtract(tf.math.scalar_mul(tf.constant(self.lambda_r), ll_decoder), tf.constant(self.b))
        term2 = tf.math.multiply(scale, ll_attention)

        term3 = 0

        sum_terms = tf.math.add(tf.math.add(ll_decoder, term2), term3)

        # Mean over batch
        mean_loss = tf.reduce_mean(sum_terms)

        # Mean over all words in caption: 1/N*sum(x) == sum((1/N).*x)
        mean_loss = tf.math.divide(mean_loss, self.caption_len)

        return mean_loss


    def call(self, features, hidden, train_mode=True):
        """
            features:   features observed from image, output of encoder,   shape: (batch_size, num_features, embedding_dim)
            hidden:     hidden state of the decoder network (RNN) from previous iteration, shape: (batch_size, hidden_size)
        """

        # hidden_expanded, shape: (batch_size, 1, hidden_size)
        hidden_expanded = tf.expand_dims(hidden, 1)

        # Calculate unnormalized Attention weights (=unnormal_attent_scores); shape: (batch_size, num_features, hidden_size)
        unnormal_attent_scores = tf.nn.tanh(self.feature_weights(features) + self.hidden_weights(hidden_expanded))

        # Normalize Attention weights to turn them into a probability-distribution (attention_probs_alpha); shape: (batch_size, num_features, 1)
        attention_probs_alpha = tf.nn.softmax(self.scoring_weights(unnormal_attent_scores), axis=1)

        # Select index of feature to attend, i.e. Attention location, and construct batch-context-vector
        context_vector_z = tf.zeros(shape=[1, features.shape[1], features.shape[2], features.shape[3]],
                                    dtype=tf.dtypes.float32)

        # 50% of the times act greedy, taking most probable location
        batch_greedy = tf.random.uniform(shape=[features.shape[0]], minval=0, maxval=2, dtype=tf.int32).numpy()

        # For each batch item in batch, act either greedy or stochastic
        for idx, act_greedy in enumerate(batch_greedy):

            # attention_location_s, shape = scalar
            if act_greedy or not train_mode:  # FIXME: correct to always choose highest probability during eval?
                # With 50% chance, set the sampled Attention location s to its expected value alpha - Not during eval!
                attention_location_s = tf.squeeze(tf.argmax(attention_probs_alpha[idx, :], axis=-1))
            else:
                # Select feature based on stochastic sampling from Multinoulli (categorical) distribution with probabilities attention_probs_alpha
                one_hot_selection = tensorflow_probability.distributions.Multinomial(total_count=1.,
                                                                                     probs=attention_probs_alpha)
                attention_location_s = tf.squeeze(tf.argmax(one_hot_selection[idx, :], axis=-1))

            # Construct context vector by selecting stochastically chosen feature to pay Attention to;
            # context_vector_z, shape after selection of feature: (batch_size, embedding_dim)
            context_vector_z = tf.concat([context_vector_z, features[idx, attention_location_s, :]], axis=0)

        return context_vector_z[1:], attention_probs_alpha  # Remove empty first element

