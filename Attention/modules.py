# Get access to parent directory
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

# Imports
import tensorflow as tf
import tensorflow_probability as tfp


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
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class HardAttention(tf.keras.Model):

    def __init__(self, units):
        """
            units:      number of internal units per layer
        """
        super(HardAttention, self).__init__()

        print('\nGoing to use HardAttention model!\nMake sure to use train_mode==False where required when predicting!\n')

        self.W1 = tf.keras.layers.Dense(units, kernel_regularizer=None)
        self.W2 = tf.keras.layers.Dense(units, kernel_regularizer=None)
        self.V = tf.keras.layers.Dense(1, kernel_regularizer=None)  # None vs tf.keras.regularizers.l2(0.01)
        self.b = 0.
        self.lambda_r = 0.5
        self.lambda_e = 0.1

    def update(self, evol_gt_likelihoods, caption_len):
        """
        :param evol_gt_likelihoods: List of batches of batch likelihood tensors, where each
                                    batch likelihood tensor contains likelihood of correct class/word per batch element.
                                    One batch element for each predicted word during generation of a full caption.
        :param caption_len: Length of longest caption in mini-batch for which this update of b is to be calculates.
        :return: Updated running average b.
        """

        batch_size = evol_gt_likelihoods[0].shape[0]

        acc_likelihoods = tf.zeros(batch_size)

        for batch in evol_gt_likelihoods:
            acc_likelihoods += batch

        # Per batch element, average prob over caption
        avg_likelihoods = acc_likelihoods / caption_len

        # Mean over batch
        mean_likelihood = tf.reduce_mean(avg_likelihoods)

        # Update b
        self.b = 0.9 * self.b + 0.1 * tf.math.log(mean_likelihood)

        print('New b:', self.b)

    def shannon_entropy(self, batch_probs):
        """
            Given a probability per location to attend (per batch element), the Shannon Entropy is computed
            per batch element. For Shannen Entropy see: https://en.wikipedia.org/wiki/Entropy_(information_theory)
        :param batch_probs: For each batch element: Probability for each element from feature vector of attending that
                            feature
        :return: Shannon entropy per batch element
        """
        log_p = tf.math.log(batch_probs)
        product = tf.math.multiply(batch_probs, log_p)
        entropy_term = tf.math.reduce_sum(product, axis=1)
        return -entropy_term

    def loss(self, target_idx, decoder_output, attention_weights, attention_location):
        """
            Computes the loss per mini-batch.
        :param target_idx: For each word in mini-batch, the ground-truth class-label/word-index
        :param decoder_output: For each mini-batch-element, the probability distribution over vocab
        :param attention_weights: For each mini-batch element, the probability distribution over attention locations
                                  when predicting the current word
        :param attention_location: For each batch element, the location to focus on as selected by the attention module
        :return: Average mini-batch loss during prediction of one of the (single) words in caption
        """

        # Make sure to avoid log(0)
        decoder_output += 0.0000001
        attention_weights += 0.0000001

        # Collect probabilities for all batch elements of correct class/word
        gt_likelihood = tf.convert_to_tensor([decoder_output[i, x] for i, x in enumerate(target_idx)])

        # Compute log-likelihood of correct class
        log_likelihood = tf.math.log(gt_likelihood)

        # Collect probabilities for all selected attention locations of batch
        att_likelihood = tf.convert_to_tensor([attention_weights[i, x] for i, x in enumerate(attention_location)])
        log_att_likelihood = tf.math.log(att_likelihood)

        # Compute Shannon Entropy term
        entropy = self.shannon_entropy(attention_weights)

        # Putting it all together
        loss = log_likelihood + self.lambda_r * (log_likelihood - self.b) * log_att_likelihood + self.lambda_e * entropy

        # Compute mean over batch
        mean_loss = tf.reduce_mean(loss)

        # Take into account averaging of loss over caption length (=N)
        #mean_loss /= self.caption_len  # Done in train_step()

        print('\nGT Likelihoods:\n', gt_likelihood)         # Likelihoods of predicting ground-truth labels/word tokens
        predictions = tf.math.argmax(decoder_output, axis=1)
        print('Predictions:\t', predictions)                # Predicted word tokens
        print('Real:\t\t', target_idx)                      # Ground truth word tokens

        return mean_loss, gt_likelihood


    def call(self, features, hidden, train_mode=True):
        """
            features:   features observed from image, output of encoder,   shape: (batch_size, num_features, embedding_dim)
            hidden:     hidden state of the decoder network (RNN) from previous iteration, shape: (batch_size, hidden_size)
        """

        batch_size = features.shape[0]

        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        attention_weights_sqzd = tf.squeeze(attention_weights, axis=2)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = []
        selected_idx = []
        attention_location = None

        # Boolean indicating for which images to compute captions greedily
        greedy_caption = tf.random.categorical(tf.constant([[0.5, 0.5]]), batch_size)

        for sample in range(batch_size):
            if greedy_caption[0][sample] or not train_mode:
                # For 50% of images, construct captions greedily
                attention_location = tf.argmax(attention_weights_sqzd[sample])
                context_vector.append(features[sample, attention_location])
            else:
                # Sample caption location from Multinoulli distribution parameterized by computed attention weights
                one_hot_select = tfp.distributions.Multinomial(total_count=1., probs=attention_weights_sqzd[sample]).sample(1)[0]
                attention_location = tf.argmax(one_hot_select)
                context_vector.append(features[sample, attention_location])

        context_vector = tf.stack(context_vector)
        selected_idx.append(attention_location)

        return context_vector, attention_weights, selected_idx
