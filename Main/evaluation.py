# Get access to parent directory
import os, sys

sys.path.append(os.path.dirname(os.getcwd()))
from Decoder.decoder import *
from Encoder.encoder import *
from preprocessing.preprocessing import *
from variables import *
import tensorflow as tf
import numpy as np
from PIL import Image
from preprocessing.preprocessing import *
from utils.utils import *
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
import math

# Unique Naming
from datetime import datetime
import random
import string


def random_string(length=10):
    """
        Generate a random string of given length. For safely storing produced images.
    """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def evaluate(test_ds_meta, encoder, attention_module, decoder, max_length, tokenizer):
    score = 0

    for (idx, (img_path, caption)) in enumerate(test_ds_meta):
        caption = caption.numpy()[1:]  # Convert to numpy array abd remove start token
        print('Idx:', idx)
        print('Eval Img path:', img_path)
        real_caption = ' '.join([tokenizer.index_word[i] for i in caption])
        print('Eval Caption:', caption)
        print('Eval Caption:', real_caption)
        hidden = decoder.reset_state(batch_size=1)  # Initial hidden state

        # Get image (pseudo-batch of 1 element)
        pseudo_img_batch = tf.expand_dims(load_image(img_path), 0)

        # Forward pass through encoder
        features = encoder(pseudo_img_batch)
        test_num_capt, test_num_capt_clip = [], []
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        result = []
        for i in range(max_length):
            # Passing the features through the attention module and decoder
            context_vector, attention_weights = attention_module(features, hidden)
            predictions, hidden = decoder(dec_input, hidden, context_vector)
            #print('Predictions:', predictions)
            predicted_id = tf.math.argmax(predictions, axis=1).numpy()[0]
            #print('Predicted id:', predicted_id)
            test_num_capt.append(predicted_id)
            #predicted_id = min(predicted_id, max_words)  # Avoid going out of bounds, which would cause exception
            #test_num_capt_clip.append(predicted_id)
            result.append(tokenizer.index_word[predicted_id])
            if tokenizer.index_word[predicted_id] == '<end>':
                score = score + sentence_bleu(real_caption, result, weights=(0, 0.5, 0.5, 0))
                break
            dec_input = tf.expand_dims([predicted_id], 0)
            #print('Dec input:', dec_input)
        print('Result:', result)
        print('Predicted w/o clipping:', test_num_capt)
        #print('Predicted wth clipping:', test_num_capt_clip)
    score = score + sentence_bleu(real_caption, result, weights=(0, 0.5, 0.5, 0))
    return score / len(list(test_ds_meta))


# Get the caption and the attention plot for the image
def get_plot_attention(img_path, encoder, attention_module, decoder, max_length, tokenizer):
    attention_plot = np.zeros((max_length, attention_features_shape))
    hidden = decoder.reset_state(batch_size=1)

    # Get image (pseudo-batch of 1 element)
    pseudo_img_batch = tf.expand_dims(load_image(img_path), 0)

    # Forward pass through encoder
    features = encoder(pseudo_img_batch)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []
    for i in range(max_length):
        # Passing the features through the attention module and decoder
        context_vector, attention_weights = attention_module(features, hidden)
        predictions, hidden = decoder(dec_input, hidden, context_vector)
        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()
        predicted_id = tf.math.argmax(predictions, axis=1).numpy()[0]
        result.append(tokenizer.index_word[predicted_id])
        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot
        dec_input = tf.expand_dims([predicted_id], 0)
    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def plot_attention(image, result, attention_plot, count):
    temp_image = np.array(Image.open(image))
    fig = plt.figure(figsize=(40, 40))
    len_result = len(result)
    square_sz = math.ceil(math.sqrt(len_result))
    print(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(square_sz, square_sz, l + 1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())
    # plt.tight_layout()
    now = datetime.now()
    TIME_STAMP = now.strftime("_%Y_%d_%m__%H_%M_%S__%f_")
    plt.savefig("test" + str(count) + TIME_STAMP + random_string() + "_" + ".png")
