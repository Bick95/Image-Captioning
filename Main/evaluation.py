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


def evaluate(test_ds_meta, encoder, attention_module, decoder, max_length, tokenizer):
    #images = test_data[0]
    #captions = test_data[1]
    score = 0
    #for image, caption in zip(images, captions):
    for (idx, (img_path, caption)) in enumerate(test_ds_meta):
        caption = caption.numpy()[1:-1]  # Convert to numpy array abd remove start token
        print('Idx:', idx)
        print('Eval Img path:', img_path)
        print('Eval Caption:', caption)
        real_caption = ' '.join([tokenizer.index_word[i] for i in caption])
        hidden = decoder.reset_state(batch_size=1)

        # Get image (pseudo-batch of 1 element)
        pseudo_img_batch = tf.expand_dims(load_image(img_path), 0)

        features = encoder(pseudo_img_batch)
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        result = []
        for i in range(max_length):
            # Passing the features through the attention module and decoder
            context_vector, attention_weights = attention_module(features, hidden)
            predictions, hidden = decoder(dec_input, context_vector)  # FIXME: prediction of caption not as in paper
            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            predicted_id = min(predicted_id, max_words)  # Avoid going out of bounds, which would cause exception
            result.append(tokenizer.index_word[predicted_id])
            if tokenizer.index_word[predicted_id] == '<end>':
                score = score + sentence_bleu(real_caption, result, weights=(0, 0.5, 0.5, 0))
                break
            dec_input = tf.expand_dims([predicted_id], 0)
        print('Result:', result)
    score = score + sentence_bleu(real_caption, result, weights=(0, 0.5, 0.5, 0))
    return score / len(list(test_ds_meta))


# Get the caption and the attention plot for the image
def get_plot_attention(image, encoder, attention_module, decoder, max_length, tokenizer):
    attention_plot = np.zeros((max_length, attention_features_shape))
    hidden = decoder.reset_state(batch_size=1)
    temp_input = tf.expand_dims(load_image(image)[0], 0)
    image_features_extract_model = img_extract_model()
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    features = encoder(img_tensor_val)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []
    for i in range(max_length):
        # Passing the features through the attention module and decoder
        context_vector, attention_weights = attention_module(features, hidden)
        predictions, hidden = decoder(dec_input, context_vector)  # FIXME: prediction of caption not as in paper
        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
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
    print(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(math.ceil(float(len_result) / 2.), len_result // 2, l + 1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())
    # plt.tight_layout()
    plt.savefig("test" + str(count) + ".png")
