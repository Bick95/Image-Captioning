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

#def evaluate(image, tokenizer, max_length, decoder, encoder):
def evaluate(test_data,encoder,decoder, max_length, tokenizer):
    images = test_data[0]
    captions = test_data[1]
    score = 0
    print("tokenizer",tokenizer)
    for image, caption in zip(images, captions):
        real_caption = ' '.join([tokenizer.index_word[i] for i in caption if i not in [0]])
        print("Image is ",image)
        print("Caption is ",caption)
        hidden = decoder.reset_state(batch_size=1)
        temp_input = tf.expand_dims(load_image(image)[0], 0)
        image_features_extract_model = img_extract_model()
        img_tensor_val = image_features_extract_model(temp_input)
        print(type(img_tensor_val))
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
        print(img_tensor_val.shape)
        features = encoder(img_tensor_val)
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        result = []
        for i in range(max_length):
            predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            print("Predicted id is ",predicted_id)
            result.append(tokenizer.index_word[predicted_id])

            if tokenizer.index_word[predicted_id] == '<end>':

                score = score + sentence_bleu(real_caption, result, weights=(0, 0.5, 0.5, 0))
                break

            dec_input = tf.expand_dims([predicted_id], 0)

    score = score + sentence_bleu(real_caption, result, weights=(0, 0.5, 0.5, 0))

    return score/len(images)


'''
def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()
'''

