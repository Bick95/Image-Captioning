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
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate import bleu
import matplotlib.pyplot as plt
from matplotlib.image import imread
import statistics
#def evaluate(image, tokenizer, max_length, decoder, encoder):

def evaluate(test_data,encoder,decoder, max_length, tokenizer, image_features_extract_model):
    images = test_data[0]
    captions = test_data[1]
    smoothie = SmoothingFunction().method4
    score1 = []  # weights - 0.25 0.25 0.25 0.25
    score2 = []  # weights - 0    0.33 0.33 0.33
    score3 = []  # weights - 0  0.5 0.5 0
    for image, caption in zip(images, captions):
        real_caption = ' '.join([tokenizer.index_word[i] for i in caption if i not in [0]])
        hidden = decoder.reset_state(batch_size=1)
        temp_input = tf.expand_dims(load_image(image)[0], 0)
        #image_features_extract_model = img_extract_model()
        img_tensor_val = image_features_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
        features = encoder(img_tensor_val)
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        result = []
        for i in range(max_length):
            #predictions, hidden = decoder(dec_input, features, hidden)
            predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            result.append(tokenizer.index_word[predicted_id])
            if tokenizer.index_word[predicted_id] == '<end>':
                result.insert(0,"<start>")
                result = " ".join(result)
                score1.append(bleu([real_caption], result, smoothing_function=smoothie,weights=(0.25, 0.25, 0.25, 0.25)))
                score2.append(bleu([real_caption], result, smoothing_function=smoothie, weights=(0, 0.33, 0.33, 0.33)))
                score3.append(bleu([real_caption], result, smoothing_function=smoothie, weights=(0, 0.5, 0.5, 0)))

                print("Score1 \n",bleu([real_caption], result, smoothing_function=smoothie,weights=(0.25, 0.25, 0.25, 0.25)))
                print("Score2 \n",bleu([real_caption], result, smoothing_function=smoothie, weights=(0, 0.33, 0.33, 0.33)))
                print("Score3 \n", bleu([real_caption], result, smoothing_function=smoothie, weights=(0, 0., 0.5, 0)))
                print("Image name",image)
                print("Real caption",real_caption)
                print("Result",result)
                print("Scores are ",score1, score2, score3)
                break
            dec_input = tf.expand_dims([predicted_id], 0)
    score1.append(bleu([real_caption], result, smoothing_function=smoothie, weights=(0.25, 0.25, 0.25, 0.25)))
    score2.append(bleu([real_caption], result, smoothing_function=smoothie, weights=(0, 0.33, 0.33, 0.33)))
    score3.append(bleu([real_caption], result, smoothing_function=smoothie, weights=(0, 0.5, 0.5, 0)))
    print("Real caption outside", real_caption)
    print("Result outside", result)
    print("Score outside", score1, score2, score3)
    return statistics.mean(score1), statistics.mean(score2), statistics.mean(score3)

#Get the caption and the attention plot for the image
def get_plot_attention(image,encoder,decoder, max_length, tokenizer):
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
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])
        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot
        dec_input = tf.expand_dims([predicted_id], 0)
    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def plot_attention(image, result, attention_plot,count):
    #temp_image = np.array(Image.open(image))
    temp_image = imread(image)
    fig = plt.figure(figsize=(40, 40))
    len_result = len(result)
    # if(len_result%2==1):
    #     row_plot = (len_result+1)//2
    #     col_plot = (len_result+1)//2
    #
    # else:
    #     row_plot=col_plot=(len_result+1)//2
    print("Results is ",result)
    try:
        for l in range(len_result):
                print("Length result and l",len_result,l)
                temp_att = np.resize(attention_plot[l], (8, 8))
                ax = fig.add_subplot((len_result+2)//2,(len_result+2)//2, l+1)
                ax.set_title(result[l])
                img = ax.imshow(temp_image)
                ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())
    except:
        print("Failed for result and count",result,count)
        #plt.tight_layout()
    plt.savefig("test" + str(count)+ ".png")

