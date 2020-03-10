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
from utils.utils import load_image

def evaluate(test_meta_ds, tokenizer, max_length, encoder, attention, decoder):
	
	# TODO: Scale to multiple tests
	ds_list = list(test_meta_ds.as_numpy_iterator())
	img_paths = [x[0] for x in ds_list]
	captions  = [x[1] for x in ds_list]

	# Get first img path + caption
	img_path = img_paths[0]
	caption  = captions[0]  # one of 5 ground truths per images

	batch_size = 1

	attention_plot = np.zeros((max_length, attention_features_shape))

	# Initialize
	hidden = decoder.reset_state(batch_size=batch_size) # Get initial hidden state
	image = tf.expand_dims(load_image(img_path), 0) # Load image
	
	# Feed image through network
	features = encoder(image)
	print(type(features))
	#features = tf.reshape(features, (features.shape[0], -1, features.shape[3]))
	#print(features.shape)

	dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
	result = []

	for i in range(max_length):
		context_vector, attention_weights = attention(features, hidden)
		predictions, hidden = decoder(dec_input, context_vector)

		attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

		predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
		result.append(tokenizer.index_word[predicted_id])

		if tokenizer.index_word[predicted_id] == '<end>':
			return result, attention_plot

		dec_input = tf.expand_dims([predicted_id], 0)

	attention_plot = attention_plot[:len(result), :]
	return result, attention_plot

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

