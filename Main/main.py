import os, sys

sys.path.append(os.path.dirname(os.getcwd()))
import csv
from utils.utils import *
from preprocessing.preprocessing import get_meta_datasets
import tensorflow as tf
from variables import embedding_dim, units, vocab_size, debug, \
                      max_words, captions_file_path, images_path
from training import training
from evaluation import evaluate
from Attention.modules import *
from Decoder.decoder import *
from Encoder.encoder import *


def main():
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    # Encoding-Attention-Decoding Architecture
    encoder = InceptionEncoder(embedding_dim)
    attention = SoftAttention(units)
    decoder = RNNDecoder(embedding_dim, units, vocab_size)

    data_split = dict(train=0.70,
                      valid=0.15,
                      test=0.15)
	
	# train_ds_meta = (img_names_train = [img_path_1, ..., img_path_n], caps_train = [num_capt_1, ..., num_capt_n])
    train_ds_meta, valid_ds_meta, test_ds_meta, max_length = get_meta_datasets(captions_file_path, images_path, tokenizer,
                                                                  max_words, data_split, debug)

	# TODO: double-check if the encoder, decoder, ... really don't need re-assignment after training. But for tokenizer it also seems to work like that. 
    training(train_ds_meta, valid_ds_meta, tokenizer, encoder, attention, decoder)

    result, attention_plot = evaluate(test_ds_meta, tokenizer, max_length, encoder, attention, decoder)

    #print('Real Caption:', real_caption) # TODO: where does that come from? There is not just 1, but 5 possible ones...
    print('Prediction Caption:', ' '.join(result))


if __name__ == '__main__':
    main()
