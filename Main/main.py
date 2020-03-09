import os, sys

sys.path.append(os.path.dirname(os.getcwd()))
import csv
from utils.utils import *
from preprocessing.preprocessing import get_meta_datasets
import tensorflow as tf
from variables import embedding_dim, units, vocab_size, dataset_path, debug, \
                      max_length, max_words, captions_file_path, images_path
from training import training
from evaluation import evaluate
from Attention.modules import *
from Decoder.decoder import *
from Encoder.encoder import *


def main():
    # img_list, caption_list = get_image_n_caption(csv_file_path, image_path, debug)
    # caption_vector, tokenizer, train_seqs = tokenize_words(max_words, caption_list)
    # max_length = max_len_tensor(train_seqs)
    # img_name_train, img_name_val, cap_train, cap_val = split(img_list, caption_vector) # TODO: reserve test data as well!

    # Encoding-Attention-Decoding Architecture
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    encoder = InceptionEncoder(embedding_dim)
    attention = SoftAttention(units)
    decoder = RNNDecoder(embedding_dim, units, vocab_size)

    data_split = dict(train=0.70,
                      valid=0.15,
                      test=0.15)

    train_ds_meta, valid_ds_meta, test_ds_meta = get_meta_datasets(captions_file_path, images_path, tokenizer,
                                                                  max_words, data_split, debug)

    training(train_ds_meta, valid_ds_meta, tokenizer, encoder, attention, decoder)

    result, attention_plot = evaluate(test_ds_meta, tokenizer, max_length, decoder, encoder)

    print('Real Caption:', real_caption)
    print('Prediction Caption:', ' '.join(result))


if __name__ == '__main__':
    main()
