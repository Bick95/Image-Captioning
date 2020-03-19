import os, sys

sys.path.append(os.path.dirname(os.getcwd()))
import csv
from utils.utils import *
from preprocessing.preprocessing import *
import tensorflow as tf
from variables import csv_file_path, image_path, debug, max_words, embedding_dim, units, vocab_size, \
    plot_attention_idx_list
from training import training
from evaluation import evaluate, get_plot_attention, plot_attention
from Attention.modules import *
from Decoder.decoder import *
from Encoder.encoder import *


def main():
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

    # Read in all image(-path) names and corresponding captions
    img_list, caption_list = get_image_n_caption(csv_file_path, image_path, debug)

    # Collect example captions and corresponding images for attention plotting
    plot_attention_img_list = [img_list[i] for i in plot_attention_idx_list]
    plot_attention_caption_list = [caption_list[i] for i in plot_attention_idx_list]

    # Remove all images & captions reserved for attention plotting from dataset
    demo_indices = [i for i, img_name in enumerate(img_list) if img_name in plot_attention_img_list]
    demo_indices = sorted(demo_indices, reverse=True)  # Delete later indices first
    for i in demo_indices:
        del img_list[i]
        del caption_list[i]

    # print(plot_attention_caption_list)
    # print(plot_attention_img_list)

    # Do tokenization
    caption_vector, train_seqs, max_length, tokenizer = tokenize_words(caption_list, tokenizer)

    # Split data into training, validation, and test data
    split_ratio = dict(train=0.70,
                       valid=0.15,
                       test=0.15)
    train_data, val_data, test_data = split(img_list, caption_vector, split_ratio)
    # print("Example of test data",test_data[0][0],test_data[1][0,:])

    # Encoding-Attention-Decoding Architecture
    encoder = InceptionEncoder(embedding_dim)
    attention_module = SoftAttention(units)
    decoder = RNNDecoder(embedding_dim, units, vocab_size)

    loss_plot_train, loss_plot_val = training(encoder, attention_module, decoder, train_data, val_data, tokenizer)
    bleu_score = evaluate(test_data, encoder, attention_module, decoder, max_length, tokenizer)
    print("bleu score is ", bleu_score)
    count = 0
    for img in plot_attention_img_list:
        result, attention_plot = get_plot_attention(img, encoder, attention_module, decoder, max_length, tokenizer)
        plot_attention(img, result, attention_plot, count)
        count = count + 1


if __name__ == '__main__':
    main()
