import os, sys

sys.path.append(os.path.dirname(os.getcwd()))
import csv
from utils.utils import *
from preprocessing.preprocessing import *
import tensorflow as tf
from variables import csv_file_path, image_path, debug, max_words, embedding_dim, units, vocab_size, \
    plot_attention_idx_list
from training import training
from evaluation import evaluate, get_plot_attention, plot_attention, random_string
from Attention.modules import *
from Decoder.decoder import *
from Encoder.encoder import *
from datetime import datetime


def main():
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

    data_split = dict(train=0.70,
                      valid=0.15,
                      test=0.15)

    # train_ds_meta = (img_names_train = [img_path_1, ..., img_path_n], caps_train = [num_capt_1, ..., num_capt_n])
    train_ds_meta, valid_ds_meta, test_ds_meta, max_capt_len, \
        plot_attention_img_list, plot_attention_caption_list = get_meta_datasets(csv_file_path, image_path, tokenizer,
                                                                                 data_split, debug)

    # Dataset Exploration
    if debug:
        print('----------------------------------')
        lst = list(train_ds_meta)
        for (img_name_tensor, cap_tensor) in lst[:10]:
            print('NEW IMAGE:')
            print('Image:', (str(img_name_tensor.numpy()).split('/')[-1]).split('.')[0])
            caption = ' '.join([tokenizer.index_word[i] for i in cap_tensor.numpy()])
            print('Caption:', caption)
        print('-----------------------------------')

    # Encoding-Attention-Decoding Architecture
    encoder = InceptionEncoder(embedding_dim)
    attention_module = SoftAttention(units)
    decoder = RNNDecoder(embedding_dim, units, vocab_size)
    print('Done setting up model.')

    loss_plot_train, loss_plot_val, encoder, attention_module, decoder = training(train_ds_meta, valid_ds_meta,
                                                                                  tokenizer, encoder, attention_module,
                                                                                  decoder)

    print('Done training.')
    print('Evolution loss on training data:\n', loss_plot_train)
    print('Evolution loss on validation data:\n', loss_plot_val)

    bleu_score = evaluate(test_ds_meta, encoder, attention_module, decoder, max_capt_len, tokenizer)
    print('Done with evaluation.')

    print("Bleu score:", bleu_score)

    count = 0
    for img_path in plot_attention_img_list:
        result, attention_plot = get_plot_attention(img_path, encoder, attention_module, decoder, max_capt_len, tokenizer)
        plot_attention(img_path, result, attention_plot, count)
        count = count + 1

    # Save model(s)
    print('Going to save models.')
    # Construct identifier
    TIME_STAMP = datetime.now().strftime("_%Y_%d_%m__%H_%M_%S__%f_")
    model_id = TIME_STAMP + '_' + random_string()

    # Save
    encoder.save_weights(model_id + '_encoder_weights')
    attention_module.save_weights(model_id + '_attention_module')
    decoder.save_weights(model_id + '_decoder')

    print('Done.')


if __name__ == '__main__':
    main()
