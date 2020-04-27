import os, sys

sys.path.append(os.path.dirname(os.getcwd()))
import json
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


def get_folder_id():
    """
        Creates folder with unique ID in which everything related to a particular testrun can be saved.
    :return: Unique folder identifier
    """
    # Construct testrun identifier
    TIME_STAMP = datetime.now().strftime("%Y_%d_%m__%H_%M_%S__%f_")
    model_folder_id = TIME_STAMP + '_' + random_string() + '/'

    try:
        os.mkdir(model_folder_id)
    except Exception as e:
        print('Exception occurred: ', e)

    return model_folder_id


def main():
    model_folder_id = get_folder_id()
    print('Folder ID: ', model_folder_id)
    
    # Init tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

    # Define data-split-ratio
    data_split = dict(train=0.70,
                      valid=0.15,
                      test=0.15)

    # train_ds_meta = (img_names_train = [img_path_1, ..., img_path_n], caps_train = [num_capt_1, ..., num_capt_n])
    train_ds_meta, valid_ds_meta, test_ds_meta, max_capt_len, \
        plot_attention_img_list, plot_attention_caption_list = get_meta_datasets(csv_file_path, image_path, tokenizer,
                                                                                 data_split, debug)

    print([tokenizer.index_word[i] for i in range(max_words)])
    print('VOCAB FULL!!!')

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

    # Train
    train_total_total_data_loss, train_total_total_avg_data_loss, \
    train_total_total_reg_loss, train_total_total_loss, \
    train_avg_total_data_loss, train_avg_total_avg_data_loss, \
    train_avg_total_reg_loss, train_avg_total_loss, \
    eval_total_total_data_loss, eval_total_total_avg_data_loss, \
    eval_total_total_reg_loss, eval_total_total_loss, \
    eval_avg_total_data_loss, eval_avg_total_avg_data_loss, \
    eval_avg_total_reg_loss, eval_avg_total_loss, \
    encoder, attention_module, decoder = training(train_ds_meta, valid_ds_meta,
                                                  tokenizer, encoder, attention_module,
                                                  decoder, model_folder_id)

    print('Done training.')
    print('Evolution loss on training data:\n', train_avg_total_loss)
    print('Evolution loss on validation data:\n', eval_avg_total_loss)

    # Evaluate
    bleu_score = evaluate(test_ds_meta, encoder, attention_module, decoder, max_capt_len, tokenizer, model_folder_id)
    print('Done with evaluation.')

    print("Bleu score:", bleu_score)

    # Generate example images
    count = 0
    for img_path in plot_attention_img_list:
        result, attention_plot = get_plot_attention(img_path, encoder, attention_module, decoder, max_capt_len, tokenizer)
        plot_attention(img_path, result, attention_plot, count, model_folder_id)
        count = count + 1

    # Save stats
    print('Going to save stats.')
    stats = {'bleu': bleu_score,
             'train_total_total_data_loss': list(train_total_total_data_loss),
             'train_total_total_avg_data_loss': list(train_total_total_avg_data_loss),
             'train_total_total_reg_loss': list(train_total_total_reg_loss),
             'train_total_total_loss': list(train_total_total_loss),
             'train_avg_total_data_loss': list(train_avg_total_data_loss),
             'train_avg_total_avg_data_loss': list(train_avg_total_avg_data_loss),
             'train_avg_total_reg_loss': list(train_avg_total_reg_loss),
             'train_avg_total_loss': list(train_avg_total_loss),
             'eval_total_total_data_loss': list(eval_total_total_data_loss),
             'eval_total_total_avg_data_loss': list(eval_total_total_avg_data_loss),
             'eval_total_total_reg_loss': list(eval_total_total_reg_loss),
             'eval_total_total_loss': list(eval_total_total_loss),
             'eval_avg_total_data_loss': list(eval_avg_total_data_loss),
             'eval_avg_total_avg_data_loss': list(eval_avg_total_avg_data_loss),
             'eval_avg_total_reg_loss': list(eval_avg_total_reg_loss),
             'eval_avg_total_loss': list(eval_avg_total_loss),
             'data_split': data_split,
             'debug': debug
             }

    print(stats)

    with open(model_folder_id + 'stats.txt', 'w') as outfile:
        json.dump(stats, outfile)

    # Save model(s)
    print('Going to save models.')

    # Save
    encoder.save_weights(model_folder_id + 'final_model/encoder_weights')
    attention_module.save_weights(model_folder_id + 'final_model/attention_module')
    decoder.save_weights(model_folder_id + 'final_model/decoder')

    print('Done.')


if __name__ == '__main__':
    main()
