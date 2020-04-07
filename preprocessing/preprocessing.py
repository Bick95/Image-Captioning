import tensorflow as tf
# import zip
import numpy as np
# from itertools import izip
from utils.utils import *
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from variables import plot_attention_idx_list


def _max_len_tensor(tensor):
    return max(len(t) for t in tensor)


def _tokenize_words(captions, tokenizer):
    tokenizer.fit_on_texts(captions)
    train_seqs = tokenizer.texts_to_sequences(captions)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    train_seqs = tokenizer.texts_to_sequences(captions)  # Redundant?
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
    max_length = _max_len_tensor(train_seqs)
    return cap_vector, train_seqs, max_length, tokenizer


def _get_image_caption_list(csv_file_path, images_path, debug):
    caption_list = []
    img_name_list = []
    last_img = ''
    with open(csv_file_path, 'r') as csvfile:
        data = csv.reader(csvfile, delimiter='|')
        for row in data:
            try:
                img_name = images_path + row[0]

                # Only one caption per image
                if not img_name == last_img:
                    last_img = img_name
                    caption = '<start> ' + row[2] + ' <end>'
                    img_name_list.append(img_name)
                    caption_list.append(caption)

            except IndexError:
                # Handle erroneous dataset entries
                print('Skipped: ', row)

            if not debug:
                continue
            if len(caption_list) == 150:
                break
    img_name_list = img_name_list[1:]  # 1st row contains column names.
    caption_list = caption_list[1:]  # 1st row contains column names.
    print('NUMBER CAPTIONS: ', len(caption_list))

    return img_name_list, caption_list


def get_meta_datasets(captions_file_path, images_path, tokenizer, data_split, debug):
    # Encode entire dataset
    img_name_list, captions_list = _get_image_caption_list(captions_file_path, images_path, debug)

    # Collect example captions and corresponding images for attention plotting
    plot_attention_img_list = [img_name_list[i] for i in plot_attention_idx_list]
    plot_attention_caption_list = [captions_list[i] for i in plot_attention_idx_list]

    # Remove all images & captions reserved for attention plotting from dataset
    demo_indices = [i for i, img_name in enumerate(img_name_list) if img_name in plot_attention_img_list]
    demo_indices = sorted(demo_indices, reverse=True)  # Delete later indices first
    for i in demo_indices:
        del img_name_list[i]
        del captions_list[i]

    print(plot_attention_caption_list)
    print(plot_attention_img_list)

    # Tokenize words
    cap_list, _, max_capt_len, tokenizer = _tokenize_words(captions_list,
                                                           tokenizer)  # cap_list = list of paddeded integer sequences representing captions

    # Shuffle captions and image names together
    cap_list, img_name_list = shuffle(cap_list,
                                      img_name_list,
                                      #random_state=1
                                      )

    # Split for training, validation, and testing
    test_val_percentage = (1. - data_split['train'])
    test_percentage = (1. - (data_split['test'] / test_val_percentage))

    print('Percentage training data:', data_split['train'])
    print('Percentage validat. data:', (test_val_percentage - data_split['test']))
    print('Percentage testing data: ', data_split['test'])

    # 1. Create training and test+validation sets using an 70-30 split
    img_names_train, img_names_val_test, caps_train, caps_val_test = train_test_split(img_name_list,
                                                                                      cap_list,
                                                                                      test_size=test_val_percentage,
                                                                                      #random_state=0
                                                                                      )

    # 2. Create validation and tetsing sets using an 50-50 split
    img_names_val, img_names_test, caps_val, caps_test = train_test_split(img_names_val_test,
                                                                          caps_val_test,
                                                                          test_size=test_percentage,
                                                                          #random_state=0
                                                                          )

    # Construct TF datasets containing image paths (i.e. not actual images, hence 'meta') and corresponding captions
    # Training data set
    train_ds_meta = tf.data.Dataset.from_tensor_slices((img_names_train, caps_train))

    # Validation data set
    valid_ds_meta = tf.data.Dataset.from_tensor_slices((img_names_val, caps_val))

    # Test data set
    test_ds_meta = tf.data.Dataset.from_tensor_slices((img_names_test, caps_test))
    print(list(test_ds_meta))
    return train_ds_meta, valid_ds_meta, test_ds_meta, max_capt_len, plot_attention_img_list, \
           plot_attention_caption_list
