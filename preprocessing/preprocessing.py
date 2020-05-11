import tensorflow as tf
# import zip
import numpy as np
# from itertools import izip
from utils.utils import *
import csv
import random
from sklearn.utils import shuffle
#from sklearn.model_selection import train_test_split
from variables import captions_per_image

from variables import plot_attention_idx_list, num_captions


def _max_len_tensor(tensor):
    return max(len(t) for t in tensor)


def _tokenize_words(captions, tokenizer, init=False):
    if init:
        tokenizer.fit_on_texts(captions)
        tokenizer.word_index['<pad>'] = 0
        tokenizer.index_word[0] = '<pad>'
    train_seqs = tokenizer.texts_to_sequences(captions)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
    max_length = _max_len_tensor(train_seqs)
    return cap_vector, train_seqs, max_length, tokenizer


def _get_image_caption_list(csv_file_path, images_path, debug):
    caption_list = []
    img_name_list = []
    unique_images = []
    capt_ctr, last_added_img = 0, ''

    with open(csv_file_path, 'r') as csvfile:
        data = csv.reader(csvfile, delimiter='|')
        for row in data:
            try:
                img_name = images_path + row[0]
                caption = '<start> ' + row[2].lower() + ' <end>'

                # Restrict number of captions per image
                if last_added_img != img_name:
                    # New image, reset counter for captions per image
                    capt_ctr = 0
                    unique_images.append(img_name)
                if not debug or capt_ctr < captions_per_image:
                    # Add caption for image if not max number of captions per image exceeded yet
                    img_name_list.append(img_name)
                    caption_list.append(caption)
                    last_added_img = img_name
                    capt_ctr += 1
                else:
                    pass  # Only n captions per image
            except IndexError:
                # Handle erroneous dataset entries
                print('Skipped training example due to import error:\n', row)

            if debug and len(caption_list) == num_captions:
                break

    img_name_list = img_name_list[1:]  # 1st row contains column names.
    caption_list = caption_list[1:]  # 1st row contains column names.
    unique_images = unique_images[1:]  # 1st row contains column names.
    print('NUMBER CAPTIONS: ', len(caption_list))
    print('Len unique images:', len(unique_images))
    return img_name_list, caption_list, unique_images


def split_dataset(img_name_list, captions_list, major_partition, unique_images, train):
    """
        Given some set of (multi-)set of image names, corresponding captions and a split ratio, devide imput dataset
        into one major dataset (containing x% of the input data, where x = major_partition), and a
        minor dataset containing the rest.
        If train, then use all provided captions per image. Otherwise, only first caption per image.
    :param img_name_list:
    :param captions_list:
    :param major_partition:
    :param unique_images: Set of each image contained in the dataset. As opposed to img_name list, which contains
                          each image as many times as there are captions associated with it. For selecting subset of
                          unique image names to be added to minor dataset.
    :param train:
    :return:
    """

    # Split into train, eval, and test data
    dataset = zip(img_name_list, captions_list)

    # Split for training, validation, and testing
    test_val_percentage = (1. - major_partition)

    test_image_indices = random.sample(range(len(unique_images)), int(len(unique_images) * test_val_percentage))
    test_image_indices = sorted(test_image_indices, reverse=False)

    images_major, captions_major = [], []
    images_minor, captions_minor = [], []
    unique_minors = [unique_images[x] for x in test_image_indices]
    unique_minors_cpy = unique_minors.copy()

    for tpl in dataset:
        if tpl[0] in unique_minors:
            if len(images_minor) == 0 or images_minor[-1] != tpl[0]:
                images_minor.append(tpl[0])
                captions_minor.append(tpl[1])
            else:
                unique_minors.remove(tpl[0])  # For efficiency, shorten list to search through
        else:
            if len(images_major) == 0 or images_major[-1] != tpl[0] or train:
                images_major.append(tpl[0])
                captions_major.append(tpl[1])
            else:
                pass

    return images_major, captions_major, images_minor, captions_minor, unique_minors_cpy

def get_meta_datasets(captions_file_path, images_path, tokenizer, data_split, debug):
    # Encode entire dataset
    img_name_list, captions_list, unique_images = _get_image_caption_list(captions_file_path, images_path, debug)
    #print('\nUnmodified image name list:\n', img_name_list, '\n')

    # Collect example captions and corresponding images for attention plotting
    plot_attention_img_list = [img_name_list[i] for i in plot_attention_idx_list]
    plot_attention_caption_list = [captions_list[i] for i in plot_attention_idx_list]

    # Remove all images & captions reserved for attention plotting from dataset
    demo_indices = [i for i, img_name in enumerate(img_name_list) if img_name in plot_attention_img_list]
    demo_indices = sorted(demo_indices, reverse=True)  # Delete later indices first
    for i in demo_indices:
        #print('Reserving for final plotting:', img_name_list[i])
        del img_name_list[i]
        del captions_list[i]

    for demo_name in plot_attention_img_list:
        try:
            unique_images.remove(demo_name)
        except Exception as e:
            print('Error during meta dataset creation:', e)

    # Split into (train) and (test and eval) data, respectively
    images_train, captions_train, images_t_e, captions_t_e, unique_minors = split_dataset(img_name_list, captions_list,
                                                                           data_split['train'], unique_images, True)

    # Split into validation, and testing
    test_percentage = round(data_split['test'] / (1. - data_split['train']), 2)

    # Split into (train) and (test and eval) data, respectively
    images_test, captions_test, images_valid, captions_valid, _ = split_dataset(images_t_e, captions_t_e,
                                                                           test_percentage, unique_minors, False)

    # Process train data
    # Tokenize words;  cap_list = list of paddeded integer sequences representing captions
    cap_list, _, max_capt_len, tokenizer = _tokenize_words(captions_train, tokenizer, init=True)
    # Shuffle captions and image names together
    caps_train, img_names_train = shuffle(cap_list,
                                          images_train,
                                          # random_state=1
                                          )
    # Process validation data
    # Tokenize words;  cap_list = list of paddeded integer sequences representing captions
    cap_list, _, _, _ = _tokenize_words(captions_valid, tokenizer)
    # Shuffle captions and image names together
    caps_val, img_names_val = shuffle(cap_list,
                                      images_valid,
                                      # random_state=1
                                      )

    # Process test data
    # Tokenize words;  cap_list = list of paddeded integer sequences representing captions
    cap_list, _, _, _ = _tokenize_words(captions_test, tokenizer)
    # Shuffle captions and image names together
    caps_test, img_names_test = shuffle(cap_list,
                                      images_test,
                                      # random_state=1
                                      )

    # Construct TF datasets containing image paths (i.e. not actual images, hence 'meta') and corresponding captions
    # Training data set
    train_ds_meta = tf.data.Dataset.from_tensor_slices((img_names_train, caps_train))

    # Validation data set
    valid_ds_meta = tf.data.Dataset.from_tensor_slices((img_names_val, caps_val))

    # Test data set
    test_ds_meta = tf.data.Dataset.from_tensor_slices((img_names_test, caps_test))

    return train_ds_meta, valid_ds_meta, test_ds_meta, max_capt_len, plot_attention_img_list, \
           plot_attention_caption_list
