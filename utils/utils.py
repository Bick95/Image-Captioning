import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
from sklearn.preprocessing import *
import tensorflow as tf
from preprocessing.preprocessing import *

from PIL import Image

'''
Functions stores the location of the image and the corresponding caption.
Location and the captions are returned as independent list.
This will also save the extracted features from the images.
'''


def get_image_n_caption(csv_file_path, images_path, debug):
    caption_list = []
    img_name_list = []
    cnt = 0
    with open(csv_file_path, 'r') as csvfile:
        data = csv.reader(csvfile, delimiter='|')
        for row in data:
            img_name = ""
            img_name = images_path + row[0]
            caption = '<start> ' + row[2] + ' <end>'
            img_name_list.append(img_name)
            caption_list.append(caption)
            if (not debug):
                continue
            cnt = cnt + 1
            if (cnt == 150):
                break
    img_name_list = img_name_list[1:]  # 1st row contains column names.
    caption_list = caption_list[1:]  # 1st row contains column names.
    encode_train = sorted(set(img_name_list))
    store_img_extracted_features(encode_train)  # Store the extracted features from the images.
    return img_name_list, caption_list


def split(img_name_vector, cap_vector, split_ratio):
    img_name_vector, cap_vector = shuffle(img_name_vector,
                                          cap_vector,
                                          random_state=1)
    img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                        cap_vector,
                                                                        test_size=0.2,
                                                                        random_state=0)

    # Split for training, validation, and testing
    test_val_percentage = (1. - split_ratio['train'])
    test_percentage = (1. - (split_ratio['test'] / test_val_percentage))
    print('Percentage training data:', split_ratio['train'])
    print('Percentage validat. data:', (test_val_percentage - split_ratio['test']))
    print('Percentage testing data: ', split_ratio['test'])
    img_names_train, img_names_val_test, caps_train, caps_val_test = train_test_split(img_name_vector,
                                                                                      cap_vector,
                                                                                      test_size=test_val_percentage,
                                                                                      random_state=0)
    # 2. Create validation and tetsing sets using an 50-50 split
    img_names_val, img_names_test, caps_val, caps_test = train_test_split(img_names_val_test,
                                                                          caps_val_test,
                                                                          test_size=test_percentage,
                                                                          random_state=0)
    # Construct TF datasets containing image paths (i.e. not actual images, hence 'meta') and corresponding captions
    # Training data set
    train_ds_meta = tf.data.Dataset.from_tensor_slices((img_names_train, caps_train))

    # Validation data set
    valid_ds_meta = tf.data.Dataset.from_tensor_slices((img_names_val, caps_val))

    test_ds_meta = [img_names_test, caps_test]
    return train_ds_meta, valid_ds_meta, test_ds_meta


def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, cap
