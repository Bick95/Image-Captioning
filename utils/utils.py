import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
'''
Functions stores the location of the image and the corresponding caption.
Location and the captions are returned as independent list
'''
def get_image_n_caption(csv_file_path,images_path,debug):
    caption_list = []
    img_name_list = []
    cnt = 0
    with open(csv_file_path, 'r') as csvfile:
        data = csv.reader(csvfile, delimiter='|')
        for row in data:
            img_name = ""
            img_name = images_path + row[0]
            caption  = '<start> ' + row[2] + ' <end>'
            img_name_list.append(img_name)
            caption_list.append(caption)
            if(not debug):
                continue
            cnt = cnt + 1
            if(cnt == 100):
                break
    return img_name_list[1:], caption_list[1:]

def max_len_tensor(tensor):
    return max(len(t) for t in tensor)

def split(img_name_vector, cap_vector):
    img_name_vector, cap_vector = shuffle(img_name_vector,
                                          cap_vector,
                                              random_state=1)
    img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                        cap_vector,
                                                                        test_size=0.2,
                                                                        random_state=0)
    return img_name_train, img_name_val, cap_train, cap_val