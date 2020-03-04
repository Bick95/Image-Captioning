import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
import csv
from utils.utils import *
from preprocessing.preprocessing import *
import tensorflow as tf
'''
GLOBAL VARIABLES
'''
csv_file_path = "../data/flickr30k_images/results.csv"
image_path = "../data/flickr30k_images/flickr30k_images/"
#Debug should be set to 1 whenever you want to test the flow of the code on your system.
#Instead of all images, it will just start training for 100 images
debug = 1
#Number of words to be considered while encoding
max_words = 5000

def main():
   img_list, caption_list = get_image_n_caption(csv_file_path,image_path,debug)
   encode_train = sorted(set(img_list))
   store_img_extracted_features(encode_train)
   #store_img_extracted_features(encode_train)
   caption_vector = tokenize_words(max_words,caption_list)
   img_name_train, img_name_val, cap_train, cap_val = split(img_list,caption_vector)

if __name__ == '__main__':
    main()