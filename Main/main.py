import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
import csv
from utils.utils import *
from preprocessing.preprocessing import *
import tensorflow as tf
from variables import *
from training import *
from evaluation import *
from Attention.modules import *
from Decoder.decoder import *
from Encoder.encoder import *


def main():
   tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words,
                                                     oov_token="<unk>",
                                                     filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
   img_list, caption_list = get_image_n_caption(csv_file_path,image_path,debug)
   plot_attention_img_list = [img_list[i] for i in plot_attention_list]
   plot_attention_caption_list = [caption_list[i] for i in plot_attention_list]

   index = [i for i, e in enumerate(img_list) if e in plot_attention_img_list]

   index.sort(reverse=True)



   for i in index:
        del img_list[i]
        del caption_list[i]


   # print(plot_attention_caption_list)
   # print(plot_attention_img_list)
   caption_vector, train_seqs, max_length, tokenizer = tokenize_words(caption_list,tokenizer)
   # Encoding-Attention-Decoding Architecture
   encoder = InceptionEncoder(embedding_dim)
   decoder = RNNDecoder(embedding_dim, units, vocab_size)
   split_ratio = dict(train=0.70,
                      valid=0.15,
                      test=0.15)
   train_data, val_data, test_data = split(img_list,caption_vector,split_ratio)
   # print("Example of test data",test_data[0][0],test_data[1][0,:])
   loss_plot_train, loss_plot_val=training(encoder, decoder, train_data, val_data, tokenizer)
   bleu_score = evaluate(test_data,encoder,decoder, max_length, tokenizer)
   print("bleu score is ",bleu_score)
   count = 0
   '''
   for img in plot_attention_img_list:

       result, attention_plot = get_plot_attention(img,encoder,decoder, max_length, tokenizer)
       plot_attention(img, result, attention_plot, count)
       count = count + 1
   '''
if __name__ == '__main__':
    main()
