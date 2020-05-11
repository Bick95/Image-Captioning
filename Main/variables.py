


# Global Variables
# Shubham's:
#csv_file_path = "../data/flickr30k_images/results.csv"
#image_path = "../data/flickr30k_images/flickr30k_images/"

# Daniel's:
csv_file_path = "/media/daniel/Elements/DeepLearning/flickr30k_images/results.csv"     # TODO
image_path = "/media/daniel/Elements/DeepLearning/flickr30k_images/flickr30k_images/"  # TODO

# Peregrine:
#csv_file_path = "../../flickr30k_images/results.csv"
#image_path = "../../flickr30k_images/flickr30k_images/"

#Debug should be set to 1 whenever you want to test the flow of the code on your system.
#Instead of all images, it will just start training for 100 images
debug = 0

#Number of words to be considered while encoding
if debug:  # Vocab size rather low in debug mode
    max_words = 3000  # experimentally tested
else:
    max_words = 5000

# Training Variables
learning_rate = 0.0001                                                             # TODO
if debug:
    BATCH_SIZE = 16                                                                # TODO
else:
    BATCH_SIZE = 32  # Debug mode - Batch size - 16, else - 32
BUFFER_SIZE = 128
#num_captions = 715  # 70% of 715 == approx 500 training images
num_captions = 50000
embedding_dim = 256
units = 512
vocab_size = max_words + 1
captions_per_image = 4
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64
EPOCHS = 2                                                                          # TODO
ckpt_frequency = 50

Patience = EPOCHS  # Patience of early stopping

plot_attention_idx_list = [1, 10, 100]

# Loss function
SPARSE_CATEGORICAL_CROSS_ENTROPY =  0
NEGATIVE_LOG_LIKELIHOOD =           1
loss_function_choice = NEGATIVE_LOG_LIKELIHOOD

attention_mode = 0
SOFT_ATTENTION = 0
HARD_ATTENTION = 1