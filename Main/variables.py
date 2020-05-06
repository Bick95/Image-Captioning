


###GLOBAL VARIABLES
csv_file_path = "../data/flickr30k_images/results.csv"
image_path = "../data/flickr30k_images/flickr30k_images/"
#Debug should be set to 1 whenever you want to test the flow of the code on your system.
#Instead of all images, it will just start training for 100 images
debug = 1
#Number of words to be considered while encoding
max_words = 5000
#TRAINING VARIABLES
BATCH_SIZE = 8  #Debug mode - Batch size - 8, else - 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = max_words + 1
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64
EPOCHS = 20

Patience = 10  #Patience of early stopping

plot_attention_list = [20,10,1]