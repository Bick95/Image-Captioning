import tensorflow as tf
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

"""
def _load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def _store_img_extracted_features(encode_train):
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_features_extract_model = img_extract_model()
    image_dataset = image_dataset.map(
        load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)
    for img, path in image_dataset:
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features,
                                    (batch_features.shape[0], -1, batch_features.shape[3]))
        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())
"""

def _tokenize_words(max_words, captions, tokenizer):

    tokenizer.fit_on_texts(captions)  # Updates internal vocabulary based on a list of texts
    train_seqs = tokenizer.texts_to_sequences(captions)  # Transforms each text in texts to a sequence of integers

    # By default, all punctuation is removed, turning the texts into space-separated sequences of words
    # (words maybe include the ' character). These sequences are then split into lists of tokens.
    # They will then be indexed or vectorized.
    # 0 is a reserved index that won't be assigned to any word.
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    train_seqs = tokenizer.texts_to_sequences(captions)  # FIXME: redundant?
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')  # Pad each vector to the max_length of the captions

    # cap_vector: padded version of captions, captions in form of integer sequences, each int identifying unique word
    # from vocab of size max_words

    # train_seqs: unpadded captions in form of integer sequences, each int identifying unique word
    # from vocab of size max_words

    return cap_vector, train_seqs


"""
    Functions stores the location of the image and the corresponding caption.
    Location and the captions are returned as independent list
"""
def get_image_caption_list(captions_file_path, images_path, debug):
    caption_list = []
    img_name_list = []

    with open(captions_file_path, 'r') as csvfile:
        data = csv.reader(csvfile, delimiter='|')
        for row in data:
            #print(row)
            img_name = images_path + row[0]
            caption = '<start> ' + row[2] + ' <end>'
            img_name_list.append(img_name)
            caption_list.append(caption)
            if not debug:
                continue
    return img_name_list[1:], caption_list[1:]  # First row is headings...


def get_meta_datasets(captions_file_path, images_path, tokenizer, max_words, data_split, debug):

    # Encode entire dataset
    img_name_list, captions_list = get_image_caption_list(captions_file_path, images_path, debug)

    # Tokenize words
    cap_list, _ = _tokenize_words(max_words, captions_list, tokenizer)  # cap_vector = list of paddeded integer sequences representing captions

    # Shuffle captions and image names together
    # Set a random state
    cap_list, img_name_list = shuffle(cap_list,
                                      img_name_list,
                                      random_state=1)

    # Split for training, validation, and testing
    test_val_percentage = (1.-data_split['train'])
    test_percentage = (1.-(data_split['test']/test_val_percentage))

    print('Percentage training data:', data_split['train'])
    print('Percentage validat. data:', (test_val_percentage-data_split['test']))
    print('Percentage testing data: ', data_split['test'])

    # 1. Create training and test+validation sets using an 70-30 split
    img_names_train, img_names_val_test, caps_train, caps_val_test = train_test_split(img_name_list,
                                                                                      cap_list,
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

    # Test data set
    test_ds_meta = tf.data.Dataset.from_tensor_slices((img_names_test, caps_test))

    return train_ds_meta, valid_ds_meta, test_ds_meta
