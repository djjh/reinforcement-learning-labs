import matplotlib.pyplot as pp
import numpy as np
import tensorflow as tf
import random


#######################################
# Initialize Algorithm Congfiguration #
#######################################

num_words = 10000
info_logging = True
random_seed = 0


###########################
# Initialize Random Seeds #
###########################

random.seed(random_seed)
np.random.seed(random_seed)
tf.set_random_seed(random_seed)


##########################
# Initialize the Dataset #
##########################

imdb = tf.keras.datasets.imdb

#
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)
num_labels = len(set(test_labels))

# Get the word index corresponding to the data indices.
word_index = imdb.get_word_index()

# Preprocess the word index, removing reserved words.
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3
padding_word_index = word_index["<PAD>"]

# Build a function for decoding the training/test data.
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


###############################
# Display Dataset Description #
###############################

if info_logging:
    print("Train Data   - shape %s, dtype %s" % (train_data.shape, train_data.dtype))
    print("Train Labels - shape %s, dtype %s" % (train_labels.shape, train_labels.dtype))
    print("Test Data    - shape %s, dtype %s" % (test_data.shape, test_data.dtype))
    print("Test Labels  - shape %s, dtype %s" % (test_labels.shape, test_labels.dtype))
    print("Sample Data: %s" % train_data[0][:10])
    print("Sample Data (Decoded): %s" % decode_review(train_data[0][:10]))


##########################
# Preprocess the Dataset #
##########################

# Padd the arrays so that they all have the same length.
review_padding_length = 256
train_data = tf.keras.preprocessing.sequence.pad_sequences(
    sequences=train_data,
    value=padding_word_index,
    padding="post",
    maxlen=review_padding_length)
test_data = tf.keras.preprocessing.sequence.pad_sequences(
    sequences=test_data,
    value=padding_word_index,
    padding="post",
    maxlen=review_padding_length)

# Create a validation set.
validation_set_size = 10000
x_val = train_data[:validation_set_size]
partial_x_train = train_data[validation_set_size:]

y_val = train_labels[:validation_set_size]
partial_y_train = train_labels[validation_set_size:]


############################################
# Display Preprocessed Dataset Description #
############################################

if info_logging:
    print("Prepocess Train Data   - shape %s, dtype %s" % (train_data.shape, train_data.dtype))


####################
# Initialize Model #
####################

model = tf.keras.Sequential()

# Add an Embedding layer as the first layer. This will convert the high dimensional
# input word-index (an index into a hyptothetical one-hot encoding length num_words)
# into a lower dimensional vector. This mapping will be learned as the model trains.
# Output dimensions: (batch, sequence, embedding-dimensions)
model.add(tf.keras.layers.Embedding(input_dim=num_words, output_dim=16, input_length=256))

# Add a global average pooling layer as the second layer. This will convert the
# variable length reviews into fixed length vectors by averaging the embeddings
# over the sequence dimension.
# Output dimensions: (batch, empedding-dimensions)
model.add(tf.keras.layers.GlobalAveragePooling1D())

# Add a fully-connected layer for learning the classification.
# Output dimensions: (batch, num-hidden-units)
model.add(tf.keras.layers.Dense(units=16, activation=tf.nn.relu))

# Add the final output layer, which ouputs a single number representing the probability
# of the binary classifcation.
# Output dimensions: (batch, 1)
model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

if info_logging:
    model.summary()

model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss=tf.keras.backend.binary_crossentropy,
    metrics=['acc'])


###############
# Train Model #
###############

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=40,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1)


##################
# Evaluate Model #
##################

results = model.evaluate(test_data, test_labels)

print(results)


################################
# Display Training Performance #
################################

history_dict = history.history
print(history_dict.keys())


acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc)+1)

pp.plot(epochs, loss, 'bo', label='Training loss')
pp.plot(epochs, val_loss, 'b', label='Validation loss')
pp.title('Training and Validation loss')
pp.xlabel('Epochs')
pp.ylabel('Loss')
pp.legend()
pp.show()

pp.clf()

pp.plot(epochs, acc, 'bo', label='Training acc')
pp.plot(epochs, val_acc, 'b', label='Validation acc')
pp.title('Training and Validation accuracy')
pp.xlabel('Epochs')
pp.ylabel('Accuracy')
pp.legend()
pp.show()
