import matplotlib.pyplot as pp
import numpy as np
import tensorflow as tf
import random

###################################
# Initialize Algorithm Parameters #
###################################

num_words = 10000
display_dataset_properties = True
display_training_performance = True
num_epochs = 40
learning_rate = 2e-2
random_seed = 0


###########################
# Initialize Random Seeds #
###########################

random.seed(random_seed)
np.random.seed(random_seed)
tf.set_random_seed(random_seed)


######################
# Initialize Dataset #
######################

imdb = tf.keras.datasets.imdb

# Load the imdb positive/negative reviews dataset.
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)
num_labels = len(set(train_labels))

# Load the index of words corresponding to the dataset.
word_index = imdb.get_word_index()

# Preprocess the word index, removing reserved words.
word_index = {k:(v+3) for k,v in word_index.items()}
padding_word_index = word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

# Build a reverse word index to decode the training/test data.
reverse_word_index = {v:k for k,v in word_index.items()}
def decode(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

##############################
# Display Dataset Properties #
##############################

if display_dataset_properties:
    print()
    print("Training Data   - shape: %s datatype: %s" % (train_data.shape, train_data.dtype))
    print("  Inner List - min length: %s, max length: %s" %
        (min([len(l) for l in train_data]), max([len(l) for l in train_data])))
    print("Training Labels - shape: %s datatype: %s" % (train_labels.shape, train_labels.dtype))
    print("Testing Data    - shape: %s datatype: %s" % (test_data.shape, test_data.dtype))
    print("  Inner List - min length: %s, max length: %s" %
        (min([len(l) for l in test_data]), max([len(l) for l in test_data])))
    print("Testing Labels  - shape: %s datatype: %s" % (test_labels.shape, test_labels.dtype))
    print()
    print("Example Review (Encoded): %s..." % train_data[0][:10])
    print("Example Review (Decoded): %s..." % decode(train_data[0][:10]))


######################
# Preprocess Dataset #
######################

# Pad the data arrays (the reviews), which are variable length so that they
# have the same fixed length and can be fed into the model.
def pad_sequences(sequences, length):
    return [s[:length] + [0]*(length-len(s)) for s in sequences]
review_length = 256
train_data = np.array(pad_sequences(train_data, review_length))
test_data = np.array(pad_sequences(test_data, review_length))

# Partition out a validation set from the training set.
validation_set_size = 1000
validation_data = train_data[:validation_set_size]
validation_labels = train_labels[:validation_set_size]
train_data = train_data[validation_set_size:]
train_labels = train_labels[validation_set_size:]


###########################################
# Display Preprocessed Dataset Properties #
###########################################

if display_dataset_properties:
    print()
    print("Preprocessed Training Data   - shape: %s datatype: %s" % (train_data.shape, train_data.dtype))
    print("  Inner List - min length: %s, max length: %s" %
        (min([len(l) for l in train_data]), max([len(l) for l in train_data])))
    print("Preprocessed Training Labels - shape: %s datatype: %s" % (train_labels.shape, train_labels.dtype))
    print("Validation Data    - shape: %s datatype: %s" % (validation_data.shape, validation_data.dtype))
    print("  Inner List - min length: %s, max length: %s" %
        (min([len(l) for l in validation_data]), max([len(l) for l in validation_data])))
    print("Validation Labels  - shape: %s datatype: %s" % (validation_labels.shape, validation_labels.dtype))
    print()

####################
# Initialize Model #
####################

# Initialize placeholders for model input/output.

# Reviews are represented as fixed size arrays of word indices.
reviews_placeholder = tf.placeholder(shape=(None, review_length), dtype=tf.int32)
labels_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)

# Initialize the input layer to the model, which is just the word indices.
input_layer = reviews_placeholder
previous_layer = input_layer

# Build embeddings layer, which will map each word in the input sequence to a
# corresponding embedding:
#    F:(batch_size, review_length) -> (batch_size, review_length, embedding_dimensions)
previous_layer = tf.keras.layers.Embedding(input_dim=num_words, output_dim=16, input_length=256)(previous_layer)

# Build a pooling layer which will average all of the embeddings for a review
# into a single embedding:
#    F:(batch_size, review_length, embedding_dimensions) -> (batch_size, embedding_dimensions)
previous_layer = tf.keras.layers.GlobalAveragePooling1D()(previous_layer)

# Build a fully connected hidden layer for learning the classifcation.
#    F:(batch_size, embedding_dimensions) -> (batch_size, hidden_layer_units)
hidden_layer_units = 16
previous_layer = tf.layers.Dense(units=hidden_layer_units, activation=tf.nn.relu)(previous_layer)

# Build the output layer.
#    F:(batch_size, hidden_layer_units) -> (batch_size, 1)
output_layer = tf.layers.Dense(units=1, activation=tf.nn.sigmoid)(previous_layer)

# Build the prediction operation, which outputs a probability of a positive review.
probability = output_layer
prediction = tf.cast(x=tf.round(probability), dtype=tf.int32)

# Build the loss function.
loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(
    target=tf.expand_dims(input=tf.cast(labels_placeholder, dtype=tf.float32), axis=1),
    output=probability,
    from_logits=False))

# Build the training operation.
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Build the metrics.
accuracy, accuracy_function = tf.metrics.accuracy(
    labels=tf.squeeze(labels_placeholder),
    predictions=prediction)

######################
# Initialize Compute #
######################

session = tf.InteractiveSession()
session.run(tf.local_variables_initializer())
session.run(tf.global_variables_initializer())


###############
# Train Model #
###############

history = {
    'training_loss': [],
    'training_accuracy': [],
    'validation_loss': [],
    'validation_accuracy': []}

# TODO: mini-batching to emulate keras.Sequential.fit
for epoch in range(num_epochs):
    _, training_loss, training_accuracy = session.run(
        fetches=[train, loss, accuracy_function],
        feed_dict={
            reviews_placeholder: train_data,
            labels_placeholder: train_labels})

    validation_loss, validation_accuracy = session.run(
        fetches=[loss, accuracy_function],
        feed_dict={
            reviews_placeholder: validation_data,
            labels_placeholder: validation_labels})

    history['training_loss'].append(training_loss/train_data.shape[0])
    history['training_accuracy'].append(training_accuracy)
    history['validation_loss'].append(validation_loss/validation_data.shape[0])
    history['validation_accuracy'].append(validation_accuracy)

    print("Epoch: %d, Train Accuracy: %f, Validation Accuracy: %f" %
        (epoch, training_accuracy, validation_accuracy))


##################
# Evaluate Model #
##################

test_accuracy = session.run(
    fetches=accuracy_function,
    feed_dict={
        reviews_placeholder: test_data,
        labels_placeholder: test_labels})

print("Test Accuracy: %.3f" % test_accuracy)


################################
# Display Training Performance #
################################

if display_training_performance:

    epochs = range(1, num_epochs+1)

    pp.plot(epochs, history['training_loss'], 'bo', label='Training Loss')
    pp.plot(epochs, history['validation_loss'], 'b', label='Validation Loss')
    pp.title('Training and Validation Loss')
    pp.xlabel('Epochs')
    pp.ylabel('Loss')
    pp.legend()
    pp.show()

    pp.clf()

    pp.plot(epochs, history['training_accuracy'], 'bo', label='Training Accuracy')
    pp.plot(epochs, history['validation_accuracy'], 'b', label='Validation Accuracy')
    pp.title('Training and Validation Accuracy')
    pp.xlabel('Epochs')
    pp.ylabel('Accuracy')
    pp.legend()
    pp.show()


#####################
# Terminate Compute #
#####################

session.close()
