import calendar
import io
import itertools
import os
import numpy as np
import matplotlib.pyplot as pp
import tensorflow as tf
from time import gmtime, strftime
import random
import shutil
import sklearn
import sklearn.metrics


###################################
# Initialize Algorithm Parameters #
###################################

experiment_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
log_directory = "/tmp/log/{}/summaries/{}".format(experiment_name, timestamp)
num_epochs = 100
random_seed = 0

if os.path.exists(log_directory):
    shutil.rmtree(log_directory)
print("log directory: '%s'" % log_directory)


################################
# Initialize Random Generators #
################################

random.seed(random_seed)
np.random.seed(random_seed)
tf.set_random_seed(random_seed)


######################
# Initialize Dataset #
######################

(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()

input_dimension = train_data.shape[1:]
num_labels = len(set(train_labels))
label_names = sorted(set(train_labels))


######################
# Preprocess Dataset #
######################

print("Training Data   - shape: %s, dtype: %s" % (train_data.shape, train_data.dtype))
print("Training Labels - shape: %s, dtype: %s" % (train_labels.shape, train_labels.dtype))
print("Testing Data    - shape: %s, dtype: %s" % (test_data.shape, test_data.dtype))
print("Testing Labels  - shape: %s, dtype: %s" % (test_labels.shape, test_labels.dtype))
print()

tmp_data = train_data
tmp_labels = train_labels

num_train_samples = len(train_data)-50
train_data = tmp_data[:num_train_samples]
train_labels = tmp_labels[:num_train_samples]

validation_data = tmp_data[num_train_samples:]
validation_labels = tmp_labels[num_train_samples:]

print(train_data.shape, len(train_data))
print(train_labels.shape, len(train_labels))

print(validation_data.shape, len(validation_data))
print(validation_labels.shape, len(validation_labels))


####################
# Build Graph #
####################

def plot_to_png(figure):
    buf = io.BytesIO()
    pp.savefig(buf, format='png')
    pp.close(figure)
    buf.seek(0)
    return buf.getvalue()

def image_grid(images, labels):
    figure = pp.figure(figsize=(10,10))
    for i in range(len(images)):
        pp.subplot(5, 5, i+1, title="{}".format(labels[i]))
        pp.xticks([])
        pp.yticks([])
        pp.grid(False)
        pp.imshow(images[i], cmap=pp.cm.binary)
    return figure

def plot_confusion_matrix(confusion_matrix, labels):
    cm = confusion_matrix

    figure = pp.figure(figsize=(5, 5))
    pp.imshow(cm, interpolation='nearest', cmap=pp.cm.Blues)
    pp.title("Confusion Matrix")
    # pp.colorbar()
    tick_marks = np.arange(len(labels))
    pp.xticks(tick_marks, labels, rotation=45)
    pp.yticks(tick_marks, labels)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text for dark background and black for light background.
    threshold = np.nanmax(a=cm)/3.
    for i, j, in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = 'white' if cm[i, j] > threshold else 'black'
        pp.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    pp.tight_layout()
    pp.ylabel('True Label')
    pp.xlabel('Predicted Label')
    return figure


def generate_confusion_matrix_png(target_labels, predicted_labels, label_names):
    # Calculate the confusion matrix.
    confusion_matrix = sklearn.metrics.confusion_matrix(
        y_true=target_labels,
        y_pred=predicted_labels,
        labels=label_names)

    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(confusion_matrix, label_names)
    return plot_to_png(figure)

graph = tf.Graph()
with graph.as_default():

    # Random seed must be set for each new graph.
    tf.set_random_seed(random_seed)

    with tf.variable_scope('Diagnostics'):
        n_images = 1
        image = np.reshape(train_data[:n_images], (-1, 28, 28, 1))
        tf.summary.image("Training Data 1", image, max_outputs=n_images)

        figure = image_grid(train_data[:25], train_labels[:25])
        png = plot_to_png(figure)
        image = tf.image.decode_png(png, channels=4)
        image = tf.expand_dims(image, 0)
        tf.summary.image("Training Data 2", image)

        confusion_matrix_png_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        confusion_matrix_image = tf.image.decode_png(contents=confusion_matrix_png_placeholder, channels=4)
        confusion_matrix_image = tf.expand_dims(confusion_matrix_image, 0)
        tf.summary.image("ConfusionMatrix", confusion_matrix_image)

    with tf.variable_scope('Features'):
        images_placeholder = tf.placeholder(shape=(None, *input_dimension), dtype=tf.int32)
        images_placeholder = tf.cast(x=images_placeholder, dtype=tf.float32)

    with tf.variable_scope('Labels'):
        labels_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)
        # labels_placeholder = tf.cast(x=labels_placeholder, dtype=tf.float32)

    with tf.variable_scope('Model'):
        input_layer = images_placeholder
        previous_layer = input_layer

        previous_layer = tf.layers.Flatten(name="Flatten")(previous_layer)
        # Alternative to Flatten:
        # with tf.variable_scope('Flatten'):
        #     shape = previous_layer.get_shape()
        #     dim = tf.reduce_prod(shape[1:])
        #     previous_layer = tf.reshape(tensor=previous_layer, shape=[-1, dim])

        layer = tf.layers.Dense(name="HiddenLayer", units=128, activation=tf.nn.relu)
        previous_layer = layer(previous_layer)
        weights = layer.weights[0]
        biases = layer.weights[1]
        # Could do this in a helper method for all such variables.
        mean = tf.reduce_mean(weights)
        tf.summary.scalar('Hidden Layer Weights Mean', mean)
        tf.summary.scalar('Hidden Layer Weights Standard Deviation', tf.sqrt(tf.reduce_mean(tf.square(weights - mean))))
        tf.summary.scalar('Hidden Layer Weights Max', tf.reduce_max(weights))
        tf.summary.scalar('Hidden Layer Weights Min', tf.reduce_min(weights))
        tf.summary.histogram(name="Hidden Layer Weights", values=weights)
        tf.summary.histogram(name="Hidden Layer Biases", values=biases)
        tf.summary.histogram(name="Hidden Layer Activations", values=previous_layer)

        layer = tf.layers.Dense(name="OutputLayer", units=num_labels, activation=tf.nn.sigmoid)
        previous_layer = layer(previous_layer)
        weights = layer.weights[0]
        biases = layer.weights[1]
        tf.summary.histogram(name="Output Layer Weights", values=weights)
        tf.summary.histogram(name="Output Layer Biases", values=biases)
        tf.summary.histogram(name="Output Layer Activations", values=previous_layer)

        output_layer = previous_layer
        probabilities = output_layer

    with tf.variable_scope('Predict'):
        predictions = tf.argmax(probabilities, axis=1)

    with tf.variable_scope('Accuracy'):
        accuracy, accuracy_function = tf.metrics.accuracy(
            labels=tf.squeeze(labels_placeholder),
            predictions=predictions)
        tf.summary.scalar(name='Accuracy', tensor=accuracy)

    with tf.variable_scope('Loss'):
        # label_masks = tf.one_hot(indices=labels_placeholder, depth=num_labels)
        # target = tf.cast(x=labels_placeholder, dtype=tf.float32)
        # loss = tf.reduce_mean((label_masks*probabilities - target)**2)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels_placeholder, logits=probabilities)
        tf.summary.scalar(name="Loss", tensor=loss)

    with tf.variable_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer()
        gradients = optimizer.compute_gradients(loss)
        for gradient in gradients:
            tensor = gradient[0]
            variable = gradient[1]
            tf.summary.histogram(name="Gradient {}".format(variable.name), values=tensor)
        train = optimizer.apply_gradients(gradients)

    summaries = tf.summary.merge_all()

######################
# Initialize Compute #
######################

with graph.as_default(), \
    tf.Session(graph=graph) as session, \
    tf.summary.FileWriter(logdir=log_directory, graph=session.graph) as writer:

    session.run(fetches=tf.global_variables_initializer())
    session.run(fetches=tf.local_variables_initializer())


    ###############
    # Train Model #
    ###############

    for epoch in range(num_epochs):
        session.run(
            fetches=train,
            feed_dict={
                images_placeholder: train_data,
                labels_placeholder: train_labels})


        if epoch % 10 == 0:
            validation_predictions = session.run(
                fetches=predictions,
                feed_dict={
                    images_placeholder: validation_data,
                    labels_placeholder: validation_labels})

            confusion_matrix_png = generate_confusion_matrix_png(
                validation_labels,
                validation_predictions,
                label_names)

            summary, _ = session.run(
                fetches=[summaries, accuracy_function],
                feed_dict={
                    images_placeholder: validation_data,
                    labels_placeholder: validation_labels,
                    confusion_matrix_png_placeholder: confusion_matrix_png})
            writer.add_summary(summary, epoch)
        print("epoch: %d" % (epoch,))


    ##################
    # Evaluate Model #
    ##################

    session.run(
        fetches=loss,
        feed_dict={
            images_placeholder: test_data,
            labels_placeholder: test_labels})
