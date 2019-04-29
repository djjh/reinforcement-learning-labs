import matplotlib.pyplot as pp
import numpy as np
import tensorflow as tf
import random


######################################
# Initialize Algorithm Configuration #
######################################

show_dataset_description = False
show_plots = False
num_epochs = 5
random_seed = 0


###########################
# Initialize Random Seeds #
###########################

random.seed(random_seed)
np.random.seed(random_seed)
tf.set_random_seed(random_seed)


################
# Load Dataset #
################

(train_images, train_labels), (test_images, test_labels) = \
    tf.keras.datasets.fashion_mnist.load_data()
label_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
num_labels = len(set(train_labels))
image_shape = train_images[0].shape

assert num_labels == len(label_names)


###############################
# Display Dataset Description #
###############################

if show_dataset_description:
    # Display description of the dataset dimensions and datatypes.
    print("Train images - shape: %s, datatype: %s" %
        (train_images.shape, train_images.dtype))
    print("Train labels - shape: %s, datatype: %s" %
        (train_labels.shape, train_labels.dtype))
    print("Test images - shape: %s, datatype: %s" %
        (test_images.shape, test_images.dtype))
    print("Test labels - shape: %s, datatype: %s" %
        (test_labels.shape, test_labels.dtype))

if show_plots:
    # Display an example image with dimensions and value range.
    pp.figure()
    pp.imshow(train_images[0])
    pp.colorbar()
    pp.grid(False)
    pp.show()


######################
# Preprocess Dataset #
######################

# Rescale the image pixels from range [0, 255] to [0, 1]
train_images = train_images / 255
test_images = test_images / 255


####################
# Initialize Model #
####################

# Initialize placeholders for input and output.
images_placeholder = tf.placeholder(shape=(None, *image_shape), dtype=tf.float32)
labels_placeholder = tf.placeholder(shape=(None,), dtype=tf.int32)

# Build model.
input_layer = images_placeholder
previous_layer = tf.layers.Flatten(input_shape=image_shape)(input_layer)
previous_layer = tf.layers.Dense(units=128,activation=tf.nn.relu)(previous_layer)
output_layer = tf.layers.Dense(units=num_labels, activation=tf.nn.softmax)(previous_layer)
probabilities = output_layer

# Build prediction operation.
predict_function = tf.argmax(probabilities, axis=1)  #tf.random.categorical(logits=probabilities, num_samples=1)

# Build loss function.
loss_function = tf.losses.sparse_softmax_cross_entropy(labels=labels_placeholder, logits=probabilities)

# Build training operation.
train_function = tf.train.AdamOptimizer().minimize(loss_function)

# Build the accuracy metric.
accuracy, accuracy_function = tf.metrics.accuracy(
    labels=tf.squeeze(labels_placeholder),
    predictions=predict_function)


######################
# Initialize Compute #
######################

session = tf.InteractiveSession()
session.run(tf.local_variables_initializer())
session.run(tf.global_variables_initializer())


###############
# Train Model #
###############

for epoch in range(num_epochs):
    _, batch_loss, batch_accuracy = session.run(
        fetches=[train_function, loss_function, accuracy_function],
        feed_dict={
            images_placeholder: train_images,
            labels_placeholder: train_labels
        })
    print("train - epoch: %3d loss: %.3f  accuracy: %.3f" % (epoch, batch_loss, batch_accuracy))


##################
# Evaluate Model #
##################

batch_loss, batch_accuracy = session.run(
    fetches=[loss_function, accuracy_function],
    feed_dict={
        images_placeholder: test_images,
        labels_placeholder: test_labels
    })
print("test - loss: %.3f  accuracy: %.3f" % (batch_loss, batch_accuracy))


####################
# Make predictions #
####################

prediction_probabilities = session.run(
    fetches=probabilities,
    feed_dict={images_placeholder: test_images})

if show_plots:

    def plot_image(prediction_probabilities, true_label, test_image):
        pp.xticks([])
        pp.yticks([])
        pp.grid(False)
        pp.imshow(test_image, cmap=pp.cm.binary)
        predicted_label = np.argmax(prediction_probabilities)
        color = 'blue' if predicted_label == true_label else 'red'
        pp.xlabel(
            "{} {:2.0f}% ({})".format(
                label_names[predicted_label],
                100 * np.max(prediction_probabilities),
                label_names[true_label]
            ),
            color=color)

    def plot_value_array(prediction_probabilities, true_label):
        pp.xticks([])
        pp.yticks([])
        pp.grid(False)
        plot = pp.bar(range(num_labels), prediction_probabilities, color='#777777')
        pp.ylim([0, 1])
        predicted_label = np.argmax(prediction_probabilities)
        plot[predicted_label].set_color('red')
        plot[true_label].set_color('blue')


    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    pp.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        pp.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(prediction_probabilities[i], test_labels[i], test_images[i])
        pp.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(prediction_probabilities[i], test_labels[i])
    pp.show()
