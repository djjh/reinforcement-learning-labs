from matplotlib import pyplot as pp
import numpy as np
import tensorflow as tf
import random


######################################
# Initialize Algorithm Configuration #
######################################

show_dataset_description = False
show_plots = False
random_seed = 0


####################
# Set Random Seeds #
####################

random.seed(random_seed)
np.random.seed(random_seed)
tf.set_random_seed(random_seed)


##############################
# Load Fashion MNIST Dataset #
##############################

# Load the dataset and labels.
(train_images, train_labels), (test_images, test_labels) = \
    tf.keras.datasets.fashion_mnist.load_data()
label_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
num_labels = 10
image_shape = train_images.shape[1:]


###############################
# Display Dataset Information #
###############################

if show_dataset_description:
    # Display dataset dimensions and datatype.
    print("train_images - shape: %s, dtype: %s" %
    (train_images.shape, train_images.dtype))
    print("train_labels - shape: %s, dtype: %s" %
        (train_labels.shape, train_labels.dtype))
    print("test_images - shape: %s, dtype: %s" %
        (test_images.shape, test_images.dtype))
    print("test_labels - shape: %s, dtype: %s" %
        (test_labels.shape, test_labels.dtype))

if show_plots:
    # Plot an example of one of the test images.
    pp.figure()
    pp.imshow(train_images[0])
    pp.colorbar()
    pp.grid(False)
    pp.show()

    # Display the first 25 images from the dataset with labels.
    pp.figure(figsize=(10, 10))
    for i in range(25):
        pp.subplot(5, 5, i+1)
        pp.xticks([])
        pp.yticks([])
        pp.grid(False)
        pp.imshow(train_images[i], cmap=pp.cm.binary)
        pp.xlabel(label_names[train_labels[i]])
    pp.show()


######################
# Preprocess Dataset #
######################

# Scale images range from [0, 255] to [0, 1].
train_images = train_images / 255
test_images = test_images / 255


####################
# Initialize Model #
####################

hidden_layer_size = 128
model = tf.keras.Sequential(layers=[
    tf.keras.layers.Flatten(input_shape=image_shape),
    tf.keras.layers.Dense(units=hidden_layer_size, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=num_labels, activation=tf.nn.softmax)])

model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss=tf.losses.sparse_softmax_cross_entropy,
    metrics=['accuracy'])


###############
# Train Model #
###############

model.fit(x=train_images, y=train_labels, epochs=5)


##################
# Evaluate Model #
##################

test_loss, test_accuracy = model.evaluate(x=test_images, y=test_labels)


#########################
# Display Diagnotistics #
#########################

print("Test Accuracy: %.3f" % test_accuracy)


####################
# Make Predictions #
####################

predictions = model.predict(test_images)


###################################
# Display Predictions Graphically #
###################################

if show_plots:

    def plot_image(prediction_probabilities, true_label, image):
        pp.xticks([])
        pp.yticks([])
        pp.grid(False)
        pp.imshow(image, cmap=pp.cm.binary)
        predicted_label = np.argmax(prediction_probabilities)
        color = 'blue' if predicted_label == true_label else 'red'
        pp.xlabel(
            "{} {:2.0f}% ({})".format(
                label_names[predicted_label],
                100 * np.max(prediction_probabilities),
                label_names[true_label]),
            color=color)

    def plot_value_array(prediction_probabilities, true_label):
        pp.xticks([])
        pp.yticks([])
        pp.grid(False)
        plot = pp.bar(range(num_labels), prediction_probabilities, color="#777777")
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
        plot_image(predictions[i], test_labels[i], test_images[i])
        pp.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(predictions[i], test_labels[i])
    pp.show()
