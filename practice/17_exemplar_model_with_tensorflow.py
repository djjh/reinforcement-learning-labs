import coloredlogs
import io
import matplotlib.pyplot as pp
import math
import multiprocessing as mp
import numpy as np
import os
import tensorflow as tf
import scipy.stats
import random
import logging

from os.path import splitext, basename
from time import strftime, gmtime

######################################
# Logging Configuration
######################################

# Basic Configuration
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
coloredlogs.install()

# Library specific logging levels.
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('Python').setLevel(logging.CRITICAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u

# Initialize the logger for this file.
logger = logging.getLogger(__name__)


######################################
# Initialize Algorithm Configuration #
######################################

experiment_name = splitext(basename(__file__))[0]
timestamp = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
log_directory = "log/{}/summaries/{}".format(experiment_name, timestamp)
num_epochs = 10000
learning_rate = 1e-3
batch_size = 256
random_seed = 0

input_dimensions = 1
encoder_dimensions = 32
feature_dimensions = 32
noise_dimensions = feature_dimensions
noise_scale = 0.1
model_dimensions = 16

num_samples = 256*4
num_bins = num_samples // 10


###########################
# Initialize Random Seeds #
###########################

random.seed(random_seed)
np.random.seed(random_seed)
tf.set_random_seed(random_seed)


######################
# Initialize Dataset #
######################

x = np.linspace(-0.1, 1, num_samples)

y1 = (0.05 * np.random.randn(num_samples, input_dimensions) + 0.7).flatten()
y2 = (0.1 * np.random.randn(num_samples, input_dimensions) + 0.2).flatten()
y1_probability = np.random.binomial(1, 0.5, num_samples)
y = y1_probability*y1 + (1 - y1_probability)*y2

positives = np.expand_dims(x, axis=1)
negatives = np.expand_dims(y, axis=1)


###############################
# Initialize Plotting Helpers #
###############################

def to_png(figure):
    buffer = io.BytesIO()
    pp.savefig(buffer, format='png')
    pp.close(figure)
    buffer.seek(0)
    return buffer.getvalue()

class Plotter:

    def __init__(self):
        if __name__ == '__main__':

            # Only 'spawn' and 'forkserver' start methods work for using matplotlib
            # in other processes. (i.e. 'fork' does not work.)
            # Notes:
            # - set_start_method must be called under `if __name__ == '__main__':`
            # - the target function for the child process must be defined outside of the main gaurd.
            if mp.get_start_method(allow_none=True) == None:
                mp.set_start_method('spawn')
            assert mp.get_start_method(allow_none=False) in ['spawn', 'forkserver']

        self.parent_pipe, self.child_pipe = mp.Pipe()
        self.process = mp.Process(target=self.child_function, args=(), daemon=True)

    def __enter__(self):
        self.process.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.parent_pipe.send(None)
        self.process.join()

    def child_function(self):
        args = self.child_pipe.recv()
        while args:
            png = args[0](args[1])
            self.child_pipe.send(png)
            args = self.child_pipe.recv()

    def plot(self, function, args):
        self.parent_pipe.send((function, args))
        return self.parent_pipe.recv()


class PlotDistributionFunction:

    def __init__(self, plotter):
        self.plotter = plotter

    def __call__(self, positives, losses):
        return self.plotter.plot(PlotDistributionFunction.plot_distribution_to_png, (positives.numpy(), losses.numpy()))

    def plot_distribution_to_png(args):
        return to_png(PlotDistributionFunction.plot_distribution(*args))

    def plot_distribution(positives, losses):
        figure, axis = pp.subplots()

        left_axis = axis
        left_axis.hist(positives, bins=num_bins, density=True, color=(0.1, 0.2, 1.0, 0.3))
        left_axis.hist(negatives, bins=num_bins, density=True, color=(1.0, 0.2, 0.5, 0.3))

        right_axis = left_axis.twinx()
        # probabilities /= np.trapz(probabilities, x=positives, axis=0)
        # right_axis.plot(positives, probabilities, color='red')
        right_axis.plot(positives, losses, color='black')

        return figure


if __name__ == '__main__':


    logger.info("Log directory: {}".format(log_directory))

    with Plotter() as plotter:

        ####################
        # Initialize Model #
        ####################

        graph = tf.Graph()
        with graph.as_default():

            # Random seed must be set for each new graph.
            tf.set_random_seed(random_seed)

            # Initialize placeholders for the model's input and output parameters.

            with tf.variable_scope('Positives'):
                positives_placeholder = tf.placeholder(shape=(None, input_dimensions), dtype=tf.float32)

            with tf.variable_scope('Negatives'):
                negatives_placeholder = tf.placeholder(shape=(None, input_dimensions), dtype=tf.float32)

            with tf.variable_scope('PositiveLabels'):
                positives_labels = tf.fill(value=1, dims=tf.shape(positives_placeholder))

            with tf.variable_scope('NegativeLabels'):
                negatives_labels = tf.fill(value=0, dims=tf.shape(negatives_placeholder))


            with tf.variable_scope('Encoding'):

                def relu(output_dimensions):
                    init = 1.0/np.sqrt(encoder_dimensions)
                    kernel_initializer = tf.initializers.random_uniform(minval=-init, maxval=init)
                    return tf.layers.Dense(units=output_dimensions, kernel_initializer=kernel_initializer, activation=tf.nn.relu)

                def linear(output_dimensions):
                    init = 1.0/np.sqrt(encoder_dimensions)
                    kernel_initializer = tf.initializers.random_uniform(minval=-init, maxval=init)
                    return tf.layers.Dense(units=output_dimensions, kernel_initializer=kernel_initializer, activation=None)

                encoder_layers = [relu(encoder_dimensions), relu(encoder_dimensions), linear(feature_dimensions)]

                def encoder(previous_layer):
                    for layer in encoder_layers:
                        previous_layer = layer(previous_layer)
                    return previous_layer

                positives_encoder_mean = encoder(positives_placeholder)
                negatives_encoder_mean = encoder(negatives_placeholder)

                positives_features = positives_encoder_mean  # aka the latent z vector
                negatives_features = negatives_encoder_mean  # aka the latent z vector

                tf.summary.histogram('PositivesMean', positives_encoder_mean)

            # Build model that will learn a stochastic equality operator.

            with tf.variable_scope('Model'):
                model_layers = [
                    tf.layers.Dense(units=model_dimensions, activation=tf.nn.tanh),
                    tf.layers.Dense(units=1, activation=tf.nn.tanh)]

                def model(previous_layer):
                    for layer in model_layers:
                        previous_layer = layer(previous_layer)
                    return previous_layer

                positives_concat = tf.concat(values=[positives_features], axis=1)
                positives_logits = model(positives_concat)

                negatives_concat = tf.concat(values=[negatives_features], axis=1)
                negatives_logits = model(negatives_concat)

            with tf.variable_scope('Predict'):
                discriminator = tf.nn.sigmoid(positives_logits)
                positives_probabilites = (1 - discriminator) / discriminator
                positives_predictions = tf.cast(tf.round(positives_probabilites), dtype=tf.int32)

            with tf.variable_scope('Loss'):
                positives_loss = tf.losses.sigmoid_cross_entropy(
                    multi_class_labels=positives_labels,
                    logits=positives_logits,
                    reduction=tf.losses.Reduction.NONE)
                negatives_loss = tf.losses.sigmoid_cross_entropy(
                    multi_class_labels=negatives_labels,
                    logits=negatives_logits,
                    reduction=tf.losses.Reduction.NONE)
                positive_loss = tf.reduce_mean(positives_loss)
                negative_loss = tf.reduce_mean(negatives_loss)

                loss = positive_loss + negative_loss

                positive_loss_summary = tf.summary.scalar(name='PositiveLoss', tensor=positive_loss)
                negative_loss_summary = tf.summary.scalar(name='NegativeLoss', tensor=negative_loss)
                regularization_loss_summary = tf.summary.scalar(name='RegularizationLoss', tensor=regularization_loss)
                loss_summary = tf.summary.scalar(name='Loss', tensor=loss)



            with tf.variable_scope('Optimizer'):
                train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


            with tf.variable_scope('Accuracy'):
                accuracy, accuracy_function = tf.metrics.accuracy(
                    labels=tf.squeeze(positives_labels),
                    predictions=positives_predictions)

            training_summaries = tf.summary.merge(inputs=[
                positive_loss_summary,
                negative_loss_summary,
                regularization_loss_summary,
                loss_summary])

            with tf.variable_scope('Diagnostics'):
                distribution_png = tf.py_function(
                    func=PlotDistributionFunction(plotter),
                    inp=[positives_placeholder, positives_loss],
                    Tout=tf.string)
                distribution_image = tf.image.decode_png(contents=distribution_png, channels=4)
                distribution_image = tf.expand_dims(distribution_image, 0)
                distribution_image_summary = tf.summary.image('Distribution', distribution_image)

            validation_summaries = tf.summary.merge(inputs=[
                positive_loss_summary,
                distribution_image_summary
                ])

        class CircularQueue:

            #Constructor
            def __init__(self, contents):
                self.queue = list(contents)
                self.head = 0
                self.tail = len(contents)
                self.maxSize = len(contents)

            #Adding elements to the queue
            def enqueue(self,data):
                if self.size() == self.maxSize-1:
                    return ("Queue Full!")
                self.queue.append(data)
                self.tail = (self.tail + 1) % self.maxSize
                return True

            #Removing elements from the queue
            def dequeue(self):
                if self.size()==0:
                    return ("Queue Empty!")
                data = self.queue[self.head]
                self.head = (self.head + 1) % self.maxSize
                return data

            #Calculating the size of the queue
            def size(self):
                if self.tail>=self.head:
                    return (self.tail-self.head)
                return (self.maxSize - (self.head-self.tail))


        ######################
        # Initialize Compute #
        ######################

        with graph.as_default(), \
            tf.Session(graph=graph) as session, \
            tf.summary.FileWriter(logdir=log_directory + "/training", graph=session.graph) as training_writer, \
            tf.summary.FileWriter(logdir=log_directory + "/validation", graph=session.graph) as validation_writer:

            session.run(tf.local_variables_initializer())
            session.run(tf.global_variables_initializer())


            ###############
            # Train Model #
            ###############

            shuffled_positives = CircularQueue(np.random.permutation(positives).tolist())
            def sample_positives(n):
                return [shuffled_positives.dequeue() for i in range(n)]

            shuffled_negatives = CircularQueue(np.random.permutation(negatives).tolist())
            def sample_negatives(n):
                return [shuffled_negatives.dequeue() for i in range(n)]

            for epoch in range(num_epochs):
                _, batch_summary = session.run(
                    fetches=[train, training_summaries],
                    feed_dict={
                        positives_placeholder: sample_positives(batch_size),
                        negatives_placeholder: sample_negatives(batch_size)})
                training_writer.add_summary(batch_summary, epoch)

                if epoch % 100 == 0:
                    batch_summary = session.run(
                        fetches=validation_summaries,
                        feed_dict={positives_placeholder: positives})

                    validation_writer.add_summary(batch_summary, epoch)

                    logger.info("epoch: %3d" % (epoch))


            ##################
            # Evaluate Model #
            ##################

            # # What self equality probability is output for a data point in the training data?
            #
            # # What self equality probability is output for data points outside the training data?
