import numpy as np
import tensorflow as tf

from rl.density import Density

def wrap(layers, previous_layer):
    for layer in layers:
        previous_layer = layer(previous_layer)
    return previous_layer

class ExemplarDensity(Density):

    def __init__(self, dimensions, log_directory, random_seed, learning_rate):
        self.dimensions = dimensions
        self.log_directory = log_directory
        self.random_seed = random_seed
        self.learning_rate = learning_rate

        self.encoder_dimensions = 12
        self.feature_dimensions = 12
        self.model_dimensions = 12

        self.graph = tf.Graph()
        update_summaries = []
        estimate_summaries = []
        log_summaries = []
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            with tf.variable_scope('X'):
                self.x_placeholder = tf.placeholder(shape=(None, self.dimensions), dtype=tf.float32)
                for i, x_dimension in enumerate(tf.split(self.x_placeholder, self.dimensions, axis=1)):
                    update_summaries.append(tf.summary.histogram('X{}'.format(i), x_dimension))
                self.x = self.x_placeholder
            with tf.variable_scope('Y'):
                self.y_placeholder = tf.placeholder(shape=(None, self.dimensions), dtype=tf.float32)
                # observations0, observations1 = tf.split(self.y_placeholder, [1, 1], 1)
                for i, y_dimension in enumerate(tf.split(self.y_placeholder, self.dimensions, axis=1)):
                    update_summaries.append(tf.summary.histogram('Y{}'.format(i), y_dimension))
                self.y = self.y_placeholder
            with tf.variable_scope('XLabels'):
                x_labels = tf.fill(value=1, dims=(tf.shape(self.x)[0], 1))
            with tf.variable_scope('YLabels'):
                y_labels = tf.fill(value=0, dims=(tf.shape(self.y)[0], 1))
            with tf.variable_scope('Encoder'):
                def relu(output_dimensions):
                    init = 1.0 / np.sqrt(self.encoder_dimensions)
                    kernel_initializer = tf.initializers.random_uniform(minval=-init, maxval=init)
                    return tf.layers.Dense(units=output_dimensions, kernel_initializer=kernel_initializer, activation=tf.nn.relu)
                def linear(output_dimensions):
                    init = 1.0 / np.sqrt(self.encoder_dimensions)
                    kernel_initializer = tf.initializers.random_uniform(minval=-init, maxval=init)
                    return tf.layers.Dense(units=output_dimensions, kernel_initializer=kernel_initializer, activation=None)
                encoder_layers = [relu(self.encoder_dimensions), relu(self.encoder_dimensions), linear(self.feature_dimensions)]
                x_mean = wrap(encoder_layers, self.x)
                y_mean = wrap(encoder_layers, self.y)
                x_features = x_mean
                y_features = y_mean
            with tf.variable_scope('Model'):
                def tanh(output_dimensions):
                    init = 1.0/np.sqrt(self.feature_dimensions)
                    kernel_initializer = tf.initializers.random_uniform(minval=-init, maxval=init)
                    return tf.layers.Dense(units=output_dimensions, kernel_initializer=kernel_initializer, activation=tf.tanh)
                model_layers = [tanh(self.model_dimensions), tanh(1)]
                x_logits = wrap(model_layers, tf.concat(values=[x_features, x_features], axis=1))
                y_logits = wrap(model_layers, tf.concat(values=[x_features, y_features], axis=1))
            with tf.variable_scope('Loss'):
                x_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=x_labels, logits=x_logits)
                labels = tf.concat(values=[x_labels, y_labels], axis=1)
                logits = tf.concat(values=[x_logits, y_logits], axis=1)
                self.loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)
                update_summaries.append(tf.summary.scalar('XLoss', x_loss))
                update_summaries.append(tf.summary.scalar('Loss', self.loss))
            with tf.variable_scope('Estimate'):
                d = tf.nn.sigmoid(x_logits)
                x_probabilities = (1 - d) / d
                self.estimate_operation = x_probabilities
                estimate_summaries.append(tf.summary.histogram('XProbabilities', x_probabilities))
            with tf.variable_scope('Optimizer'):
                self.training_operation = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            with tf.variable_scope('Rewards'):
                self.actual_distribution_png_placeholder = tf.placeholder(shape=(), dtype=tf.string)
                distribution_image = tf.image.decode_png(contents=self.actual_distribution_png_placeholder, channels=4)
                distribution_image = tf.expand_dims(distribution_image, 0)
                distribution_image_summary = tf.summary.image('Actual', distribution_image)
                log_summaries.append(distribution_image_summary)

                self.weights_png_placeholder = tf.placeholder(shape=(), dtype=tf.string)
                distribution_image = tf.image.decode_png(contents=self.weights_png_placeholder, channels=4)
                distribution_image = tf.expand_dims(distribution_image, 0)
                distribution_image_summary = tf.summary.image('Weights', distribution_image)
                log_summaries.append(distribution_image_summary)

                self.bonuses_png_placeholder = tf.placeholder(shape=(), dtype=tf.string)
                distribution_image = tf.image.decode_png(contents=self.bonuses_png_placeholder, channels=4)
                distribution_image = tf.expand_dims(distribution_image, 0)
                distribution_image_summary = tf.summary.image('Bonuses', distribution_image)
                log_summaries.append(distribution_image_summary)

            # with tf.variable_scope('Density'):
            #     self.predicted_distribution_png_placeholder = tf.placeholder(shape=(), dtype=tf.string)
            #     distribution_image = tf.image.decode_png(contents=self.predicted_distribution_png_placeholder, channels=4)
            #     distribution_image = tf.expand_dims(distribution_image, 0)
            #     distribution_image_summary = tf.summary.image('Predicted', distribution_image)
            #     log_summaries.append(distribution_image_summary)

            self.estimate_summary = tf.summary.merge(inputs=estimate_summaries)
            self.update_summary = tf.summary.merge(inputs=update_summaries)
            self.log_summary = tf.summary.merge(inputs=log_summaries)

        self.session = tf.Session(graph=self.graph)
        self.estimate_writer = tf.summary.FileWriter(logdir=self.log_directory + "/density/estimate", graph=self.graph)
        self.update_writer = tf.summary.FileWriter(logdir=self.log_directory + "/density/update", graph=self.graph)

    def __enter__(self):
        self.session.__enter__()
        self.estimate_writer.__enter__()
        self.update_writer.__enter__()
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.update_writer.__exit__(exc_type, exc_val, exc_tb)
        self.estimate_writer.__exit__(exc_type, exc_val, exc_tb)
        self.session.__exit__(exc_type, exc_val, exc_tb)

    def estimate(self, epoch, x):
        probabilities, summary = self.session.run(
            fetches=[self.estimate_operation, self.estimate_summary],
            feed_dict={
                self.x_placeholder: np.array(x)})
        self.estimate_writer.add_summary(summary, epoch)
        return np.squeeze(probabilities, axis=1)

    def update(self, epoch, x, y, last):
        _, loss, summary = self.session.run(
            fetches=[self.training_operation, self.loss, self.update_summary],
            feed_dict={
                self.x_placeholder: np.array(x),
                self.y_placeholder: np.array(y)})
        if last:
            self.update_writer.add_summary(summary, epoch)
        return loss

    def log(self, epoch, x, w, rb, grid):
        # probabilities = self.estimate(epoch, grid)
        summary = self.session.run(
            fetches=self.log_summary,
            feed_dict={
                self.actual_distribution_png_placeholder: to_png(histogram(x)),
                # self.predicted_distribution_png_placeholder: to_png(probability_contours(grid, probabilities)),
                self.weights_png_placeholder: to_png(weighted_histogram(x, w)),
                self.bonuses_png_placeholder: to_png(weighted_histogram(x, rb))})
        self.update_writer.add_summary(summary, epoch)
