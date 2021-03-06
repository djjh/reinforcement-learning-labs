import numpy as np
import tensorflow as tf

from . import ValueFunction
from rl.core import Episodes
from rl.core import discount_cumsum


class LinearValueFunction(ValueFunction):

    # TODO: (observation) or (observation, action) as input to the value function...
    def __init__(self, environment, input_factory, iterations, learning_rate):
        self._environment = environment
        self._input_factory = input_factory
        self._iterations = iterations
        self._learning_rate = learning_rate

        self._observation_space = environment.observation_space
        self._action_space = environment.action_space

        self._graph = tf.Graph()
        with self._graph.as_default():

            # set random seed here, or somewhere
            # tf.set_random_seed(self._random_seed)

            self._session = tf.Session(graph=self._graph)

            self._input = input_factory.create(self._observation_space, None)
            self._observations = self._input.get_input()
            self._model = self._create_model(self._input)
            self._values = self._model

            self._returns = tf.placeholder(shape=(None,), dtype=tf.float32)
            self._loss = tf.reduce_mean(tf.square(self._values - self._returns))
            self._train = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self._loss)


            self._session.run(tf.global_variables_initializer())
            self._session.run(tf.local_variables_initializer())

    def __enter__(self):
        self._session.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._session.__exit__(exc_type, exc_val, exc_tb)


    def _create_model(self, input):
        previous_layer = input.get_input()
        return  tf.layers.Dense(units=1)(previous_layer)

    def get_parameters(self):
        raise NotImplementedError()

    def set_parameters(self, parameters):
        raise NotImplementedError();

    # def get_value(self, inp):
    #
    #     observation = observation.reshape(1, -1)
    #     values = self._session.run(
    #         fetches=self._values,
    #         feed_dict={self._observations: observation})
    #     return values[0]

    def get_values(self, episodes):
        if not isinstance(episodes, Episodes):
            episodes = [episodes]

        observations = np.array([observation
            for episode in episodes
            for observation in episode.get_observations()])

        values = self._session.run(
            fetches=self._values,
            feed_dict={self._observations: observations})

        return values.flatten()


    def update(self, observations, returns):
        for iteration in range(self._iterations):
            self._session.run(
                fetches=self._train,
                feed_dict={
                    self._observations: observations,
                    # self._actions: actions,
                    self._returns: returns})
