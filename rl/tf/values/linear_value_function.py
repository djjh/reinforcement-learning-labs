import numpy as np
import tensorflow as tf

from rl.core import Episodes


class LinearValueFunction:

    # TODO: (observation) or (observation, action) as input to the value function...
    def __init__(self, environment, input_factory, iterations, learning_rate, session):
        self._environment = environment
        self._input_factory = input_factory
        self._iterations = iterations
        self._learning_rate = learning_rate
        self._session = session

        self._observation_space = environment.observation_space
        self._action_space = environment.action_space
        self._input = input_factory.create(self._observation_space, None)
        self._observations =self._input.get_input()
        self._model = self._create_model(self._input)
        self._values = self._model

        self._returns = tf.placeholder(shape=(None,), dtype=tf.float32)
        self._loss = tf.reduce_mean(tf.square(self._values - self._returns))
        self._train = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self._loss)

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
        observations = np.array([o for e in input for o in e.get_observations()])
        print(observations.shape)

        if actions:
            actions = np.array(actions)
            print(actions.shape)

        values = self._session.run(
            fetches=self._values,
            feed_dict={self._observations: observations})
        print(values.shape)
        return values.flatten()


    def update(self, observations, returns, actions=None):
        for iteration in range(self._iterations):
            self._session.run(
                fetches=self._train,
                feed_dict={
                    self._observations: observations,
                    # self._actions: actions,
                    self._returns: returns})
