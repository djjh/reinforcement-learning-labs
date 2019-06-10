import nevergrad as ng
import numpy as np
import tensorflow as tf

from rl.types import Episode, Episodes
from rl.core import Algorithm, Policy, PolicyFactory
from rl.weights import RewardToGoWeights
from gym.spaces import Box, Discrete
from sklearn.preprocessing import PolynomialFeatures


###############################
# Deterministic Linear Policy #
###############################

class DeterministicDiscreteLinearNumpyArgMaxPolicy(Policy):

    def __init__(self, environment):
        assert isinstance(environment.action_space, Discrete), "Must be Discrete action space."

        self.environment = environment
        self.observation_dimensions = np.prod(self.environment.observation_space.shape)
        self.action_dimensions = self.environment.action_space.n
        self.model = np.random.randn(self.action_dimensions, self.observation_dimensions)
        self.deterministic = True

    def get_parameters(self):
        return [self.model]

    def set_parameters(self, parameters):
        assert len(parameters) == 1, "Invalid number of parameters."
        assert parameters[0].shape == self.model.shape, "Invalid parameter shape."
        self.model = parameters[0]

    def set_deterministic(self, deterministic):
        self.deterministic = deterministic

    def __call__(self, observation):
        probabilities = self.model.dot(observation.reshape(-1, 1))
        if self.deterministic:
            return np.argmax(probabilities)
        else:
            return np.random.choice(self.action_dimensions, p=probabilities)




############################
# Stochastic Linear Policy #
############################

class StochasticDiscreteLinearNumpySoftMaxPolicy(Policy):
    def __init__(self, environment):
        assert isinstance(environment.action_space, Discrete), "Must be Discrete action space."

        self.environment = environment
        self.observation_dimensions = np.prod(self.environment.observation_space.shape)
        self.action_dimensions = self.environment.action_space.n
        self.model = np.random.randn(self.action_dimensions, self.observation_dimensions)
        self.deterministic = False

    def get_parameters(self):
        return [self.model]

    def set_parameters(self, parameters):
        assert len(parameters) == 1, "Invalid number of parameters."
        assert parameters[0].shape == self.model.shape, "Invalid parameter shape."
        self.model = parameters[0]

    def __call__(self, observation):
        z = self.model.dot(observation.reshape(-1, 1)).flatten()
        if self.deterministic:
            return np.argmax(z)
        else:
            exp = np.exp(z)
            softmax = exp / np.sum(exp)
            return np.random.choice(self.action_dimensions, p=softmax)

class StochasticLinearPolicyFactory(PolicyFactory):

    def create_policy(self, environment):
        if isinstance(environment.action_space, Box):
            raise NotImplementedError()
        elif isinstance(environment.action_space, Discrete):
            return StochasticDiscreteLinearNumpySoftMaxPolicy(environment)
        else:
            raise NotImplementedError()

#####################
# Polynomial Policy #
#####################

class PolynomialPolicy(Policy):

    def __init__(self, environment):
        self.environment = environment
        self.observation_dimensions = np.prod(self.environment.observation_space.shape)
        self.action_dimensions = self.environment.action_space.n
        self.degree = 2
        self.input_dimensions = self.observation_dimensions * self.degree
        self.model = np.random.randn(self.action_dimensions, self.input_dimensions)

    def get_parameters(self):
        return [self.model]

    def set_parameters(self, parameters):
        assert len(parameters) == 1, "Invalid number of parameters."
        assert parameters[0].shape == self.model.shape, "Invalid parameter shape."
        self.model = parameters[0]

    def __call__(self, observation):
        input = np.vander(observation, self.degree).flatten()
        return np.argmax(self.model.dot(input))

class PolynomialPolicyFactory(PolicyFactory):

    def create_policy(self, environment):
        return PolynomialPolicy(environment)



####################
# Custom Algorithm #
####################

class RandomSearch(Algorithm):

    """
    Algorithm as per https://en.wikipedia.org/wiki/Random_search:
        Let f: ℝn → ℝ be the fitness or cost function which must be minimized. Let x ∈ ℝn designate a position or candidate solution in the search-space. The basic RS algorithm can then be described as:

        Initialize x with a random position in the search-space.
        Until a termination criterion is met (e.g. number of iterations performed, or adequate fitness reached), repeat the following:
        Sample a new position y from the hypersphere of a given radius surrounding the current position x (see e.g. Marsaglia's technique for sampling a hypersphere.)
        If f(y) < f(x) then move to the new position by setting x = y
    """

    def __init__(self, environment, policy_factory, rollout_function, batch_size, explore):
        self.environment = environment
        self.policy_factory = policy_factory
        self.rollout_function = rollout_function
        self.batch_size = batch_size
        self.explore = explore

        self.policy = policy_factory.create_policy(self.environment)
        self.policy_return = -np.inf

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_policy(self):
        return self.policy

    def update(self):

        def random_parameter(explore, p):
            return p + explore * np.random.standard_normal(p.shape)

        def random_parameters(explore, ps):
            return [random_parameter(explore, p) for p in ps]

        best_return = -np.inf
        best_policy = None
        parameters = self.policy.get_parameters()

        for i in range(self.batch_size):
            policy = self.policy_factory.create_policy(self.environment)
            policy.set_parameters(random_parameters(self.explore, parameters))
            episode = self.rollout_function(self.environment, policy, render=False)
            episode_return = episode.get_return()
            if episode_return > best_return:
                best_return = episode_return
                best_policy = policy
        if best_return > self.policy_return:
            self.policy_return = best_return
            self.policy = best_policy


class DiscreteLinearPolicyGradient(Algorithm, Policy):

    def __init__(
        self,
        environment,
        rollout_function,
        min_steps_per_batch):

        self.environment = environment
        self.rollout_function = rollout_function
        self.min_steps_per_batch = min_steps_per_batch

        self.observation_dimensions  = np.prod(self.environment.observation_space.shape)
        self.action_dimensions = self.environment.action_space.n
        self.W = np.random.rand(self.observation_dimensions, self.action_dimensions)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_parameters(self):
        return self.W

    def set_parameters(self, parameters):
        raise NotImplementedError()

    def __call__(self, observation):
        return np.argmax(self.probabilities(observation))

    def probabilities(self, observation):
        z = observation.dot(self.W)
        exp = np.exp(z)
        return exp / np.sum(exp)

    def get_policy(self):
        return self

    def update(self):
        self.deterministic = False
        episodes = Episodes()
        episodes_probabilities = []

        while episodes.num_steps() < self.min_steps_per_batch:
            episode_probabilities = []
            def recordingPolicy(observation):
                probabilities = self.probabilities(observation)
                episode_probabilities.append(probabilities)
                return np.random.choice(self.action_dimensions, p=probabilities)
            episode = self.rollout_function(self.environment, recordingPolicy, render=False)
            episodes.append(episode)
            episodes_probabilities.append(episode_probabilities)

        grads = []
        rewards = []
        for i in range(len(episodes)):
            episode = episodes[i]
            episode_probabilities = episodes_probabilities[i]
            for j in range(len(episode)):
                observation = episode.observations[j]
                action = episode.actions[j]
                reward = episode.rewards[j]
                probabilities = episode_probabilities[j]

                softmax = probabilities
                s = softmax.reshape(-1,1)
                grad_softmax = (np.diagflat(s) - np.dot(s, s.T))
                grad_softmax = grad_softmax[action,:]
                grad_log = grad_softmax / softmax[action]

                grads.append(observation[None,:].T.dot(grad_log[None,:]))
                rewards.append(reward)

        for i in range(len(grads)):
            self.W += 0.000025 * grads[i] * sum([ r * (0.99 ** r) for t,r in enumerate(rewards[i:])])


########################
# Nevergrad Algorithms #
########################

class Nevergrad(Algorithm):

    def __init__(self, environment, policy_factory, rollout_function, optimizer, budget, low, high):
        self.environment = environment
        self.policy_factory = policy_factory
        self.rollout_function = rollout_function

        self.policy = policy_factory.create_policy(self.environment)
        self.policy_return = -np.inf

        self.shape = self.policy.get_parameters()[0].shape
        self.dims = np.prod(self.policy.get_parameters()[0].shape)

        instrumentation = ng.Instrumentation(ng.var.Array(self.dims).bounded(low, high))

        self.optimizer = ng.optimizers.OnePlusOne(instrumentation=instrumentation, budget=budget)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_policy(self):
        return self.policy

    def update(self):

        def rewards(parameters):

            parameters = [np.array(parameters).reshape(self.shape)]
            # May want to include multiple iterations here in case of a stochastic environment or policy.
            policy = self.policy_factory.create_policy(self.environment)
            policy.set_parameters(parameters)
            episode = self.rollout_function(self.environment, policy, render=False)
            return -episode.get_return()  # nevergrad optimizers minimize!

        recommendation = self.optimizer.optimize(rewards)
        # print(recommendation)
        parameters = recommendation.args[0]
        parameters = [np.array(parameters).reshape(self.shape)]

        policy = self.policy_factory.create_policy(self.environment)
        policy.set_parameters(parameters)
        self.policy = policy


class OnePlusOne(Algorithm):

    def __init__(self, environment, policy_factory, rollout_function, budget, low, high):
        self.algorithm = Nevergrad(
            environment=environment,
            policy_factory=policy_factory,
            rollout_function=rollout_function,
            optimizer=ng.optimizers.OnePlusOne,
            budget=budget,
            low=low,
            high=high)

    def __enter__(self):
        self.algorithm.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.algorithm.__exit__(exc_type, exc_val, exc_tb)

    def get_policy(self):
        return self.algorithm.get_policy()

    def update(self):
        self.algorithm.update()


class TwoPointsDE(Algorithm):

    def __init__(self, environment, policy_factory, rollout_function, budget, low, high):
        self.algorithm = Nevergrad(
            environment=environment,
            policy_factory=policy_factory,
            rollout_function=rollout_function,
            optimizer=ng.optimizers.TwoPointsDE,
            budget=budget,
            low=low,
            high=high)

    def __enter__(self):
        self.algorithm.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.algorithm.__exit__(exc_type, exc_val, exc_tb)

    def get_policy(self):
        return self.algorithm.get_policy()

    def update(self):
        self.algorithm.update()


class CMA(Algorithm):

    def __init__(self, environment, policy_factory, rollout_function, budget, low, high):
        self.algorithm = Nevergrad(
            environment=environment,
            policy_factory=policy_factory,
            rollout_function=rollout_function,
            optimizer=ng.optimizers.CMA,
            budget=budget,
            low=low,
            high=high)

    def __enter__(self):
        self.algorithm.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.algorithm.__exit__(exc_type, exc_val, exc_tb)

    def get_policy(self):
        return self.algorithm.get_policy()

    def update(self):
        self.algorithm.update()


#########################
# Tensorflow Algorithms #
#########################

class DiscreteLinearTensorflowPolicyGradient(Algorithm, Policy):

    def __init__(
        self,
        environment,
        rollout_function,
        min_steps_per_batch):

        self.environment = environment
        self.rollout_function = rollout_function
        self.min_steps_per_batch = min_steps_per_batch

        self.weights_helper = RewardToGoWeights(discount=0.1)

        self.random_seed = 0

        # self.learning_rate = 1e-3
        # self.linear = False
        self.learning_rate = 1e-5
        self.linear = True

        self.observation_dimensions = np.prod(self.environment.observation_space.shape)
        self.action_dimensions = self.environment.action_space.n
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed);
            self.observations = tf.placeholder(shape=(None, self.observation_dimensions), dtype=tf.float32)
            self.actions = tf.placeholder(shape=(None,), dtype=tf.int32)
            self.weights = tf.placeholder(shape=(None,), dtype=tf.float32)

            if self.linear:
                self.logits = tf.layers.Dense(units=self.action_dimensions, activation=None, use_bias=False)(self.observations)
            else:
                layers = [
                    tf.layers.Dense(units=32, activation=tf.tanh),
                    tf.layers.Dense(units=self.action_dimensions, activation=None)]
                previous_layer = self.observations
                for layer in layers:
                    previous_layer = layer(previous_layer)
                self.logits = previous_layer

            self.probabilities = tf.nn.softmax(self.logits)

            self.action_masks = tf.one_hot(indices=self.actions, depth=self.action_dimensions)
            self.log_probabilities = tf.reduce_sum(self.action_masks * tf.nn.log_softmax(self.logits), axis=1)
            self.psuedo_loss = -tf.reduce_mean(self.weights * self.log_probabilities)
            self.optimize = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.psuedo_loss)
        self.session = tf.Session(graph=self.graph)

        self.deterministic = False

    def __enter__(self):
        self.session.__enter__()
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.__exit__(exc_type, exc_val, exc_tb)

    def get_parameters(self):
        return self.session.run(
            fetches=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),
            feed_dict={})

    def set_parameters(self, parameters):
        # for variable, parameter in zip(self.get_parameters(), parameters):
        #     variable.assign(parameter)
        pass

    def __call__(self, observation):
        probabilities = self.session.run(
            fetches=self.probabilities,
            feed_dict={self.observations: np.array(observation.reshape(1, -1))})[0]
        if self.deterministic:
            return np.argmax(probabilities)
        else:
            normalized_probabilities = np.array(probabilities) / probabilities.sum()
            return np.random.choice(self.action_dimensions, p=normalized_probabilities)

    def get_policy(self):
        return self

    def update(self):
        self.deterministic = False
        episodes = Episodes()
        while episodes.num_steps() < self.min_steps_per_batch:
            episode = self.rollout_function(self.environment, self, render=False)
            episodes.append(episode)
        _, psuedo_loss = self.session.run(
            fetches=[self.optimize, self.psuedo_loss],
            feed_dict={
                self.observations: episodes.get_batch_observations(),
                self.actions: episodes.get_batch_actions(),
                self.weights: self.weights_helper.get_batch_weights(episodes)})
        self.deterministic = True
