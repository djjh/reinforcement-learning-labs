import numpy as np
import tensorflow as tf

from rl.core import Episodes
from rl.core.model import Algorithm


class VanillaPolicyGradient(Algorithm):

    def __init__(self, environment, random_seed, policy_factory, advantage_function,
            Rollout, min_steps_per_batch):

        self._environment = environment
        self._random_seed = random_seed
        self._policy_factory = policy_factory
        self._advantage_function = advantage_function
        self._Rollout = Rollout
        self._min_steps_per_batch = min_steps_per_batch

        self._graph = tf.Graph()
        with self._graph.as_default():

            # set random seed here, or somewhere
            tf.set_random_seed(self._random_seed)

            self._session = tf.Session(graph=self._graph)
            self._policy = policy_factory.create_policy(
                observation_space=environment.observation_space,
                action_space=environment.action_space,
                session=self._session)

            self._session.run(tf.global_variables_initializer())
            self._session.run(tf.local_variables_initializer())

        self._policy_return = -np.inf
        self._policy_steps = -np.inf

    def __enter__(self):
        self._session.__enter__()
        self._advantage_function.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._advantage_function.__exit__(exc_type, exc_val, exc_tb)
        self._session.__exit__(exc_type, exc_val, exc_tb)

    def action(self, observation, deterministic):
        return self._policy.action(observation, deterministic)

    def update(self):
        episodes = self._generate_episodes()
        self._policy.update(
            observations=self._get_batch_observations(episodes),
            actions=self._get_batch_actions(episodes),
            advantages=self._advantage_function.get_advantages(episodes))
        self._advantage_function.update(episodes=episodes)

    def _generate_episodes(self):
        episodes = Episodes()
        while episodes.num_steps() < self._min_steps_per_batch:
            episode = self._Rollout(
                environment=self._environment,
                policy=self._policy,
                random_seed=self._random_seed,
                deterministic=False,
                render=False)
            episodes.append(episode)
        return episodes

    def _get_batch_observations(self, episodes):
        return [observation
            for episode in episodes
            for observation in episode.get_observations() ]

    def _get_batch_actions(self, episodes):
        return [action
            for episode in episodes
            for action in episode.get_actions() ]
