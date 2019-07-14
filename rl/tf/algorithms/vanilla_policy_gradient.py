import numpy as np
import tensorflow as tf

from rl.core import Episodes


from rl.tf.advantages import *

from rl.tf.values import *



class VanillaPolicyGradient:

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
            # self._advantage_function.initialize()
            # = self._advantage_function_factory.create_advantage_function()
            # GeneralizedAdvantageFunction(
            #     value_function=LinearValueFunction(
            #         observation_space=environment.observation_space,
            #         action_space=environment.action_space,
            #         input_factory=self._policy._input_factory, ## HACK!
            #         iterations=20,                             ## HACK!
            #         learning_rate=self._policy._learning_rate, ## HACK!
            #         session=self._session))
            # self._advantage_function = CumulativeRewardAdvantageFunction()
            # self._advantage_function = RewardToGoAdvantageFunction()

        self._policy_return = -np.inf
        self._policy_steps = -np.inf

    def __enter__(self):
        self._session.__enter__()
        self._session.run(tf.global_variables_initializer())
        self._session.run(tf.local_variables_initializer())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._session.__exit__(exc_type, exc_val, exc_tb)

    def action(self, observation, deterministic):
        return self._policy.action(observation, deterministic)

    def update(self):
        experience = self._generate_experience()
        self._policy.update(
            observations=self._get_batch_observations(experience),
            actions=self._get_batch_actions(experience),
            advantages=self._advantage_function.get_advantages(experience))
        self._advantage_function.update(experience=experience)

    def _generate_experience(self):
        experience = Episodes()
        while experience.num_steps() < self._min_steps_per_batch:
            episode = self._Rollout(
                environment=self._environment,
                policy=self._policy,
                random_seed=self._random_seed,
                deterministic=False,
                render=False)
            experience.append(episode)
        return experience

    def _get_batch_observations(self, experience):
        return [observation
            for episode in experience
            for observation in episode.get_observations() ]

    def _get_batch_actions(self, experience):
        return [action
            for episode in experience
            for action in episode.get_actions() ]
