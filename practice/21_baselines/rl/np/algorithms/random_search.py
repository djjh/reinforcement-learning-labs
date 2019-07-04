import numpy as np

from rl.core import Experience


class RandomSearch:

    def __init__(self, environment, random_seed, policy_factory, create_rollout, batch_size, explore):
        self._environment = environment
        self._random_seed = random_seed
        self._policy_factory = policy_factory
        self._create_rollout = create_rollout
        self._batch_size = batch_size
        self._explore = explore
        self._deterministic_update_policy = True

        self._observation_space = environment.observation_space
        self._action_space = environment.action_space
        self._policy = self._create_policy()
        self._policy_return = -np.inf
        self._policy_steps = -np.inf

    def _create_policy(self):
        return self._policy_factory.create_policy(
            observation_space=self._observation_space,
            action_space=self._action_space)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def action(self, observation, deterministic):
        action = self._policy.action(observation, deterministic)
        return action

    def update(self):
        def random_parameter(explore, p):
            return p + explore * np.random.standard_normal(p.shape)
        best_steps = -np.inf
        best_return = -np.inf
        best_policy = None
        parameters = self._policy.get_parameters()['model']
        for i in range(self._batch_size):
            policy = self._create_policy()
            policy.set_parameters({'model': random_parameter(self._explore, parameters)})
            episode = self._create_rollout(
                self._environment,
                policy,
                random_seed=self._random_seed,
                deterministic=self._deterministic_update_policy,
                render=False)
            episode_return = episode.get_return()
            episode_steps = len(episode)
            # print('\n', episode_return)
            if episode_return > best_return or episode_steps > best_steps:
                best_return = episode_return
                best_steps = episode_steps
                best_policy = policy
        if best_return >= self._policy_return or best_steps >= self._policy_steps:
            self._policy_return = best_return
            self._policy_steps = best_steps
            self._policy = best_policy
