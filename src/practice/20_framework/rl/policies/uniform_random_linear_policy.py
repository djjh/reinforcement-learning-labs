import numpy as np

from rl.core import Policy


class UniformRandomLinearPolicy(Policy):

    def __init__(self, environment, rollout_function, batch_size):
        self.environment = environment
        self.rollout_function = rollout_function
        self.batch_size = batch_size

        self.observation_dimensions = np.prod(self.environment.observation_space.shape)
        self.action_dimensions = self.environment.action_space.n

        self.model = np.random.randn(self.action_dimensions, self.observation_dimensions)
        self.model_return = -np.inf

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_action(self, observation):
        return np.argmax(self.model.dot(observation))

    def update(self):
        linf_norm = 3
        best_return = -np.inf
        best_model = np.empty((self.action_dimensions, self.observation_dimensions))
        for i in range(self.batch_size):
            model = np.random.uniform(-linf_norm,linf_norm,(self.action_dimensions, self.observation_dimensions))
            def action(observation):
                return np.argmax(model.dot(observation))
            episode = self.rollout_function(self.environment, action, render=False)
            episode_return = episode.get_return()
            if episode_return > best_return:
                best_return = episode_return
                best_model = model
        if best_return > self.model_return:
            self.model_return = best_return
            self.model = best_model
