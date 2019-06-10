from rl.core import Policy

class RandomPolicy(Policy):

    def __init__(self, environment):
        self.environment = environment

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_action(self, observation):
        return self.environment.action_space.sample()

    def update(self, epoch, episodes):
        pass
