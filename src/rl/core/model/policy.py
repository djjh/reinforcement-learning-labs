
class Policy:

    def __init__(self):
        raise NotImplementedError()

    def get_parameters(self):
        raise NotImplementedError()

    def set_parameters(self, parameters):
        raise NotImplementedError();

    def get_observations(self):
        raise NotImplementedError()

    def get_observations(self):
        raise NotImplementedError()

    def get_actions(self):
        raise NotImplementedError()

    def get_deterministic_actions(self):
        raise NotImplementedError()


    def get_log_probabilities(self):
        raise NotImplementedError()

    # Non-batch
    def action(self, observation, deterministic):
        raise NotImplementedError()

    def step(self, observation, deterministic):
        raise NotImplementedError()

    def update(self, observations, actions, advantages):
        raise NotImplementedError()
