

class RecordingPolicy:

    def __init__(self, policy):
        self._policy = policy
        self._probabilities = []

    def action(self, observation, deterministic):
        action, probability_distribution = self._policy.step(observation, deterministic)
        self._probabilities.append(probability_distribution.probabilities())
        return action

    def get_probabilities(self):
        return self._probabilities
