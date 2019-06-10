
class Episode:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    def append(self, observation, action, reward):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)

    def get_return(self):
        return sum(self.rewards)

    def __len__(self):
        return len(self.observations)
