from rl.types import Episode

class Episodes:
    def __init__(self):
        self.episodes = []
        self.cumulative_length = 0

    def __iter__(self):
        return iter(self.episodes)

    def __getitem__(self, index):
        return self.episodes[index]

    def __len__(self):
        return len(self.episodes)

    def append(self, episode):
        self.episodes.append(episode)
        self.cumulative_length += len(episode)

    def num_steps(self):
        return self.cumulative_length

    def get_batch_observations(self):
        return [observation
            for episode in self.episodes
            for observation in episode.observations]

    def get_batch_actions(self):
        return [action
            for episode in self.episodes
            for action in episode.actions]
