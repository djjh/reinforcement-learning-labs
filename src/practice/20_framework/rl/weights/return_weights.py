from rl.weights import Weights

class ReturnWeights(Weights):

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_batch_weights(self, episodes):
        return [weight
            for episode in episodes
            for weight in self.get_weights(episode)]

    def get_weights(self, episode):
        return [sum(episode.rewards)] * len(episode.rewards)

    def update(self, epoch, episodes):
        pass
