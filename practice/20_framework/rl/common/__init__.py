import numpy as np

from rl.types import Episode

class Sampler:
    def sample(self, probabilities):
        raise NotImplementedError()

class DeterminisiticSampler(Sampler):
    def sample(self, probabilities):
        return np.argmax(probabilities)

class StochasticSampler(Sampler):
    def sample(self, probabilities):
        return np.random.choice(range(len(probabilities)), p=probabilities)



def rollout(environment, policy, render):
    episode = Episode()
    observation = environment.reset()
    while True:
        if render:
            environment.render()
        action = policy(observation)
        next_observation, reward, done, info = environment.step(action)
        episode.append(observation, action, reward)
        if done:
            break;
        else:
            observation = next_observation
    return episode
