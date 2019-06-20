import gym
import numpy as np
import random

class ProbabilityDistributionFactoryFactory:

    def probability_distribution_factory(self, space):
        if isinstance(space, gym.spaces.Discrete):
            return CategoricalProbabilityDistributionFactory(space.n)
        else:
            raise NotImplementedError()


class CategoricalProbabilityDistributionFactory:

    def __init__(self, n_categories):
        self.n_categories = n_categories

    def probability_distribution(self, logits):
        return CategoricalProbabilityDistribution(logits)

    def parameter_shape(self):
        return (self.n_categories,)

    def sample_shape(self):
        return ()

    def sample_dtype(self):
        return np.int32  # how to do this across frameworks? e.g. tf.int32...


class CategoricalProbabilityDistribution:

    def __init__(self, logits):
        self.logits = logits
        self.probabilities = logits / np.sum(logits)

    def mode(self):
        # The most frequent value is the same as argmax, e.g.
        # for PMF [0.1, 0.6, 0.3]
        # index 0 will be sampled 0.1 of the time,
        # index 1 will be sampled 0.6 of the time,
        # and index 2 will be sampled 0.3 of the time.
        # Thus the mode is the index of the value with the highest value,
        # aka argmax.
        return np.argmax(self.logits)

    def sample(self):
        return np.random.choice(len(self.probabilities), p=self.probabilities)


class LinearPolicy:

    def __init__(self, observation_space, action_space, pd_factory_factory):
        self.observation_space = observation_space
        self.action_space = action_space

        self.pd_factory = pd_factory_factory.probability_distribution_factory(action_space)

        self.observation_dimensions = np.prod(observation_space.shape)
        self.action_dimensions = np.prod(self.pd_factory.parameter_shape())

        self.model = np.random.randn(self.action_dimensions, self.observation_dimensions)

    def get_parameters(self):
        return { 'model': np.array(self.model) }

    def set_parameters(self, parameters):
        self.model = np.array(parameters['model'])

    # Should we have setters for input, a step method, and getters for output?
    # e.g.
    #     + def observe(self, observation)
    #     + def step(self)
    #     + def probabilities(self) -> ProbabilityDistribution
    #     + def action(self) -> Action
    # or simply
    #     + def step(self, observation) -> Action, ProbabilityDistribution
    #
    # Should deterministic be an argument or should we have mode and sample methods?
    # e.g.
    #     + def mode(self, observation) -> Action
    #     + def sample(self, observation) -> Action
    # or simply
    #     + def action(self, observation, deterministic) -> Action
    #
    def action(self, observation, deterministic):
        observation = observation.reshape(-1, 1)
        action_logits = self.model.dot(observation).flatten()
        action_distribution = self.pd_factory.probability_distribution(action_logits)
        # print(self.model.shape, observation.shape, action_logits.shape)
        if deterministic:
            return action_distribution.mode()
        else:
            return action_distribution.sample()


class LinearPolicyFactory:

    def create(self, observation_space, action_space, pd_factory_factory):
        return LinearPolicy(observation_space, action_space, pd_factory_factory)


class UniformRandom:

    def __init__(self, environment, policy_factory, create_rollout, batch_size, low, high):
        self.environment = environment
        self.policy_factory = policy_factory
        self.create_rollout = create_rollout
        self.batch_size = batch_size
        self.low = low
        self.high = high
        self.deterministic_update_policy = True

        self.observation_space = environment.observation_space
        self.action_space = environment.action_space

        self.graph = None
        self.session = None
        # parameters = None ?

        self.policy = self.policy_factory.create(
            observation_space=self.observation_space,
            action_space=self.action_space,
            pd_factory_factory=ProbabilityDistributionFactoryFactory())
        self.policy_return = -np.inf
        self.policy_steps = -np.inf

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def action(self, observation, deterministic):
        action = self.policy.action(observation, deterministic)
        return action

    def update(self):

        def uniform_sample_new_parameters(p):
            return np.random.uniform(self.low, self.high, p.shape)

        best_steps = -np.inf
        best_return = -np.inf
        best_policy = None
        parameters = self.policy.get_parameters()['model']

        for i in range(self.batch_size):

            policy = self.policy_factory.create(
                observation_space=self.observation_space,
                action_space=self.action_space,
                pd_factory_factory=ProbabilityDistributionFactoryFactory())
            policy.set_parameters({'model': uniform_sample_new_parameters(parameters)})
            episode = self.create_rollout(
                self.environment,
                policy,
                deterministic=self.deterministic_update_policy,
                render=False)

            episode_return = episode.get_return()
            episode_steps = len(episode)
            if episode_return > best_return or episode_steps > best_steps:
                best_return = episode_return
                best_steps = episode_steps
                best_policy = policy
        if best_return >= self.policy_return or best_steps >= self.policy_steps:
            print()
            print(best_return, self.policy_return, best_steps, self.policy_steps)
            self.policy_return = best_return
            self.policy_steps = best_steps
            self.policy = best_policy

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


def rollout(environment, policy, deterministic, render):
    episode = Episode()
    environment.seed(random_seed) # This is needed on every roll out for determinism...
    observation = environment.reset()
    while True:
        if render:
            environment.render()
        action = policy.action(observation, deterministic)
        next_observation, reward, done, info = environment.step(action)
        episode.append(observation, action, reward)
        if done:
            break;
        else:
            observation = next_observation
    return episode


# def main():

environment_name = 'CartPole-v0'
random_seed = 0
max_epochs = 10000

environment = gym.make(environment_name)
specification = gym.spec(environment_name)
algorithm = UniformRandom(
    environment=environment,
    policy_factory=LinearPolicyFactory(),
    create_rollout=rollout,
    batch_size=1,
    low=-1.0,
    high=1.0)

np.random.seed(random_seed)
random.seed(random_seed)
environment.seed(random_seed)

with algorithm, environment:

    max_episode_steps = specification.max_episode_steps
    reward_threshold = specification.reward_threshold
    has_reward_threshold = reward_threshold is not None

    for epoch in range(1, max_epochs+1):

        algorithm.update()

        episode_stepss = []
        episode_rewards = []
        required_wins = 100
        win_count = 0
        win = True

        while win and win_count < required_wins:
            policy = algorithm  # or should we have def policy(self) -> Policy ?
            episode = rollout(environment, policy, deterministic=True, render=False)
            episode_steps = len(episode)
            episode_reward = episode.get_return()
            episode_stepss.append(episode_steps)
            episode_rewards.append(episode_reward)
            win = has_reward_threshold and episode_reward >= reward_threshold
            if win:
                win_count += 1

        print('                                                                                 ',
            end="\r")
        print('epoch: {}, wins: {}, length: {}, reward: {}'.format(epoch, win_count, np.mean(episode_steps), np.mean(episode_rewards)),
            end="\r")

        if win:
            break

    policy = algorithm   # or should we have def policy(self) -> Policy ?
    episode = rollout(environment,  policy, deterministic=True, render=True)
    episode_steps = len(episode)
    episode_reward = episode.get_return()

    print('Epochs: {}'.format(epoch))
    if has_reward_threshold:
        print('Target -> length: {}, return: {}'.format(max_episode_steps, reward_threshold))
        print('Actual -> length: {}, return: {}'.format(episode_steps, episode_reward))
        win = has_reward_threshold and episode_reward >= reward_threshold
        print('Win!' if win else 'Lose!')
    else:
        print('Max return: {}'.format(episode_reward))
    if specification.nondeterministic:
        print('The environment was nondeterministic, so we should check the mean.');

    if environment.viewer and environment.viewer.window:
        environment.viewer.window.set_visible(False)
        environment.viewer.window.dispatch_events()


#
# if __name__ == '__main__':
#     main()
