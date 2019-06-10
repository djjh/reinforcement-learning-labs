# This experiment benchmarks different algorithms to optimize a linear policy
# on a variety of environments.

import gym
import logging
import numpy as np
import random

from rl.algorithms import \
    UniformRandom
from rl.common import rollout
from rl.types import Episode, Episodes
from rl.logging import get_expermiment_logging_directory
from rl.policies import DeterministicLinearPolicyFactory
# from rl.policies import \
#     RandomSearch, \
#     UniformRandom, \
#     OnePlusOne, \
#     TwoPointsDE, CMA, \
#     DeterministicLinearPolicyFactory, \
#     StochasticLinearPolicyFactory, \
#     PolynomialPolicyFactory, \
#     DiscreteLinearTensorflowPolicyGradient, \
#     DiscreteLinearPolicyGradient



######################
# Initialize Logging #
######################

logger = logging.getLogger(__name__)
logging.getLogger('rl.policies.lstm_vanilla_policy').setLevel(logging.INFO)
logging.getLogger('rl.weights.reward_to_go_weights').setLevel(logging.INFO)
log_directory = get_expermiment_logging_directory(__file__)
logger.info("Logging to directory: {}".format(log_directory))


###########################
# Initialize Random Seeds #
###########################
random_seed = 0
np.random.seed(random_seed)
random.seed(random_seed)


###############################
# Initialize Experiment Setup #
###############################

class Experiment:

    def __init__(self, max_epochs, random_seed, environment_names, algorithm_functions):
        self.max_epochs = max_epochs
        self.random_seed = random_seed
        self.environment_names = environment_names
        self.algorithm_functions = algorithm_functions

    def run(self):
        for environment_name in self.environment_names:
            for algorithm_function in self.algorithm_functions:

                environment = gym.make(environment_name)
                specification = gym.spec(environment_name)
                environment.seed(self.random_seed)
                algorithm = algorithm_function(environment)

                with algorithm, environment:

                    max_episode_steps = specification.max_episode_steps
                    reward_threshold = specification.reward_threshold
                    has_reward_threshold = reward_threshold is not None

                    for epoch in range(1, self.max_epochs+1):

                        algorithm.update()

                        episode_stepss = []
                        episode_rewards = []
                        required_wins = 100
                        win_count = 0
                        win = True

                        while win and win_count < required_wins:
                            policy = algorithm.get_policy()
                            policy.deterministic = True
                            episode = rollout(environment, policy, render=False)
                            policy.deterministic = False
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

                    policy = algorithm.get_policy()
                    policy.deterministic = True
                    episode = rollout(environment,  policy, render=True)
                    policy.deterministic = False
                    episode_steps = len(episode)
                    episode_reward = episode.get_return()

                    logger.info('Epochs: {}'.format(epoch))
                    if has_reward_threshold:
                        logger.info('Target -> length: {}, return: {}'.format(max_episode_steps, reward_threshold))
                        logger.info('Actual -> length: {}, return: {}'.format(episode_steps, episode_reward))
                        win = has_reward_threshold and episode_reward >= reward_threshold
                        logger.info('Win!' if win else 'Lose!')
                    else:
                        logger.info('Max return: {}'.format(episode_reward))
                    if specification.nondeterministic:
                        logger.info('The environment was nondeterministic, so we should check the mean.');

                    if environment.viewer and environment.viewer.window:
                        environment.viewer.window.set_visible(False)
                        environment.viewer.window.dispatch_events()


# #################################
# # CartPole-v0 : Box -> Discrete #
# #################################
environment_name = 'CartPole-v0'
max_epochs = 10000
algorithm_factories = []

# Win at 7 epochs.
def makeUniformRandom(environment):
    logger.info("    ---- UniformRandom ----    ");

    return UniformRandom(
        environment=environment,
        policy_factory=DeterministicLinearPolicyFactory(),
        rollout_function=rollout,
        batch_size=1,
        low=-1.0,
        high=1.0)
algorithm_factories.append(makeUniformRandom)

# # Winning at 5 epochs.
# def makeRandomSearch(environment):
#     logger.info("    ---- RandomSearch ----    ");
#     return RandomSearch(
#         environment=environment,
#         policy_factory=DeterministicLinearPolicyFactory(),
#         rollout_function=rollout,
#         batch_size=1,
#         explore=10.0)
# algorithm_factories.append(makeRandomSearch)

# # Win at 1 epoch.
# def makeOnePlusOne(environment):
#     logger.info("    ---- OnePlusOne ----    ");
#     return OnePlusOne(
#         environment=environment,
#         policy_factory=DeterministicLinearPolicyFactory(),
#         rollout_function=rollout,
#         budget=100,
#         low=-1.0,
#         high=1.0)
# algorithm_factories.append(makeOnePlusOne)
#
# # Win at 1 epoch.
# def makeTwoPointsDE(environment):
#     logger.info("    ---- TwoPointsDE ----    ");
#     return TwoPointsDE(
#         environment=environment,
#         policy_factory=DeterministicLinearPolicyFactory(),
#         rollout_function=rollout,
#         budget=100,
#         low=-1.0,
#         high=1.0)
# algorithm_factories.append(makeTwoPointsDE)
#
# # Win at 1 epoch.
# def makeCMA(environment):
#     logger.info("    ---- CMA ----    ");
#     return CMA(
#         environment=environment,
#         policy_factory=DeterministicLinearPolicyFactory(),
#         rollout_function=rollout,
#         budget=100,
#         low=-1.0,
#         high=1.0)
# algorithm_factories.append(makeCMA)

# # Win at 2762 epochs. -> The problem is the Stochastic policy...
# # ...which means that is the problem for the PG too.
# def makeUniformRandom(environment):
#     logger.info("    ---- UniformRandom ----    ");
#     return UniformRandom(
#         environment=environment,
#         policy_factory=StochasticLinearPolicyFactory(),
#         rollout_function=rollout,
#         batch_size=1,
#         low=-1.0,
#         high=1.0)
# algorithm_factories.append(makeUniformRandom)

# # Wins at 1324 epochs
# def makeDiscreteLinearPolicyGradient(environment):
#     logger.info("    ---- DiscreteLinearPolicyGradient ----    ");
#     return DiscreteLinearPolicyGradient(
#         environment=environment,
#         rollout_function=rollout,
#         min_steps_per_batch=1)
# algorithm_factories.append(makeDiscreteLinearPolicyGradient)


#
# # Loses forever.
# def makeDiscreteLinearTensorflowPolicyGradient(environment):
#     logger.info("    ---- DiscreteLinearTensorflowPolicyGradient ----    ");
#     return DiscreteLinearTensorflowPolicyGradient(
#         environment=environment,
#         rollout_function=rollout,
#         min_steps_per_batch=1)
# algorithm_factories.append(makeDiscreteLinearTensorflowPolicyGradient)


experiment = Experiment(
    max_epochs=max_epochs,
    random_seed=random_seed,
    environment_names=[environment_name],
    algorithm_functions=algorithm_factories)

experiment.run()

# #################################
# # CartPole-v1 : Box -> Discrete #
# #################################
# environment_name = 'CartPole-v1'
# max_epochs = 1000
# environment = gym.make(environment_name)
# specification = gym.spec(environment_name)
# environment.seed(random_seed)
# # Win at 7 epochs.
# algorithm = UniformRandom(
#     environment=environment,
#     policy_factory=DeterministicLinearPolicyFactory(),
#     rollout_function=rollout,
#     batch_size=1,
#     low=-1.0,
#     high=1.0)
# # Winning at 5 epochs.
# algorithm = RandomSearch(
#     environment=environment,
#     policy_factory=DeterministicLinearPolicyFactory(),
#     rollout_function=rollout,
#     batch_size=1,
#     explore=10.0)
# # Win at 1 epoch.
# algorithm = OnePlusOne(
#     environment=environment,
#     policy_factory=DeterministicLinearPolicyFactory(),
#     rollout_function=rollout,
#     budget=200,
#     low=-1.0,
#     high=1.0)



# ####################################
# # MountainCar-v0 : Box -> Discrete #
# ####################################
# environment_name = 'MountainCar-v0'
# max_epochs = 1000
# environment = gym.make(environment_name)
# specification = gym.spec(environment_name)
# environment.seed(random_seed)
# # Loses at 1000 epochs.
# algorithm = UniformRandom(
#     environment=environment,
#     policy_factory=DeterministicLinearPolicyFactory(),
#     rollout_function=rollout,
#     batch_size=1,
#     low=-100.0,
#     high=100.0)
# # Wins at 662 epochs.
# algorithm = RandomSearch(
#     environment=environment,
#     policy_factory=DeterministicLinearPolicyFactory(),
#     rollout_function=rollout,
#     batch_size=1,
#     explore=600.0)
# # Loses at 1000 epochs.
# algorithm = OnePlusOne(
#     environment=environment,
#     policy_factory=DeterministicLinearPolicyFactory(),
#     rollout_function=rollout,
#     budget=200,
#     low=-1.0,
#     high=1.0)


# # ################################
# # # Acrobot-v1 : Box -> Discrete #
# # ################################
# environment_name = 'Acrobot-v1'
# max_epochs = 1000
# environment = gym.make(environment_name)
# specification = gym.spec(environment_name)
# environment.seed(random_seed)
# # Wins at 19 epochs.
# algorithm = UniformRandom(
#     environment=environment,
#     policy_factory=DeterministicLinearPolicyFactory(),
#     rollout_function=rollout,
#     batch_size=1,
#     low=-1.0,
#     high=1.0)
# # Wins at 105 epochs.
# algorithm = RandomSearch(
#     environment=environment,
#     policy_factory=DeterministicLinearPolicyFactory(),
#     rollout_function=rollout,
#     batch_size=1,
#     explore=600.0)
# # Wins at 1 epoch.
# algorithm = OnePlusOne(
#     environment=environment,
#     policy_factory=DeterministicLinearPolicyFactory(),
#     rollout_function=rollout,
#     budget=200,
#     low=-1.0,
#     high=1.0)


# ############################
# # Pendulum-v0 : Box -> Box #
# ############################
# environment_name = 'Pendulum-v0'
# max_epochs = 100
# environment = gym.make(environment_name)
# specification = gym.spec(environment_name)
# environment.seed(random_seed)
# # Episode reward at 100 epochs is -1657.
# algorithm = UniformRandom(
#     environment=environment,
#     policy_factory=DeterministicLinearPolicyFactory(), #PolynomialPolicyFactory(),
#     rollout_function=rollout,
#     batch_size=1,
#     low=environment.action_space.low[0],
#     high=environment.action_space.high[0])
# # Episode reward at 100 epochs is -1492.
# algorithm = RandomSearch(
#     environment=environment,
#     policy_factory=DeterministicLinearPolicyFactory(),
#     rollout_function=rollout,
#     batch_size=1,
#     explore=10.0)
# Episode reward at 100 epochs is -1474.
# algorithm = OnePlusOne(
#     environment=environment,
#     policy_factory=DeterministicLinearPolicyFactory(),
#     rollout_function=rollout,
#     budget=200,
#     low=environment.action_space.low[0],
#     high=environment.action_space.high[0])


# #########################################
# # MountainCarContinuous-v0 : Box -> Box #
# #########################################
# environment_name = 'MountainCarContinuous-v0'
# max_epochs = 1000
# environment = gym.make(environment_name)
# specification = gym.spec(environment_name)
# environment.seed(random_seed)
# # Wins at 164 epochs.
# algorithm = UniformRandom(
#     environment=environment,
#     policy_factory=DeterministicLinearPolicyFactory(),
#     rollout_function=rollout,
#     batch_size=1,
#     low=100*environment.action_space.low[0],
#     high=100*environment.action_space.high[0])
# # Wins at 493 epochs.
# algorithm = RandomSearch(
#     environment=environment,
#     policy_factory=DeterministicLinearPolicyFactory(),
#     rollout_function=rollout,
#     batch_size=1,
#     explore=100.0)
# # Wins at 46 epochs.
# algorithm = OnePlusOne(
#     environment=environment,
#     policy_factory=DeterministicLinearPolicyFactory(),
#     rollout_function=rollout,
#     budget=200,
#     low=100*environment.action_space.low[0],
#     high=100*environment.action_space.high[0])


# ####################################
# # LunarLander-v2 : Box -> Discrete #
# ####################################
# environment_name = 'LunarLander-v2'
# max_epochs = 100
# environment = gym.make(environment_name)
# specification = gym.spec(environment_name)
# environment.seed(random_seed)
# # Loses at 1000 epochs.
# algorithm = UniformRandom(
#     environment=environment,
#     policy_factory=DeterministicLinearPolicyFactory(),
#     rollout_function=rollout,
#     batch_size=1,
#     low=-1,
#     high=1)
# # Loses at 100 epochs.
# algorithm = RandomSearch(
#     environment=environment,
#     policy_factory=DeterministicLinearPolicyFactory(),
#     rollout_function=rollout,
#     batch_size=1,
#     explore=100.0)
# # Loses at 1000 epochs.
# algorithm = OnePlusOne(
#     environment=environment,
#     policy_factory=DeterministicLinearPolicyFactory(),
#     rollout_function=rollout,
#     budget=200,
#     low=-1,
#     high=1)


##################
# Run Experiment #
##################
