import gym
import io
import numpy as np
import multiprocessing as mp
import os
import tensorflow as tf
import random

from collections import deque
from gym import wrappers
from os.path import splitext, basename
from matplotlib import pyplot as pp
from time import strftime, gmtime


from core import Episode
from policies import MyPolicy


#################
# Configuration #
#################

experiment_name = splitext(basename(__file__))[0]
timestamp = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
log_directory = "log/{}/{}".format(experiment_name, timestamp)
render = True
render_epoch = 49

# environment_name = 'CartPole-v0'
environment_name = 'MountainCar-v0'
random_seed = 0

policy_learning_rate = 1e-3
num_epochs = 50
min_steps_per_epoch = 5000
class PolicyConfig:
    determinisitic = True

density_learning_rate = 1e-3
density_iterations = 100
bonus_coefficient = 1e-0
bonus_function = negative_log
use_actions = False
bonus_only = True


###########################
# Initialize Random Seeds #
###########################

random.seed(random_seed)
np.random.seed(random_seed)
tf.set_random_seed(random_seed)


##########################
# Initialize Environment #
##########################

environment = gym.make(environment_name)


#####################
# Initialize Policy #
#####################

policy = MyPolicy(
    environment=environment,
    log_directory=log_directory,
    random_seed=random_seed,
    learning_rate=policy_learning_rate)

with policy:

    ################
    # Train Policy #
    ################


    for epoch in range(num_epochs):

        ########################
        # Gather Training Data #
        ########################

        episodes = []
        episode = Episode()
        observation = environment.reset()

        while True:
            action = policy.get_action(observation=observation)
            previous_observation = observation
            observation, reward, done, info = environment.step(action)
            episode.append(previous_observation, action, reward)
            if done:
                episodes.append(episode)
                if len(episodes) >= min_steps_per_epoch:
                    break
                episode = Episode()
                observation = environment.reset()

        #################
        # Update Policy #
        #################

        policy.update(epoch=epoch, episodes=episodes)

        print('epoch:', epoch)
