import gym
import numpy as np
import tensorflow as tf

environment_name = 'CartPole-v0'
num_epochs = 1
max_steps_per_epoch = 100

# Initialize the environment.
environment = gym.make(environment_name)

for epoch in range(num_epochs):

    # Retrieve the initial observation corresponding to the initial state.
    observation = environment.reset()

    for step in range(max_steps_per_epoch):

        # Optionally Rrender the environment for debugging via visual
        # inspection.
        environment.render()

        # Generate the next actions given the current observation.
        # This will generally take the form of
        #
        #     action = policy(observation)
        #
        action = environment.action_space.sample()

        # Step the environment for the
        observation, reward, done, info = environment.step(actions)

        if done:
            break


