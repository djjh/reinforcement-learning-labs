import gym
import logging
import numpy as np
from rl.common import DeterminisiticSampler, StochasticSampler
from rl.types import Episode, Episodes
from rl.logging import get_expermiment_logging_directory
from rl.policies import RandomPolicy, VanillaPolicy, LSTMVanillaPolicy, NewPolicy
from rl.weights import ReturnWeights, RewardToGoWeights, ExemplarDensityWeights


######################
# Initialize Logging #
######################

# logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger('rl.policies.lstm_vanilla_policy').setLevel(logging.INFO)
logging.getLogger('rl.weights.reward_to_go_weights').setLevel(logging.INFO)

log_directory = get_expermiment_logging_directory(__file__)

logger.info("Logging to directory: {}".format(log_directory))


###############################
# Initialize Experiment Setup #
###############################

environment = gym.make('CartPole-v0')
# environment = gym.make('MountainCar-v0')

# random_policy = RandomPolicy(environment)



vanilla_policy = VanillaPolicy(
    environment=environment,
    weights=RewardToGoWeights(discount=0.0),
    action_sampler=StochasticSampler(),
    log_directory=log_directory,
    random_seed=0,
    learning_rate=1e-2)

# weights = ExemplarDensityWeights(
#     environment=environment,
#     use_actions=False,
#     log_directory=log_directory,
#     random_seed=0,
#     learning_rate=1e-3)
#
# lstm_vanilla_policy = LSTMVanillaPolicy(
#     environment=environment,
#     weights=weights, # RewardToGoWeights(discount=0.0),
#     action_sampler=StochasticSampler(),
#     log_directory=log_directory,
#     random_seed=0,
#     learning_rate=1e-2)

def rollout(epoch, first, environment, policy):
    episode = Episode()
    observation = environment.reset()
    done = False
    while not done:
        if epoch % 10 == 0 and first:
            environment.render()
        action = policy.get_action(observation)
        previous_observation = observation
        observation, reward, done, info = environment.step(action)
        episode.append(previous_observation, action, reward)
    return episode

def rollouts(epoch, environment, policy, min_steps):
    episodes = Episodes()
    first = True
    while episodes.get_cumulative_length() < min_steps:
        episodes.append(rollout(epoch, first, environment, policy))
        first = False
    return episodes

#
# new_policy = NewPolicy(
#     environment=environment,
#     rollout_function=rollout_function,
#     rollouts_function=rollouts_function)


##################
# Run Experiment #
##################

with vanilla_policy as policy:

    print(vanilla_policy.get_parameters())

    # observation = environment.reset()
    # logger.info(policy.session.run(
    #     fetches=policy.lstm,
    #     feed_dict={
    #         policy.observations_placeholder: np.array(observation).reshape(1, 1, -1)
    #     }));

    # for epoch in range(50):
    #     policy.
    #     policy.update(epoch, rollouts(epoch, environment, policy, 5000))
    #     logger.info('epoch: {}'.format(epoch))
