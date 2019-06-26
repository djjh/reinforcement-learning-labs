
from rl.core import Episode

# Yes this is wierd, it's a temporary hack.
def Rollout(environment, policy, random_seed, deterministic, render):
    episode = Episode()
    if deterministic:
    #     # TODO: should replace this with supplying the seed instead, and
    #     # should reseed all random seeds here.
    #     np.random.seed(random_seed)
    #     random.seed(random_seed)
        environment.seed(random_seed)
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
