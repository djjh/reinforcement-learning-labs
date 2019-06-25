import sys, os
from pathlib import Path
sys.path.append(str(Path(os.path.join(os.path.dirname(__file__), '..')).resolve()))

from rl import AlgorithmFactory
from rl import EnvironmentFactory
from rl import Runner


if __name__ == '__main__':

    algorithm_factory = AlgorithmFactory()
    environment_factory = EnvironmentFactory()

    runner = Runner(algorithm_factory, environment_factory)

    algorithm_names = [
        'Tensorflow-VanillaPolicyGradient-v0',
        'Scratch-UniformRandom-v0',
        'Scratch-RandomSearch-v0',
        'Scratch-OnePlusOne-v0',
        'Scratch-VanillaPolicyGradient-v0'
    ]
    environment_names = [
        'CartPole-v0',
        'CartPole-v1'
    ]

    for algorithm_name in algorithm_names:
        for environment_name in environment_names:
            print('---- {} - {} ----\n'.format(algorithm_name, environment_name))
            runner.run(
                algorithm_name=algorithm_name,
                environment_name=environment_name,
                random_seed=0,
                max_epochs=1000,
                deterministic=True)
