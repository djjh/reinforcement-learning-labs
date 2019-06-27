import sys, os
from pathlib import Path
sys.path.append(str(Path(os.path.join(os.path.dirname(__file__), '..')).resolve()))

from rl.core import AlgorithmFactory
from rl.core import EnvironmentFactory
from rl.core import Runner


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
        # 'CartPole-v1'
    ]

    problems = []

    for algorithm_name in algorithm_names:
        for environment_name in environment_names:
            print('---- {} - {} ----\n'.format(algorithm_name, environment_name))
            try:
                runner.run(
                    algorithm_name=algorithm_name,
                    environment_name=environment_name,
                    random_seed=0,
                    max_epochs=1000,
                    deterministic=True)
            except Exception as e:
                problems.append({'algorithm': algorithm_name, 'exception': e})

    if len(problems) > 0:
        print('Failed:')
    for problem in problems:
        print('START------------------------------------')
        print(problem['algorithm'])
        print(problem['exception'])
        print('END------------------------------------')
