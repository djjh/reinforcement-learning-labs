import rl


class AlgorithmFactory:

    def __init__(self):
        self._algorithms = {}
        self._algorithms.update(rl.tf.factories.AlgorithmFactory().get_algorithms())
        self._algorithms.update(rl.np.factories.AlgorithmFactory().get_algorithms())

    def create_algorithm(self, algorithm_name, environment, random_seed):
        return self._algorithms[algorithm_name](environment, random_seed)
