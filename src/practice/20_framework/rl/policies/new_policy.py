# import tensorflow_probability as tfp

from rl.core import Policy


def objective_function():
    return 1.0

# Maybe this should be called a 'PolicyOptimizer' instead?
class NewPolicy(Policy):

    def __init__(self, environment, rollout_function, rollouts_function):
        self.environment = environment
        self.rollout_function = rollout_function
        self.rollouts_function = rollouts_function

        # policy = VanillaPolicy()
        # parameters = policy.get_parameters()
        # policy.set_parameters(parameters)
        #
        #
        # assignment.eval()
        # # self.population =

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_action(self, observation):
        return self.best_strategy

    def update(self, epoch, episodes):
        pass

        # results = tfp.optimizers.differential_evolution_minimize(
        #     objective_function,
        #     initial_population=self.population)
        #
        # self.population = results.final_population
