from rl.core import Policy, PolicyFactory


class DisceteLinearPolicy(Policy):

    def __init__(self, environment, action_sampler):

        assert isinstance(environment.action_space, Discrete), "Must be Discrete action space."

        self.environment = environment
        self.probability_mass_function_sampler = probability_mass_function_sampler
        self.observation_dimensions = np.prod(self.environment.observation_space.shape)
        self.action_dimensions = self.environment.action_space.n
        self.model = np.random.randn(self.action_dimensions, self.observation_dimensions)

    def get_parameters(self):
        return [self.model]

    def set_parameters(self, parameters):
        assert len(parameters) == 1, "Invalid number of parameters."
        assert parameters[0].shape == self.model.shape, "Invalid parameter shape."
        self.model = parameters[0]

    def __call__(self, observation):
        probabilities = self.model.dot(observation.reshape(-1, 1))
        return action_sampler(probabilities)

class BoxLinearPolicy(Policy):

    def __init__(self, environment, action_sampler):
        assert isinstance(environment.action_space, Box), "Must be Box action space."

        self.environment = environment
        self.observation_dimensions = np.prod(self.environment.observation_space.shape)
        self.action_dimensions = np.prod(self.environment.action_space.shape)
        self.model = np.random.randn(self.action_dimensions, self.observation_dimensions)

    def get_parameters(self):
        return [self.model]

    def set_parameters(self, parameters):
        assert len(parameters) == 1, "Invalid number of parameters."
        assert parameters[0].shape == self.model.shape, "Invalid parameter shape."
        self.model = parameters[0]

    def set_deterministic(self, deterministic):
        self.deterministic = deterministic

    def __call__(self, observation):
        mean_actions = self.model.dot(observation.reshape(-1, 1)).reshape(self.environment.action_space.shape)
        return mean_actions

class LinearPolicyFactory(PolicyFactory):
    def create_policy(self, environment, action_sampler) -> Policy:
        if isinstance(environment.action_space, Discrete):
            return DisceteLinearPolicy(environment, action_sampler)
        elif isinstance(environment.action_space, Box):
            return BoxLinearPolicy(environment, action_sampler)
        else:
            raise NotImplementedError()
