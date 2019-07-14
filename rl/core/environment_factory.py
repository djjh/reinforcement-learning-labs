import gym

class EnvironmentFactory:

    def __init__(self):
        pass

    def create_environment(self, environment_name):
        specification = gym.spec(environment_name)
        environment = gym.make(environment_name)
        if specification and environment:
            return specification, environment
        else:
            raise NotImplementedError()
