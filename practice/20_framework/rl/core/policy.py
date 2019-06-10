
class Policy:

    def __call__(self, observation):
        raise NotImplementedError()

    def get_parameters(self):
        raise NotImplementedError()

    def set_parameters(self, parameters):
        raise NotImplementedError()
