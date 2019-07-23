

class ValueFunction:

    def __init__(self):
        raise NotImplementedError()

    def get_values(self, episodes):
        raise NotImplementedError()

    def update(self, episodes):
        raise NotImplementedError()
