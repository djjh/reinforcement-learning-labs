

class AdvantageFunction:

    def __init__(self):
        raise NotImplementedError()

    def get_advantages(self, episodes):
        raise NotImplementedError()

    def update(self, episodes):
        raise NotImplementedError()
