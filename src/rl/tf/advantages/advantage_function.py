

class AdvantageFunction:

    def __init__(self):
        raise NotImplementedError()

    def get_advantages(self, experience):
        raise NotImplementedError()

    def update(self, experience):
        raise NotImplementedError()
