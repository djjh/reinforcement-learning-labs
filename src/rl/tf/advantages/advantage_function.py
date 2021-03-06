

class AdvantageFunction:

    def __init__(self):
        raise NotImplementedError()

    def __enter__(self):
        raise NotImplementedError()

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError()

    def get_advantages(self, episodes):
        raise NotImplementedError()

    def update(self, episodes):
        raise NotImplementedError()
