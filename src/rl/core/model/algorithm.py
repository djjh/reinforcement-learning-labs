

class Algorithm:
    def __init__():
        raise NotImplementedError()

    def __enter__(self):
        raise NotImplementedError()

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError()

    def action(self, observation, deterministic):
        """ Returns an action """
        raise NotImplementedError()

    def update(self):
        """ Train for an epoch """
        raise NotImplementedError()
