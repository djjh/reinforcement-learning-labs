from rl.core import Policy

class Algorithm:

    def __init__(self):
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def policy(self, deterministic: bool) -> Policy:
        raise NotImplementedError()

    def update(self):
        raise NotImplementedError()
