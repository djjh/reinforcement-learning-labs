from rl.core import Policy

class PolicyFactory:

    def create_policy(self) -> Policy:
        raise NotImplementedError()
