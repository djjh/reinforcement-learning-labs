class Weights:

    def __enter__(self):
        raise NotImplementedError()

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError()

    def get_batch_weights(self, episodes):
        raise NotImplementedError()

    def update(self, epoch, episodes):
        raise NotImplementedError()
