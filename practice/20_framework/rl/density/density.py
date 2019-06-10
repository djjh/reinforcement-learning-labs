class Density():

    def __enter__(self):
        raise NotImplementedError()

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError()

    def estimate(self, epoch, x):
        raise NotImplementedError()

    def update(self, epoch, episodes):
        raise NotImplementedError()

    def log(self, epoch, x, w, rb, grid):
        # probabilities = self.estimate(epoch, grid)
        summary = self.session.run(
            fetches=self.log_summary,
            feed_dict={
                self.actual_distribution_png_placeholder: to_png(histogram(x)),
                # self.predicted_distribution_png_placeholder: to_png(probability_contours(grid, probabilities)),
                self.weights_png_placeholder: to_png(weighted_histogram(x, w)),
                self.bonuses_png_placeholder: to_png(weighted_histogram(x, rb))})
        self.update_writer.add_summary(summary, epoch)
