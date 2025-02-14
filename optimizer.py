from pydgn.training.callback.optimizer import Optimizer


class Optimizer(Optimizer):

    def on_fit_start(self, state):
        pass  # do not load the state of the optimizer, it causes a strange
        # bug and it is necessary only if we suddenly interrupt the evaluation
        # (also, not that relevant)