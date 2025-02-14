import math
import torch.distributed.optim.optimizer
from pydgn.experiment.util import s2c
from pydgn.training.callback.optimizer import Optimizer
from pydgn.training.callback.scheduler import Scheduler, EpochScheduler
from pydgn.training.engine import TrainingEngine
from pydgn.training.event.state import State
from torch.optim.lr_scheduler import MultiplicativeLR, StepLR, LambdaLR


class ExplicitLR(StepLR):
    """"""

    def __init__(self, optimizer, lrs, last_epoch=-1, verbose=False):
        self.lrs = lrs
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch >= len(self.lrs):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [self.lrs[self.last_epoch] for _ in self.optimizer.param_groups]


class CifarScheduler(Scheduler):
    def __init__(
        self, scheduler_class_name: str, optimizer: Optimizer, **kwargs: dict
    ):
        lrs = [0.01] * 5 + [0.1] * 195 + [0.01] * 100 + [0.001] * 100
        self.scheduler = ExplicitLR(optimizer, lrs)

    def on_epoch_start(self, state: State):
        super(CifarScheduler, self).on_epoch_start(state)
        self.scheduler.last_epoch = state.epoch


class CosineAnnealingLinearWarmup(EpochScheduler):

    def __init__(
        self, scheduler_class_name: str, optimizer: Optimizer, **kwargs: dict
    ):

        assert scheduler_class_name == 'torch.optim.lr_scheduler.LambdaLR'

        num_warmup_steps = kwargs['num_warmup_steps']
        num_training_steps = kwargs['num_training_steps']
        num_cycles = kwargs['num_cycles']
        last_epoch = kwargs.get('last_epoch', -1)

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return max(1e-6, float(current_step) / float(
                    max(1, num_warmup_steps)))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(
                math.pi * float(num_cycles) * 2.0 * progress)))

        self.scheduler = LambdaLR(optimizer, lr_lambda, last_epoch)
