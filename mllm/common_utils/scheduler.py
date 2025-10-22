"""
Learning rate schedulers for training.
"""
import numpy as np
import torch

class InverseLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Inverse decay learning rate schedule with exponential warmup.

    When last_epoch=-1, sets initial lr as lr.
    inv_gamma is the number of steps/epochs required for the learning rate to decay to
    (1 / 2)**power of its original value.

    Args:
        optimizer: Wrapped optimizer
        inv_gamma: Inverse multiplicative factor of learning rate decay
        power: Exponential factor of learning rate decay
        warmup: Exponential warmup factor (0 <= warmup < 1, 0 to disable)
        final_lr: The final learning rate
        last_epoch: The index of last epoch
        verbose: If True, prints a message to stdout for each update
    """

    def __init__(self, optimizer: torch.optim.Optimizer, inv_gamma: float = 1., power: float = 1.,
                 warmup: float = 0., final_lr: float = 0., last_epoch: int = -1, verbose: bool = False):
        self.inv_gamma = inv_gamma
        self.power = power
        if not 0. <= warmup < 1:
            raise ValueError('Invalid value for warmup')
        self.warmup = warmup
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self) -> list[float]:
        """Compute learning rate in closed form."""
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        lr_mult = (1 + self.last_epoch / self.inv_gamma) ** -self.power
        return [warmup * max(self.final_lr, base_lr * lr_mult)
                for base_lr in self.base_lrs]


class LinearWarmupCosineDecay(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup and cosine decay learning rate schedule.

    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr: Minimum learning rate
        last_epoch: The index of last epoch
        verbose: If True, prints a message to stdout for each update
    """
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int = 100, total_steps: int = 10000,
                 min_lr: float = 1e-8, last_epoch: int = -1, verbose: bool = False):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Get current learning rates with warmup and cosine decay."""
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.")

        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup
            scale = self.min_lr + (1 - self.min_lr) * (step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            scale = 0.5 * (1.0 + torch.cos(torch.tensor(np.pi * progress)))
            scale = self.min_lr + (1 - self.min_lr) * scale
        return [scale * base_lr for base_lr in self.base_lrs]
