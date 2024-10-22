import torch
from typing import Any


def get_scheduler(optimizer: torch.optim, num_warmup_epochs: int, lr_min: float
                 ) -> torch.optim.lr_scheduler:
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=num_warmup_epochs,
        eta_min=lr_min
    )