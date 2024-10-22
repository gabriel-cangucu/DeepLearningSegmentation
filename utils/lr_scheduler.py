import torch
import sys
from typing import Any
from torch.optim.lr_scheduler import (
    ExponentialLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
)


def get_scheduler(config: dict[str, Any], optimizer: torch.optim) -> torch.optim.lr_scheduler:
    scheduler_map = {
        'exponential': (ExponentialLR, {'gamma': 0.1}),
        'plateau': (ReduceLROnPlateau, {}),
        'cosine': (CosineAnnealingWarmRestarts, {'T_0': 10, 'eta_min': 5e-6})
    }

    solver_config = config['SOLVER']

    assert type(solver_config['lr_scheduler']) == str

    try:
        scheduler_fn, args = scheduler_map[solver_config['lr_scheduler']]
    except KeyError:
        sys.exit(f'{solver_config["lr_scheduler"]} is not a valid learning rate scheduler.')

    return scheduler_fn(optimizer, **args)