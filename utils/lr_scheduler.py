import torch
import sys
from typing import Any
from torch.optim.lr_scheduler import (
    ExponentialLR, OneCycleLR, CosineAnnealingWarmRestarts
)


def get_scheduler(config: dict[str, Any], optimizer: torch.optim) -> torch.optim.lr_scheduler:
    solver_config = config['SOLVER']

    assert type(solver_config['lr_scheduler']) == str
    lr_base = float(solver_config['lr_base'])
    lr_min = float(solver_config['lr_min'])
    
    scheduler_map = {
        'exponential': (ExponentialLR, {'gamma': 0.1}),
        'cyclic': (OneCycleLR, {'max_lr': lr_base, 'epochs': 10, 'steps_per_epoch': 10}),
        'cosine': (CosineAnnealingWarmRestarts, {'T_0': 10, 'eta_min': lr_min})
    }

    try:
        scheduler_fn, args = scheduler_map[solver_config['lr_scheduler']]
    except KeyError:
        sys.exit(f'{solver_config["lr_scheduler"]} is not a valid learning rate scheduler.')

    return scheduler_fn(optimizer, **args)