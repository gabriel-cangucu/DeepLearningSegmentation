import torch
from typing import Any


def get_scheduler(config: dict[str, Any], optimizer: torch.optim,
                  num_train_steps: int) -> torch.optim.lr_scheduler:
    solver_config = config['SOLVER']

    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(solver_config['num_epochs'])*num_train_steps,
        eta_min=float(solver_config['lr_min'])
    )