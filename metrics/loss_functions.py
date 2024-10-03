import torch
import sys
from typing import Any


def get_loss(config: dict[str, Any], device: torch.device, reduction: str='mean') -> torch.nn:
    loss_map = {
        'cross_entropy': torch.nn.CrossEntropyLoss
    }

    model_config = config['MODEL']
    loss_config = config['SOLVER']

    assert type(loss_config['loss_function']) == str

    try:
        loss_fn = loss_map[loss_config['loss_function']]
    except KeyError:
        sys.exit(f'{loss_config["loss_function"]} is not a valid loss function.')
    
    class_weights = loss_config['class_weights']

    if class_weights is not None:
        assert type(class_weights) in [list, tuple]
        assert len(class_weights) == model_config['num_classes']

        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    ignore_index = loss_config['ignore_index']

    if ignore_index is None:
        ignore_index = -100

    return loss_fn(weight=class_weights, reduction=reduction, ignore_index=ignore_index).to(device)
