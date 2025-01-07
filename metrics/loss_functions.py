import torch
import sys
import segmentation_models_pytorch as smp
from typing import Any


def get_loss(config: dict[str, Any], device: torch.device, reduction: str='mean') -> torch.nn:
    model_config = config['MODEL']
    loss_config = config['SOLVER']
    
    class_weights = loss_config['class_weights']

    if class_weights is not None:
        assert type(class_weights) in [list, tuple]
        assert len(class_weights) == model_config['num_classes']
        assert all(type(w) in [int, float] for w in class_weights)

        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    ignore_index = loss_config['ignore_index']

    if ignore_index is not None:
        ignore_index = int(ignore_index)
    else:
        ignore_index = -100
    
    mode = 'binary' if model_config['num_classes'] == 1 else 'multiclass'
    
    loss_map = {
        'cross_entropy': (torch.nn.CrossEntropyLoss, {'weight': class_weights, 'ignore_index': ignore_index}),
        'binary_cross_entropy': (torch.nn.BCEWithLogitsLoss, {'pos_weight': class_weights}),
        'dice': (smp.losses.DiceLoss, {'mode': mode, 'ignore_index': ignore_index})
    }

    assert type(loss_config['loss_function']) == str

    try:
        loss_fn, args = loss_map[loss_config['loss_function']]
    except KeyError:
        sys.exit(f'{loss_config["loss_function"]} is not a valid loss function.')

    return loss_fn(**args).to(device)
