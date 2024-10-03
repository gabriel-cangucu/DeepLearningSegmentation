import torch
import sys
from typing import Any

from utils.device_utils import get_current_device
from models.unet import Unet
from models.vit import ViT
from models.tsvit import TSViT


def get_model(config: dict[str, Any]) -> torch.nn.Module:
    model_map = {
        'unet': Unet,
        'vit': ViT,
        'tsvit': TSViT
    }

    model_config = config['MODEL']

    assert type(model_config['architecture']) == str

    try:
        model_class = model_map[model_config['architecture'].lower()]
    except KeyError:
        sys.exit(f'{model_config["architecture"]} is not a valid architecture.')
    
    device = get_current_device()

    model = model_class(model_config)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model.to(device), device_ids=[device]
    )

    return model