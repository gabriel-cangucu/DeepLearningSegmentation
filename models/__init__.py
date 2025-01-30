import torch
import sys
from typing import Any

import utils.distributed_utils as dist
from models.unet import Unet
from models.vit import ViT
from models.tsvit import TSViT
from models.flexi_tsvit import FlexiTSViT
from models.tsvit_sin import TSViT_Sin


def get_model(config: dict[str, Any]) -> torch.nn.Module:
    model_map = {
        'unet': Unet,
        'vit': ViT,
        'tsvit': TSViT,
        'flexi_tsvit': FlexiTSViT,
        'tsvit_sin': TSViT_Sin
    }

    model_config = config['MODEL']

    assert type(model_config['architecture']) == str

    try:
        model_class = model_map[model_config['architecture'].lower()]
    except KeyError:
        sys.exit(f'{model_config["architecture"]} is not a valid architecture.')
    
    device = dist.get_current_device()

    model = model_class(model_config).to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    if dist.is_distributed():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device]
        )

    return model