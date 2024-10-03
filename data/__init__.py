import sys
import torch
from typing import Any

from data.PASTIS24.dataloader import get_dataloader as get_pastis_dataloader
from data.PASTIS24.data_transforms import get_transforms as get_pastis_transforms


def get_dataloaders(config: dict[str, Any]) -> dict[str, torch.utils.data.DataLoader]:
    dataset_map = {
        'PASTIS24': (get_pastis_dataloader, get_pastis_transforms)
    }

    dataloader = {}
    model_config = config['MODEL']

    # Train data
    train_config = config['DATASETS']['train']
    assert type(train_config['dataset_name']) == str

    try:
        loader_fn, transform_fn = dataset_map[train_config['dataset_name']]
    except KeyError:
        sys.exit(f'{train_config["dataset_name"]} is not a valid dataset.')
    
    assert type(train_config['root_dir']) == type(train_config['csv_path']) == str

    dataloader['train'] = loader_fn(
        root_dir=train_config['root_dir'],
        csv_path=train_config['csv_path'],
        batch_size=int(train_config['batch_size']),
        num_workers=int(train_config['num_workers']),
        transform=transform_fn(model_config, is_training=True),
        shuffle=True
    )

    # Val data
    if 'val' in config['DATASETS'].keys():
        val_config = config['DATASETS']['val']
        assert type(val_config['dataset_name']) == str

        try:
            loader_fn, transform_fn = dataset_map[val_config['dataset_name']]
        except KeyError:
            sys.exit(f'{val_config["dataset_name"]} is not a valid dataset.')
        
        assert type(val_config['root_dir']) == type(val_config['csv_path']) == str

        dataloader['val'] = loader_fn(
            root_dir=val_config['root_dir'],
            csv_path=val_config['csv_path'],
            batch_size=int(val_config['batch_size']),
            num_workers=int(val_config['num_workers']),
            transform=transform_fn(model_config, is_training=False),
            shuffle=False
        )

    return dataloader