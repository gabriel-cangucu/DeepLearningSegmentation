import sys
import torch
from typing import Any

from data.PASTIS24.dataloader import get_dataloader as get_pastis_dataloader
from data.PASTIS24.data_transforms import get_transforms as get_pastis_transforms
from data.Synthetic.dataloader import get_dataloader as get_synthetic_dataloader
from data.Synthetic.data_transforms import get_transforms as get_synthetic_transforms


def get_dataloaders(config: dict[str, Any]) -> tuple[dict[str, torch.utils.data.DataLoader],
                                                     torch.utils.data.Sampler]:
    dataset_map = {
        'PASTIS24': (get_pastis_dataloader, get_pastis_transforms),
        'Synthetic': (get_synthetic_dataloader, get_synthetic_transforms)
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

    dataloader['train'], sampler = loader_fn(
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

        dataloader['val'], _ = loader_fn(
            root_dir=val_config['root_dir'],
            csv_path=val_config['csv_path'],
            batch_size=int(val_config['batch_size']),
            num_workers=int(val_config['num_workers']),
            transform=transform_fn(model_config, is_training=False),
            shuffle=False
        )
    
    # Test data
    if 'test' in config['DATASETS'].keys():
        test_config = config['DATASETS']['test']
        assert type(test_config['dataset_name']) == str

        try:
            loader_fn, transform_fn = dataset_map[test_config['dataset_name']]
        except KeyError:
            sys.exit(f'{test_config["dataset_name"]} is not a valid dataset.')
        
        assert type(test_config['root_dir']) == type(test_config['csv_path']) == str

        dataloader['test'], _ = loader_fn(
            root_dir=test_config['root_dir'],
            csv_path=test_config['csv_path'],
            batch_size=int(test_config['batch_size']),
            num_workers=int(test_config['num_workers']),
            transform=transform_fn(model_config, is_training=False),
            shuffle=False
        )

    return dataloader, sampler