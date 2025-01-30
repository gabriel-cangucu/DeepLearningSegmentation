import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from typing import Any

import utils.distributed_utils as dist
from data import get_dataloaders
from models import get_model
from metrics.loss_functions import get_loss
from metrics.numpy_metrics import RunningMetrics
from utils.config_files_utils import read_yaml
from utils.summaries import write_summaries
from utils.torch_utils import *


def test_and_predict(net: torch.nn.Module, dataloaders: torch.utils.data.DataLoader,
                     sampler: torch.utils.data.Sampler, config: dict[str, Any]) -> None:
    
    def test(net: torch.nn.Module, test_loader: torch.utils.data.DataLoader, config: dict[str, Any],
             loss_fn: torch.nn, save_path: str, device: torch.device) -> dict[str, float]:
        net.eval()

        num_classes = int(config['MODEL']['num_classes'])
        ignore_index = config['SOLVER']['ignore_index']
        running_test_metrics = RunningMetrics(num_classes, ignore_index)

        all_losses_tensor = torch.zeros(len(test_loader)).to(device)

        save_preds = config['CHECKPOINT']['save_predictions']
        assert save_preds is not None
        assert type(save_preds) == bool

        for idx, sample in enumerate(test_loader):
            print(f'Doing batch {idx+1}')

            inputs = sample['inputs'].to(device)
            targets = sample['labels'].to(device)

            outputs = net(inputs)
            loss_tensor = loss_fn(outputs, targets)

            all_losses_tensor[idx] = loss_tensor.item()

            is_multiclass = True if config['MODEL']['num_classes'] > 1 else False
            preds = logits_to_preds(outputs, is_multiclass=is_multiclass)

            # Saving net predictions
            if save_preds:
                store_batch_predictions(preds, save_path, indexes=sample['indexes'])
            
            running_test_metrics.update(preds, targets)
        
        all_losses_tensor = all_losses_tensor.mean()
        dist.all_reduce_mean(all_losses_tensor)
        loss = all_losses_tensor.item()
        
        test_metrics = running_test_metrics.get_scores()
        test_metrics['loss'] = loss
        running_test_metrics.reset()

        return test_metrics

    
    checkpoint_path = config['CHECKPOINT']['load_from_checkpoint']

    try:
        load_from_checkpoint(net, checkpoint_path)
    except FileNotFoundError:
        sys.exit('Invalid path to stored model in .pt format.') 

    save_path = config['CHECKPOINT']['save_path']
    assert type(save_path) == str
    os.makedirs(save_path, exist_ok=True)

    device = dist.get_current_device()
    loss_fn = get_loss(config, device, reduction='mean')

    writer = SummaryWriter(save_path)

    test_metrics = test(net, test_loader=dataloaders['test'], config=config,
                        loss_fn=loss_fn, save_path=save_path, device=device)
    
    # Printing metrics
    for metric, value in test_metrics.items():
        print(f'{metric}: {value}')


if __name__ == '__main__':
    parser = ArgumentParser(description='Test deep learning segmentation models')

    parser.add_argument(
        '--config',
        help='Configurarion (.yaml) file to use',
        type=str
    )
    args = parser.parse_args()

    try:
        config = read_yaml(args.config)
    except TypeError:
        sys.exit('Expected path to config file via the --config flag.')
    
    print('PyTorch version:  ', torch.__version__)
    print('Is CUDA available:', torch.cuda.is_available())

    dataloaders, sampler = get_dataloaders(config)
    net = get_model(config)

    test_and_predict(net, dataloaders, sampler, config)