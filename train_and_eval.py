import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from typing import Any

import utils.distributed_utils as dist
import utils.ssl_utils as ssl
from data import get_dataloaders
from models import get_model
from metrics.loss_functions import get_loss
from metrics.numpy_metrics import RunningMetrics
from utils.config_files_utils import read_yaml
from utils.lr_scheduler import get_scheduler
from utils.summaries import write_summaries
from utils.torch_utils import load_from_checkpoint, logits_to_preds
from utils.validation_utils import ValidationMonitor


def train_and_evaluate(net: torch.nn.Module, dataloaders: torch.utils.data.DataLoader,
                       sampler: torch.utils.data.Sampler, config: dict[str, Any]) -> None:

    def train_step(net: torch.nn.Module, sample: dict[str, torch.Tensor], running_train_metrics: RunningMetrics,
                   loss_fn: torch.nn, optimizer: torch.optim, num_classes: int, device: torch.device) -> dict[str, float]:
        net.train()

        inputs = sample['inputs'].to(device)
        targets = sample['labels'].to(device)

        scaler = torch.cuda.amp.GradScaler()

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            outputs = net(inputs)
            loss_tensor = loss_fn(outputs, targets)
        
        scaler.scale(loss_tensor).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad(set_to_none=True)
        
        dist.all_reduce_mean(loss_tensor)
        loss = loss_tensor.item()

        is_multiclass = True if num_classes > 1 else False
        preds = logits_to_preds(outputs, is_multiclass=is_multiclass)

        running_train_metrics.update(preds, targets)
        train_metrics = running_train_metrics.get_scores()
        train_metrics['loss'] = loss

        return train_metrics


    def evaluate(net: torch.nn.Module, val_loader: torch.utils.data.DataLoader, config: dict[str, Any],
                 loss_fn: torch.nn, device: torch.device) -> dict[str, float]:
        net.eval()

        num_classes = int(config['MODEL']['num_classes'])
        ignore_index = config['SOLVER']['ignore_index']
        running_val_metrics = RunningMetrics(num_classes, ignore_index)

        all_losses_tensor = torch.zeros(len(val_loader)).to(device)

        for idx, sample in enumerate(val_loader):
            inputs = sample['inputs'].to(device)
            targets = sample['labels'].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                outputs = net(inputs)

                loss_tensor = loss_fn(outputs, targets)
                all_losses_tensor[idx] = loss_tensor.item()

            is_multiclass = True if num_classes > 1 else False
            preds = logits_to_preds(outputs, is_multiclass=is_multiclass)

            running_val_metrics.update(preds, targets)
        
        all_losses_tensor = all_losses_tensor.mean()
        dist.all_reduce_mean(all_losses_tensor)
        loss = all_losses_tensor.item()
        
        val_metrics = running_val_metrics.get_scores()
        val_metrics['loss'] = loss
        running_val_metrics.reset()

        return val_metrics


    # To continue training a previous model
    checkpoint_path = config['CHECKPOINT']['load_from_checkpoint']

    if checkpoint_path is not None:
        load_from_checkpoint(net, checkpoint_path)
    
    # To load encoder weights such as imagenet or from SSL
    weights_path = config['CHECKPOINT']['load_from_pretrained_weights']

    if weights_path is not None:
        ssl.load_pretrained_weights(net, weights_path)
        ssl.freeze_encoder_weights(net)

    save_path = config['CHECKPOINT']['save_path']
    assert type(save_path) == str
    os.makedirs(save_path, exist_ok=True)
    
    device = dist.get_current_device()

    loss_fn = get_loss(config, device)

    learning_rate = float(config['SOLVER']['lr_base'])
    weight_decay = float(config['SOLVER']['weight_decay'])
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_scheduler(config, optimizer)

    num_classes = int(config['MODEL']['num_classes'])
    ignore_index = config['SOLVER']['ignore_index']
    running_train_metrics = RunningMetrics(num_classes, ignore_index)

    writer = SummaryWriter(save_path)

    start_epoch = int(config['SOLVER']['start_epoch'])
    num_epochs = int(config['SOLVER']['num_epochs'])

    val_monitor = ValidationMonitor(patience=3, delta=0.)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        if dist.is_distributed():
            sampler.set_epoch(epoch)

        for idx, sample in enumerate(dataloaders['train'], start=1):
            step = idx + (epoch - start_epoch)*len(dataloaders['train'])

            train_metrics = train_step(net, sample=sample, running_train_metrics=running_train_metrics,
                                       loss_fn=loss_fn, optimizer=optimizer, num_classes=num_classes,
                                       device=device)

            # Printing metrics
            if step % int(config['CHECKPOINT']['train_metrics_steps']) == 0:
                write_summaries(writer, metrics=train_metrics, step=step, mode='train',
                                learning_rate=scheduler.get_last_lr()[0])

                if dist.is_main_process():
                    print((
                        f'Step: {step}, '
                        f'Epoch: {epoch}, '
                        f'Loss: {train_metrics["loss"]:.3f}, '
                        f'Lr: {scheduler.get_last_lr()[0]:.5f}, '
                        f'Train mean IoU: {train_metrics["mean_iou"]:.3f}'
                    ))

            # Validation step
            if step % int(config['CHECKPOINT']['eval_steps']) == 0:

                val_metrics = evaluate(net, val_loader=dataloaders['val'], config=config,
                                       loss_fn=loss_fn, device=device)
                
                # Saving the best model
                if val_monitor.improved_model(mean_iou=val_metrics['mean_iou']):
                    torch.save(net.state_dict(), f'{save_path}/best_model.pt')

                # Unfreezing encoder weights if validation loss stops improving
                if weights_path is not None and val_monitor.reached_plateau(val_loss=val_metrics['loss']):
                    ssl.unfreeze_encoder_weights(net)

                write_summaries(writer, metrics=val_metrics, step=step, mode='val',
                                learning_rate=scheduler.get_last_lr()[0])
        
        # Adjusting the learning rate
        scheduler.step()
        running_train_metrics.reset()


if __name__ == '__main__':
    parser = ArgumentParser(description='Train and eval deep learning segmentation models')

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

    dist.init_distributed_process()

    if dist.is_main_process():
        print('PyTorch version:  ', torch.__version__)
        print('Is CUDA available:', torch.cuda.is_available())
        print('Number of GPUs:   ', torch.distributed.get_world_size())

    dataloaders, sampler = get_dataloaders(config)
    net = get_model(config)

    train_and_evaluate(net, dataloaders, sampler, config)