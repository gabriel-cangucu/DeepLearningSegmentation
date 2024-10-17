import os
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
from utils.torch_utils import load_from_checkpoint
from utils.lr_scheduler import get_scheduler
from utils.summaries import write_summaries


def train_and_evaluate(net: torch.nn.Module, dataloaders: torch.utils.data.DataLoader,
                       sampler: torch.utils.data.Sampler, config: dict[str, Any]) -> None:
    def train_step(net: torch.nn.Module, sample: dict[str, torch.tensor],
                   running_train_metrics: RunningMetrics, loss_fn: torch.nn, optimizer: torch.optim,
                   device: torch.device) -> tuple[float, dict[str, float]]:
        net.train()

        inputs = sample['inputs'].to(device)
        targets = sample['labels'].to(device)

        scaler = torch.GradScaler()

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            outputs = net(inputs)
            loss_tensor = loss_fn(outputs, targets)
        
        scaler.scale(loss_tensor).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad(set_to_none=True)
        loss = dist.all_reduce_mean(loss_tensor).item()

        running_train_metrics.update(outputs, targets)
        train_metrics = running_train_metrics.get_scores()

        return loss, train_metrics


    def evaluate(net: torch.nn.Module, val_loader: torch.utils.data.DataLoader,
                 config: dict[str, Any], loss_fn: torch.nn, device: torch.device
                 ) -> tuple[float, dict[str, float]]:
        net.eval()

        num_classes = int(config['MODEL']['num_classes'])
        ignore_index = int(config['SOLVER']['ignore_index'])
        running_val_metrics = RunningMetrics(num_classes, ignore_index)

        all_losses_tensor = torch.zeros(len(val_loader)).to(device)

        for idx, sample in enumerate(val_loader):
            inputs = sample['inputs'].to(device)
            targets = sample['labels'].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                outputs = net(inputs)
                loss_tensor = loss_fn(outputs, targets)

                all_losses_tensor[idx] = loss_tensor.item()
            
            running_val_metrics.update(outputs, targets)
        
        all_losses_tensor = all_losses_tensor.mean()
        loss = dist.all_reduce_mean(all_losses_tensor).item()
        
        val_metrics = running_val_metrics.get_scores()
        running_val_metrics.reset()

        return loss, val_metrics


    checkpoint_path = config['CHECKPOINT']['load_from_checkpoint']

    if checkpoint_path:
        load_from_checkpoint(net, checkpoint_path)

    save_path = config['CHECKPOINT']['save_path']
    assert type(save_path) == str

    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
    
    device = dist.get_current_device()

    loss_fn = get_loss(config, device, reduction='mean')

    learning_rate = float(config['SOLVER']['lr_base'])
    weight_decay = float(config['SOLVER']['weight_decay'])
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    num_train_steps = len(dataloaders['train'])
    scheduler = get_scheduler(config, optimizer, num_train_steps)

    num_classes = config['MODEL']['num_classes']
    ignore_index = config['SOLVER']['ignore_index']
    running_train_metrics = RunningMetrics(num_classes, ignore_index)

    writer = SummaryWriter(save_path)

    start_epoch = int(config['SOLVER']['start_epoch'])
    num_epochs = int(config['SOLVER']['num_epochs'])

    BEST_IOU = 0.

    for epoch in range(start_epoch, start_epoch + num_epochs):
        if dist.is_distributed():
            sampler.set_epoch(epoch)

        for idx, sample in enumerate(dataloaders['train'], start=1):
            step = idx + (epoch - start_epoch)*num_train_steps

            loss, train_metrics = train_step(net, sample=sample, running_train_metrics=running_train_metrics,
                                             loss_fn=loss_fn, optimizer=optimizer, device=device)

            # Printing metrics
            if step % config['CHECKPOINT']['train_metrics_steps'] == 0:
                write_summaries(writer, metrics=train_metrics, loss=loss, step=step,
                                mode='train', optimizer=optimizer)

                if dist.is_main_process():
                    print((f'Step: {step}, '
                        f'Loss: {loss:.3f}, '
                        f'Lr: {scheduler.get_last_lr()[0]:.5f}, '
                        f'Mean IOU: {train_metrics["mean_iou"]:.3f}'
                    ))

            # Storing the best model
            if step % config['CHECKPOINT']['eval_steps'] == 0:
                loss, val_metrics = evaluate(net, val_loader=dataloaders['val'], config=config,
                                             loss_fn=loss_fn, device=device)
            
                if val_metrics['mean_iou'] > BEST_IOU:
                    torch.save(net.state_dict(), f'{save_path}/best_model.pt')
                    BEST_IOU = val_metrics['mean_iou']

                write_summaries(writer, metrics=val_metrics, loss=loss, step=step,
                                mode='val', optimizer=optimizer)
        
        # Adjusting the learning rate
        scheduler.step()
        running_train_metrics.reset()


if __name__ == '__main__':
    parser = ArgumentParser(description='Deep learning segmentation models')

    parser.add_argument(
        '--config',
        help='Configurarion (.yaml) file to use',
        type=str
    )

    args = parser.parse_args()
    config = read_yaml(args.config)

    dist.init_distributed_process()

    if dist.is_main_process():
        print('Is CUDA available:', torch.cuda.is_available())
        print('Number of GPUs:', torch.distributed.get_world_size())

    dataloaders, sampler = get_dataloaders(config)
    net = get_model(config)

    train_and_evaluate(net, dataloaders, sampler, config)