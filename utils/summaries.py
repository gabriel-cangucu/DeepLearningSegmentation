import torch


def write_summaries(writer: torch.utils.tensorboard.SummaryWriter, metrics: dict[str, float],
                    loss: float, step: int, mode: str='train', optimizer: torch.optim=None) -> None:
    for metric in metrics.keys():
        writer.add_scalars(
            metric,
            tag_scalar_dict={f'{mode}': metrics[metric]},
            global_step=step
        )
    
    writer.add_scalar(f'loss/{mode}', scalar_value=loss, global_step=step)
    
    if optimizer is not None:
        writer.add_scalar(
            'learning_rate',
            scalar_value=optimizer.param_groups[0]['lr'],
            global_step=step
        )