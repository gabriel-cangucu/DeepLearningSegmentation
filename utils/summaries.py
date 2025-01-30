import torch
from typing import Optional


def write_summaries(writer: torch.utils.tensorboard.SummaryWriter, metrics: dict[str, float],
                    step: int, mode: str='train', learning_rate: Optional[float]=None) -> None:
    for metric in metrics.keys():
        writer.add_scalars(
            metric,
            tag_scalar_dict={mode: metrics[metric]},
            global_step=step
        )
    
    if learning_rate is not None:
        writer.add_scalar(
            'learning_rate',
            scalar_value=learning_rate,
            global_step=step
        )