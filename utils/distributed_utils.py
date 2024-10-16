import torch


def all_reduce_mean(tensor: torch.tensor) -> torch.tensor:
    tensor = tensor.detach().cpu()
    loss = torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.AVG)

    return loss