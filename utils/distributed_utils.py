import torch


def all_reduce_mean(tensor: torch.tensor) -> torch.tensor:
    tensor = tensor.detach().cpu()

    if torch.distributed.get_world_size() > 1:
        tensor = torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.AVG)

    return tensor