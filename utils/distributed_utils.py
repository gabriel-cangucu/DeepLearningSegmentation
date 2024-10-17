import torch
import sys
import os


def is_distributed() -> bool:
    return (torch.distributed.is_available() and 
            torch.distributed.is_initialized() and
            torch.distributed.get_world_size() > 1)


def get_rank() -> int:
    if not is_distributed():
        return 0
    else:
        return torch.distributed.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def get_current_device() -> torch.device:
    rank = get_rank()
    device = torch.device(f'cuda:{rank}')

    return device


def init_distributed_process() -> None:
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])

    elif torch.cuda.is_available():
        rank, gpu, world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    
    else:
        print('Does not support training without GPU.')
        sys.exit(1)
    
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )

    torch.cuda.set_device(gpu)
    torch.distributed.barrier()


def all_reduce_mean(tensor: torch.tensor) -> torch.tensor:
    tensor = tensor.detach().cpu()

    if torch.distributed.get_world_size() > 1:
        tensor = torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.AVG)

    return tensor