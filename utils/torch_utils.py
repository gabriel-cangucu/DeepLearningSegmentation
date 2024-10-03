import os
import torch


def load_from_checkpoint(net: torch.nn.Module, checkpoint_path: str):
    assert type(checkpoint_path) == str

    if os.path.isfile(checkpoint_path):
        print(f'Loading model from checkpoint: {checkpoint_path}')

        saved_net = torch.load(checkpoint_path)
        net.load_state_dict(saved_net)
    else:
        raise FileNotFoundError(f'Invalid checkpoint path: {checkpoint_path}')