import os
import torch


def load_pretrained_weights(net: torch.nn.Module, weights_path: str) -> None:
    '''
    Loads the encoder pretrained weights in .pt format.
    '''
    assert type(weights_path) == str

    if os.path.isfile(weights_path):
        assert(os.path.splitext(weights_path)[1]) == '.pt'
        print(f'Loading encoder weights: {weights_path}')

        custom_weights = torch.load(weights_path)
        net.encoder.load_state_dict(custom_weights['state_dict'])
    else:
        raise FileNotFoundError(f'Invalid weights path: {weights_path}')


def freeze_encoder_weights(net: torch.nn.Module) -> None:
    for param in net.encoder.parameters():
        param.requires_grad = False


def unfreeze_encoder_weights(net: torch.nn.Module) -> None:
    for param in net.encoder.parameters():
        param.requires_grad = True
