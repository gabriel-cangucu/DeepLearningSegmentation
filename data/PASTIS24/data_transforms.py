import torch
import numpy as np
from typing import Any
from torchvision.transforms import v2 as transforms
from einops import rearrange


def get_transforms(config: dict[str, Any], is_training: bool) -> transforms.Compose:
    transform_list = []

    input_height = input_width = int(config['img_res'])
    max_seq_len = int(config['max_seq_len'])

    transform_list.append(ToTensor())
    transform_list.append(Normalize())
    transform_list.append(ConcatDates(height=input_height, width=input_width))
    transform_list.append(CutOrPad(max_seq_len=max_seq_len))

    return transforms.Compose(transform_list)


class ToTensor(object):
    '''
    Converts numpy arrays in a sample into torch tensors
    '''

    def __init__(self) -> None:
        pass


    def __call__(self, sample: dict[str, np.array]) -> dict[str, torch.tensor]:
        tensor_sample = {}

        tensor_sample['inputs'] = torch.tensor(sample['img'].astype(np.float32))
        tensor_sample['labels'] = torch.tensor(sample['labels'].astype(np.float32)).long()
        tensor_sample['dates'] = torch.tensor(np.array(sample['doy']))

        return tensor_sample


class Normalize(object):
    '''
    Normalizes inputs based on precomputed values
    '''

    def __init__(self) -> None:
        self.mean_fold1 = np.array([
            1165.9398193359375,
            1375.6534423828125,
            1429.2191162109375,
            1764.798828125,
            2719.273193359375,
            3063.61181640625,
            3205.90185546875,
            3319.109619140625,
            2422.904296875,
            1639.370361328125
        ]).astype(np.float32)

        self.std_fold1 = np.array([
            1942.6156005859375,
            1881.9234619140625,
            1959.3798828125,
            1867.2239990234375,
            1754.5850830078125,
            1769.4046630859375,
            1784.860595703125,
            1767.7100830078125,
            1458.963623046875,
            1299.2833251953125
        ]).astype(np.float32)


    def __call__(self, sample: dict[str, torch.tensor]) -> dict[str, torch.tensor]:
        mean_fold1 = rearrange(self.mean_fold1, 'c -> 1 c 1 1')
        std_fold1 = rearrange(self.std_fold1, 'c -> 1 c 1 1')

        sample['inputs'] = (sample['inputs'] - mean_fold1) / std_fold1

        return sample


class ConcatDates(object):
    '''
    Turns a 1D dates array to match the height and width of the input and concats it
    '''

    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width


    def __call__(self, sample: dict[str, torch.tensor]) -> dict[str, torch.tensor]:
        dates = sample['dates']
        dates = dates.repeat(1, self.height, self.width, 1)
        dates = rearrange(dates, 'c h w t -> t c h w')

        sample['inputs'] = torch.cat((sample['inputs'], dates), dim=1)
        del sample['dates']

        return sample


class CutOrPad(object):
    '''
    Pads time series with zeros to a max sequence length or cuts sequential parts
    '''
    
    def __init__(self, max_seq_len: int) -> None:
        self.max_seq_len = max_seq_len
    

    def __call__(self, sample: dict[str, torch.tensor]) -> dict[str, torch.tensor]:
        resized_inputs = sample['inputs']

        seq_len = resized_inputs.shape[0]
        diff = self.max_seq_len - seq_len

        # Padding with zeros
        if diff > 0:
            pad_shape = (diff, *resized_inputs.shape[1:])
            resized_inputs = torch.cat((resized_inputs, torch.zeros(pad_shape, dtype=torch.float32)), dim=0)
        # Cutting the sequence
        elif diff < 0:
            resized_inputs = resized_inputs[:self.max_seq_len]

        sample['inputs'] = resized_inputs
        
        return sample