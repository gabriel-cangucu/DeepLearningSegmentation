import pandas as pd
import sys
import os
import pickle
import torch
from torchvision import transforms


def get_dataloader(root_dir: str, csv_path: str, batch_size: int=32, num_workers: int=4,
                   shuffle: bool=True, transform: transforms.Compose=None
                   ) -> torch.utils.data.DataLoader:
    dataset = PastisDataset(root_dir=root_dir, csv_path=csv_path, transform=transform)

    # Distributed dataloader
    if torch.distributed.get_world_size() > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                 num_workers=num_workers, pin_memory=True,
                                                 sampler=sampler)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                 num_workers=num_workers)

    return dataloader


class PastisDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir: str, csv_path: str, transform: transforms.Compose=None) -> None:
        self.root_dir = root_dir
        self.transform = transform

        try:
            self.data_files = pd.read_csv(csv_path, header=None)[0]
        except FileNotFoundError:
            sys.exit(f'Invalid path for csv file: {csv_path}')
    

    def __len__(self) -> int:
        return len(self.data_files)
    

    def __getitem__(self, idx: list[str]) -> dict[str, torch.tensor]:
        img_path = os.path.join(self.root_dir, self.data_files.iloc[idx])

        try:
            with open(img_path, 'rb') as handle:
                sample = pickle.load(handle, encoding='latin1')
        except FileNotFoundError:
            sys.exit(f'Invalid path for data file: {img_path}')
        
        if self.transform:
            sample = self.transform(sample)

        return sample