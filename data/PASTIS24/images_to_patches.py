import geopandas as gpd
import os
import numpy as np
import pickle
import datetime
import torch
import argparse
from tqdm import tqdm


def get_day_of_year(date: str) -> int:
    Y = date[:4]
    m = date[4:6]
    d = date[6:]

    date = "%s.%s.%s" % (Y, m, d)
    dt = datetime.datetime.strptime(date, '%Y.%m.%d')

    return dt.timetuple().tm_yday


def unfold_reshape(img: torch.Tensor, size: int):
    if len(img.shape) == 4:
        T, C, H, W = img.shape
        img = img.unfold(2, size=size, step=size).unfold(3, size=size, step=size)
        img = img.reshape(T, C, -1, size, size).permute(2, 0, 1, 3, 4)

    elif len(img.shape) == 3:
        _, H, W = img.shape
        img = img.unfold(1, size=size, step=size).unfold(2, size=size, step=size)
        img = img.reshape(-1, size, size)

    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Turns original high res images into smaller patches')

    parser.add_argument('--rootdir', type=str, help='PASTIS24 root dir')
    parser.add_argument('--savedir', type=str, help='Where to save new data')
    parser.add_argument('--size', type=int, default=24, help='Size of extracted windows')

    args = parser.parse_args()

    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)
    
    meta_patch = gpd.read_file(os.path.join(args.rootdir, 'metadata.geojson'))

    labels = []

    for i in tqdm(range(meta_patch.shape[0])):
        img = np.load(os.path.join(args.rootdir, 'DATA_S2/S2_%d.npy' % meta_patch['ID_PATCH'].iloc[i]))
        lab = np.load(os.path.join(args.rootdir, 'ANNOTATIONS/TARGET_%d.npy' % meta_patch['ID_PATCH'].iloc[i]))
        ids = np.load(os.path.join(args.rootdir, 'ANNOTATIONS/ParcelIDs_%d.npy' % meta_patch['ID_PATCH'].iloc[i]))

        dates = eval(meta_patch['dates-S2'].iloc[i])

        doy = np.array([get_day_of_year(str(d)) for d in dates.values()])
        idx = np.argsort(doy)

        img = img[idx]
        doy = doy[idx]

        unfolded_images = unfold_reshape(torch.Tensor(img), args.size).numpy()
        unfolded_labels = unfold_reshape(torch.Tensor(lab.astype(np.int64)), args.size).numpy()

        for j in range(unfolded_images.shape[0]):
            sample = {'img': unfolded_images[j], 'labels': unfolded_labels[j], 'doy': doy}

            with open(os.path.join(args.savedir, '%d_%d.pickle' % (meta_patch['ID_PATCH'].iloc[i], j)), 'wb') as f:
                pickle.dump(sample, f)