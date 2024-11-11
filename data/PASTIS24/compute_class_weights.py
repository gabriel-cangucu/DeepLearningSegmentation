import os
import argparse
import pickle
import numpy as np
import csv
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computes a weight vector based on class frequencies for PASTIS')

    parser.add_argument('--rootdir', type=str, help='PASTIS24 root dir')
    parser.add_argument('--fold', type=int, help='Which fold to use as train')
    args = parser.parse_args()

    fold_files = {
        1: 'folds_1_123_paths.csv',
        2: 'folds_2_234_paths.csv',
        3: 'folds_3_345_paths.csv',
        4: 'folds_4_451_paths.csv',
        5: 'folds_5_512_paths.csv'
    }

    csv_filename = os.path.join('fold-paths', fold_files[args.fold])

    num_classes = 20
    class_counts = np.zeros(num_classes)

    with open(os.path.join(args.rootdir, csv_filename)) as csv_file:
        datareader = csv.reader(csv_file)

        for row in tqdm(datareader):
            sample_name = row[0]

            with open(os.path.join(args.rootdir, sample_name), 'rb') as pickle_handler:
                sample = pickle.load(pickle_handler, encoding='latin1')
                label = sample['labels'].flatten()
                
                class_counts += np.bincount(label, minlength=num_classes)[:num_classes]
    
    total_pixels = np.sum(class_counts)
    class_weights = total_pixels / (num_classes*class_counts)

    print(np.round(class_weights, decimals=2).tolist())