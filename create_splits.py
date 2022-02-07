import argparse
import glob
import os
import shutil

import numpy as np

from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    # TODO: Implement function

    # Create train/val/test directories
    train_dir = os.path.join(destination, "train")
    val_dir = os.path.join(destination, "val")
    test_dir = os.path.join(destination, "test")

    try:
        os.mkdir(train_dir)
        os.mkdir(val_dir)
        os.mkdir(test_dir)
    except OSError as error:
        # print(error)
        pass

    # Get list of data to use
    record_files = [filename for filename in glob.glob(f'{source}/*.tfrecord')]
    if len(record_files) == 0:
        print("No data to split, Exit")
        return
    np.random.shuffle(record_files)

    # Split dataset
    train_files, val_files, test_files = np.split(record_files, [int(.75 * len(record_files)), int(.9 * len(record_files))])

    for file in train_files:
        shutil.move(file, train_dir)

    for file in val_files:
        shutil.move(file, val_dir)

    for file in test_files:
        shutil.move(file, test_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)
