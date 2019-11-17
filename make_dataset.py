import os
import sys
import logging
import argparse

from utils import dataset


def main(dataset_dir, subset):

    if not os.path.exists(dataset_dir):
        raise OSError('Directory {} does not exist'.format(dataset))

    if subset == 'Shanghai_A':
        for split in ['train_data', 'test_data']:
            section_dir = os.path.join(dataset_dir, 'part_A_final', split)
            save_dir = os.path.join(section_dir, 'densities')

            # Save dir is where all density maps are going to be saved
            if not os.path.exists(save_dir):
                logging.info('Creating {} directory'.format(save_dir))
                os.mkdir(save_dir)
            data_loader = dataset.Shanghai_A(section_dir, save_dir)
            data_loader.create_groundtruth()

    elif subset == 'Shanghai_B':
        for split in ['train_data', 'test_data']:
            section_dir = os.path.join(dataset_dir, 'part_B_final', split)
            save_dir = os.path.join(section_dir, 'densities')

            # Save dir is where all density maps are going to be saved
            if not os.path.exists(save_dir):
                logging.info('Creating {} directory'.format(save_dir))
                os.mkdir(save_dir)

            data_loader = dataset.Shanghai_B(section_dir, save_dir)
            data_loader.create_groundtruth()

    else:
        raise ValueError('Subset not available for the moment')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=str, help='Path to dataset directory')
    parser.add_argument('subset', type=str, help='Shanghai_A or Shanghai_B for the moment')
    return vars(parser.parse_args())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(**parse_arguments(sys.argv[1:]))
