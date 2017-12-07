import struct
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import join
from utils import matrix_type_from_magic


class SmallNORBDataset:

    def __init__(self, dataset_root):
        """
        Initialize small NORB dataset wrapper
        
        Parameters
        ----------
        dataset_root: str
            Path to directory where small NORB archives have been extracted.
        """
        self.dataset_root = dataset_root

        # List small NORB files and store path for each file
        # For ease of compatibility the original filename is kept
        train_files = {key: join(self.dataset_root,
                                 'smallnorb-5x46789x9x18x6x2x96x96-training-{}.mat'.format(key))
                       for key in ['cat', 'info', 'dat']}
        test_files = {key: join(self.dataset_root,
                                'smallnorb-5x01235x9x18x6x2x96x96-testing-{}.mat'.format(key))
                      for key in ['cat', 'info', 'dat']}
        self.dataset_files = {'train': train_files, 'test': test_files}

        self.train_dat = self._parse_NORB_data_file(self.dataset_files['train']['dat'])
        self.test_dat  = self._parse_NORB_data_file(self.dataset_files['test']['dat'])

        self.train_info = self._parse_NORB_info_file(self.dataset_files['train']['info'])
        self.test_info  = self._parse_NORB_info_file(self.dataset_files['test']['info'])

    @staticmethod
    def _parse_small_NORB_header(file_pointer):

        # Read magic number
        magic = struct.unpack('<BBBB', file_pointer.read(4))  # '<' is little endian)
        print('Magic number: {} --> {}'.format(magic, matrix_type_from_magic(magic)))

        # Read dimensions
        dimensions = []
        num_dims, = struct.unpack('<i', file_pointer.read(4))  # '<' is little endian)
        for _ in range(num_dims):
            dimensions.extend(struct.unpack('<i', file_pointer.read(4)))
        print('Dimensions: {}'.format(dimensions))

        file_header_data = {'magic_number': magic,
                            'matrix_type': matrix_type_from_magic(magic),
                            'dimensions': dimensions}
        return file_header_data

    @staticmethod
    def _parse_NORB_data_file(file_path):

        with open(file_path, mode='rb') as f:

            header = SmallNORBDataset._parse_small_NORB_header(f)

            num_examples, channels, height, width = header['dimensions']

            examples = np.zeros(shape=(num_examples * channels, height, width), dtype=np.uint8)

            for i in tqdm(range(num_examples * channels), desc='Loading images...'):

                # Read raw image data and restore shape as appropriate
                image = struct.unpack('<' + height * width * 'B', f.read(height * width))
                image = np.uint8(np.reshape(image, newshape=(height, width)))

                examples[i] = image

        return examples

    @staticmethod
    def _parse_NORB_info_file(file_path):

        with open(file_path, mode='rb') as f:

            header = SmallNORBDataset._parse_small_NORB_header(f)

            struct.unpack('<BBBB', f.read(4))  # ignore this integer

            num_examples, num_info = header['dimensions']

            examples = np.zeros(shape=(num_examples, num_info), dtype=np.int32)

            for r in tqdm(range(num_examples), desc='Loading info...'):
                for c in range(num_info):
                    info, = struct.unpack('<i', f.read(4))
                    examples[r, c] = info

        return examples


if __name__ == '__main__':

    plt.ion()

    dataset = SmallNORBDataset(dataset_root='/media/minotauro/DATA/smallnorb/')
    for image in dataset.test_dat:
        plt.imshow(image, cmap='gray')
        plt.waitforbuttonpress()
