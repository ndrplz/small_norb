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

        self.train_images = self._parse_NORB_data_file(self.dataset_files['train']['dat'])
        self.test_images  = self._parse_NORB_data_file(self.dataset_files['test']['dat'])

    @staticmethod
    def _parse_NORB_data_file(file_path):

        with open(file_path, mode='rb') as f:

            # Read magic number
            magic = struct.unpack('<BBBB', f.read(4))  # '<' is little endian)
            print('Magic number: {} --> {}'.format(magic, matrix_type_from_magic(magic)))

            # Read dimensions
            dimensions = []
            num_dims, = struct.unpack('<i', f.read(4))  # '<' is little endian)
            for _ in range(num_dims):
                dimensions.extend(struct.unpack('<i', f.read(4)))
            print('Dimensions: {}'.format(dimensions))

            num_examples, channels, height, width = dimensions

            examples = np.zeros(shape=(num_examples * channels, height, width), dtype=np.uint8)

            for i in tqdm(range(num_examples * channels), desc='Loading images...'):

                # Read raw image data and restore shape as appropriate
                image = struct.unpack('<' + height * width * 'B', f.read(height * width))
                image = np.uint8(np.reshape(image, newshape=(height, width)))

                examples[i] = image

        return examples


if __name__ == '__main__':

    plt.ion()

    dataset = SmallNORBDataset(dataset_root='/media/minotauro/DATA/smallnorb/')
    for image in dataset.test_images:
        plt.imshow(image, cmap='gray')
        plt.waitforbuttonpress()
