import struct
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import join
from utils import matrix_type_from_magic


class SmallNORBExample:

    def __init__(self):
        self.image_1   = None
        self.image_2   = None
        self.category  = None
        self.instance  = None
        self.elevation = None
        self.azimuth   = None
        self.lighting  = None

    def show(self, axes):
        axes[0].imshow(self.image_1, cmap='gray')
        axes[1].imshow(self.image_2, cmap='gray')


class SmallNORBDataset:

    def __init__(self, dataset_root):
        """
        Initialize small NORB dataset wrapper
        
        Parameters
        ----------
        dataset_root: str
            Path to directory where small NORB archives have been extracted.
        """
        self.num_examples = 24300
        self.dataset_root = dataset_root

        # List small NORB files and store path for each file
        # For ease of compatibility the original filename is kept
        train_files = {key: join(self.dataset_root,
                                 'smallnorb-5x46789x9x18x6x2x96x96-training-{}.mat'.format(key))
                       for key in ['cat', 'info', 'dat']}
        test_files = {key: join(self.dataset_root,
                                'smallnorb-5x01235x9x18x6x2x96x96-testing-{}.mat'.format(key))
                      for key in ['cat', 'info', 'dat']}

        self.dataset_files = {'train': train_files,
                              'test':  test_files}

        self.data = {'train': [SmallNORBExample() for _ in range(self.num_examples)],
                     'test':  [SmallNORBExample() for _ in range(self.num_examples)]}

        for data_split in ['train']:                  # todo , 'test']:
            self._fill_data_structures(data_split)

    def _fill_data_structures(self, dataset_split):
        dat_data  = self._parse_NORB_dat_file(self.dataset_files[dataset_split]['dat'])
        cat_data  = self._parse_NORB_cat_file(self.dataset_files[dataset_split]['cat'])
        info_data = self._parse_NORB_info_file(self.dataset_files[dataset_split]['info'])
        for i in range(self.num_examples):
            self.data[dataset_split][i].image_1   = dat_data[2 * i]
            self.data[dataset_split][i].image_2   = dat_data[2 * i + 1]
            self.data[dataset_split][i].category  = cat_data[i]
            self.data[dataset_split][i].instance  = info_data[i][0]
            self.data[dataset_split][i].elevation = info_data[i][1]
            self.data[dataset_split][i].azimuth   = info_data[i][2]
            self.data[dataset_split][i].lighting  = info_data[i][3]

    @staticmethod
    def _parse_small_NORB_header(file_pointer):
        """
        Parse header of small NORB binary file
        
        Parameters
        ----------
        file_pointer: BufferedReader
            File pointer just opened in a small NORB binary file

        Returns
        -------
        file_header_data: dict
            Dictionary containing header information
        """
        # Read magic number
        magic = struct.unpack('<BBBB', file_pointer.read(4))  # '<' is little endian)

        # Read dimensions
        dimensions = []
        num_dims, = struct.unpack('<i', file_pointer.read(4))  # '<' is little endian)
        for _ in range(num_dims):
            dimensions.extend(struct.unpack('<i', file_pointer.read(4)))

        file_header_data = {'magic_number': magic,
                            'matrix_type': matrix_type_from_magic(magic),
                            'dimensions': dimensions}
        return file_header_data

    @staticmethod
    def _parse_NORB_cat_file(file_path):
        """
        Parse small NORB category file
        
        Parameters
        ----------
        file_path: str
            Path of the small NORB `*-cat.mat` file

        Returns
        -------
        examples: ndarray
            Ndarray of shape (24300,) containing the category of each example
        """
        with open(file_path, mode='rb') as f:
            header = SmallNORBDataset._parse_small_NORB_header(f)

            num_examples, = header['dimensions']

            struct.unpack('<BBBB', f.read(4))  # ignore this integer
            struct.unpack('<BBBB', f.read(4))  # ignore this integer

            examples = np.zeros(shape=num_examples, dtype=np.int32)
            for i in tqdm(range(num_examples), desc='Loading categories...'):
                category, = struct.unpack('<i', f.read(4))
                examples[i] = category

            return examples

    @staticmethod
    def _parse_NORB_dat_file(file_path):
        """
        Parse small NORB data file

        Parameters
        ----------
        file_path: str
            Path of the small NORB `*-dat.mat` file

        Returns
        -------
        examples: ndarray
            Ndarray of shape (48600, 96, 96) containing images couples. Each image couple
            is stored in position [i, :, :] and [i+1, :, :]
        """
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
        """
        Parse small NORB information file

        Parameters
        ----------
        file_path: str
            Path of the small NORB `*-info.mat` file

        Returns
        -------
        examples: ndarray
            Ndarray of shape (24300,4) containing the additional info of each example.
            
             - column 1: the instance in the category (0 to 9)
             - column 2: the elevation (0 to 8, which mean cameras are 30, 35,40,45,50,55,60,65,70 
               degrees from the horizontal respectively)
             - column 3: the azimuth (0,2,4,...,34, multiply by 10 to get the azimuth in degrees)
             - column 4: the lighting condition (0 to 5)
        """
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
    _, axes = plt.subplots(nrows=1, ncols=2)
    for example in dataset.data['train']:
        example.show(axes)
        plt.waitforbuttonpress()
        plt.cla()
