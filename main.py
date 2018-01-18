import matplotlib.pyplot as plt
from smallnorb.dataset import SmallNORBDataset


plt.ion()


if __name__ == '__main__':

    # Initialize the dataset from the folder in which
    # dataset archives have been uncompressed
    dataset = SmallNORBDataset(dataset_root='./smallnorb/')

    # Dump all images to disk
    dataset.export_to_jpg(export_dir='smallnorb_export')

    # Explore random examples of the training set
    # to show how data look like
    dataset.explore_random_examples(dataset_split='train')
