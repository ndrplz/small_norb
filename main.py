import matplotlib.pyplot as plt
from dataset import SmallNORBDataset


plt.ion()


if __name__ == '__main__':

    dataset = SmallNORBDataset(dataset_root='/media/minotauro/DATA/smallnorb/')

    dataset.explore_random_examples(dataset_split='train')
