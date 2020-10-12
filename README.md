# small_norb

**Plug-and-play python wrapper around the small NORB dataset.**

Since I saw no plug-and-play python wrappers around the [small NORB dataset](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/) (which is distributed in *binary matrix* file format) I made this simple class to encapsule the binary data parsing.

To install requirements: `pip install -r requirements.txt`

## Small NORB Dataset
The [small NORB dataset](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/) contains images of 50 toys belonging to 5 generic categories: four-legged animals, human figures, airplanes, trucks, and cars. The objects were imaged by two cameras under 6 lighting conditions, 9 elevations (30 to 70 degrees every 5 degrees), and 18 azimuths (0 to 340 every 20 degrees). The training set is composed of 5 instances of each category (instances 4, 6, 7, 8 and 9), and the test set of the remaining 5 instances (instances 0, 1, 2, 3, and 5). The dataset features 24300 train and 24300 test examples.

The dataset is distributed in the form of compressed binary matrices, downloadable through the following links:
- **Train Set:** ([data](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz), [categories](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz), [info](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz))
- **Test Set:** ([data](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz), [categories](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz), [info](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz))

## Python Wrapper Usage

Once downloaded data have been uncompressed in a `small_norb_root` directory, usage is as simple as:
````
     dataset = SmallNORBDataset(dataset_root='small_norb_root')
````
once initialized, small NORB data will be available in `dataset.data['train']` and `dataset.data['test']` respectively.

## Dataset Exploration

To check that the dataset has been correctly initialized, the method `explore_random_examples` can be used as follows:
````
     dataset.explore_random_examples(dataset_split='train')
````
If everything went well, the output should look like the following:
<p align="center"><img src="https://github.com/ndrplz/small_norb/blob/master/docs/small_norb_explore.png" width="500"></p>

