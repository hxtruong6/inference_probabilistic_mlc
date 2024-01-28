import numpy as np

print(f"Numpy version: {np.__version__}")

import pandas as pd

import scipy.io.arff as arff


# Define a constant for seed
SEED = 6


class HandleMulanDatasetForMultiLabelArffFile:
    """Class to handle Mulan datasets stored in ARFF files."""

    def __init__(
        self,
        path,
        dataset_name,
        target_at_first=False,
        is_train=False,
        is_test=False,
        train_index=None,
    ):
        """Initialize the dataset handler."""
        self.path = path
        self.data = arff.loadarff(self.path)
        self.df = pd.DataFrame(self.data[0])

        # Handle train-test split for some datasets
        if is_train:
            self.df = self.df.sample(frac=0.8, random_state=SEED)
            self.train_index = self.df.index
        elif is_test:
            self.df = self.df.drop(train_index)

        self.dataset_name = dataset_name
        y_split_index = self._get_Y_split_index()

        if target_at_first:
            self.Y = self.df.iloc[:, :y_split_index].astype(int)
            self.X = self.df.iloc[:, y_split_index:]
        else:
            self.X = self.df.iloc[:, :-y_split_index]
            self.Y = self.df.iloc[:, -y_split_index:].astype(int)

        # convert to numpy array
        self.X = self.X.to_numpy()
        self.Y = self.Y.to_numpy()

    def _get_Y_split_index(self):
        """Get the index for splitting Y from X based on the dataset name."""
        if self.dataset_name == "emotions":
            return 6
        elif self.dataset_name == "corel5k":
            return 374
        elif self.dataset_name == "bitex":
            return 159
        elif self.dataset_name == "scene":
            return 6
        elif self.dataset_name == "yeast":
            return 14
        elif self.dataset_name == "CAL500":
            return 174
        elif self.dataset_name == "mediaMill":
            return 101
        elif self.dataset_name == "VirusGO_sparse":
            return 6
        elif self.dataset_name == "Water-quality":
            return 14
        elif self.dataset_name == "CHD_49":
            return 6

        else:
            raise Exception("Dataset name is not supported")
