import pandas as pd
import scipy.io.arff as arff
from sklearn.model_selection import KFold

# Define a constant for seed
SEED = 6


class MultiLabelArffDataset:
    """Class to handle Mulan datasets stored in ARFF files."""

    def __init__(
        self,
        dataset_name,
        path=None,
        target_at_first=False,
        X=None,
        Y=None,
    ):
        """Initialize the dataset handler."""
        self.dataset_name = dataset_name

        if (
            (X is not None)
            and (Y is not None)
            and isinstance(X, pd.DataFrame)
            and isinstance(Y, pd.DataFrame)
        ):
            self.X = X.to_numpy()
            self.Y = Y.to_numpy()
            print(f"\U0001F4A5 Number of features: {self.X.shape}")
            print(f"\U0001F4A6 Number of labels: {self.Y.shape}")
            return

        self.path = path
        self.data = arff.loadarff(self.path)
        self.df = pd.DataFrame(self.data[0])

        y_split_index = self._get_Y_split_index()

        if target_at_first:
            self.Y = self.df.iloc[:, :y_split_index].astype(int)
            self.X = self.df.iloc[:, y_split_index:]
        else:
            self.X = self.df.iloc[:, :-y_split_index]
            self.Y = self.df.iloc[:, -y_split_index:].astype(int)

        self.X = self.X.to_numpy()
        self.Y = self.Y.to_numpy()

        print(f"\U0001F4A5 Number of features: {self.X.shape}")
        print(f"\U0001F4A6 Number of labels: {self.Y.shape}")

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

    def get_cross_validation_folds(self, n_splits=5, shuffle=True, random_state=SEED):
        """Provides cross-validation folds for the dataset.

        Args:
            n_splits: Number of folds for cross-validation.
            shuffle:  Whether to shuffle the data before splitting.
            random_state: Random seed for reproducibility.

        Yields:
            Tuple of (train_index, test_index) for each fold.
        """
        skf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for train_index, test_index in skf.split(self.X, self.Y):
            yield train_index, test_index
