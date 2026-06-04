import pandas as pd
import scipy.io.arff as arff
from sklearn.model_selection import KFold

# Number of labels per (dense) MULAN ARFF used in the paper.
_LABEL_COUNTS = {
    "emotions": 6,
    "scene": 6,
    "yeast": 14,
    "Water-quality": 14,
    "CHD_49": 6,
}


class MultiLabelArffDataset:
    """Class to handle Mulan datasets stored in ARFF files.

    Stores raw (unscaled) feature arrays. Callers are responsible for
    applying StandardScaler per fold AFTER train/test split to avoid
    data leakage.
    """

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

        if X is not None and Y is not None and isinstance(X, pd.DataFrame) and isinstance(Y, pd.DataFrame):
            self.X, self.Y = self._to_numpy(X, Y)
            return

        self.path = path
        data = arff.loadarff(self.path)
        df = pd.DataFrame(data[0])

        n_labels = self._get_label_count()
        if target_at_first:
            Y_df = df.iloc[:, :n_labels].astype(int)
            X_df = df.iloc[:, n_labels:]
        else:
            X_df = df.iloc[:, :-n_labels]
            Y_df = df.iloc[:, -n_labels:].astype(int)

        self.X, self.Y = self._to_numpy(X_df, Y_df)

    def _to_numpy(self, X, Y):
        """Convert DataFrames to numpy arrays (no scaling applied here)."""
        X_np = X.to_numpy().astype(float)
        Y_np = Y.to_numpy()
        print(f"Features shape: {X_np.shape}")
        print(f"Labels shape:   {Y_np.shape}")
        return X_np, Y_np

    def _get_label_count(self):
        """Return the number of labels for the dataset."""
        if self.dataset_name not in _LABEL_COUNTS:
            raise Exception(f"Dataset '{self.dataset_name}' is not supported. "
                            f"Known datasets: {list(_LABEL_COUNTS)}")
        return _LABEL_COUNTS[self.dataset_name]

    def get_cross_validation_folds(self, n_splits=5, shuffle=True, random_state=None):
        """Yield (train_index, test_index) tuples for k-fold cross-validation.

        Args:
            n_splits: Number of folds.
            shuffle: Whether to shuffle before splitting.
            random_state: Random seed for reproducibility.
        """
        skf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        yield from skf.split(self.X, self.Y)
