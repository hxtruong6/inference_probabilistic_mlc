import logging

import pandas as pd
import scipy.io.arff as arff
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


class MultiLabelArffDataset:
    """Class to handle Mulan datasets stored in ARFF files.

    Stores raw (unscaled) feature arrays. Callers are responsible for
    applying StandardScaler per fold AFTER train/test split to avoid
    data leakage.

    Two ways to construct:
    - From an ARFF file: pass ``path`` and ``n_labels`` (the label count comes
      from the dataset registry, see ``dacaf_mlc.datasets``).
    - From in-memory data: pass ``X`` and ``Y`` DataFrames; ``n_labels`` is then
      inferred from ``Y`` and not required.
    """

    def __init__(
        self,
        dataset_name,
        path=None,
        target_at_first=False,
        X=None,
        Y=None,
        n_labels=None,
    ):
        """Initialize the dataset handler."""
        self.dataset_name = dataset_name

        if X is not None and Y is not None and isinstance(X, pd.DataFrame) and isinstance(Y, pd.DataFrame):
            self.X, self.Y = self._to_numpy(X, Y)
            return

        if n_labels is None:
            raise ValueError(
                f"n_labels is required to load ARFF dataset '{dataset_name}'."
            )

        self.path = path
        data = arff.loadarff(self.path)
        df = pd.DataFrame(data[0])

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
        logger.info("Features shape: %s | Labels shape: %s", X_np.shape, Y_np.shape)
        return X_np, Y_np

    def get_cross_validation_folds(self, n_splits=5, shuffle=True, random_state=None):
        """Yield (train_index, test_index) tuples for k-fold cross-validation.

        Args:
            n_splits: Number of folds.
            shuffle: Whether to shuffle before splitting.
            random_state: Random seed for reproducibility.
        """
        skf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        yield from skf.split(self.X, self.Y)
