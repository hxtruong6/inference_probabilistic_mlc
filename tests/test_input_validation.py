"""Input-validation contracts: specific exception types, not bare Exception."""
import numpy as np
import pytest

from dacaf_mlc.evaluation_metrics import EvaluationMetrics as EM
from dacaf_mlc.datasets import read_datasets_from_folder


def test_metric_shape_mismatch_raises_valueerror():
    Y_true = np.array([[1, 0, 1]])
    Y_pred = np.array([[1, 0]])
    with pytest.raises(ValueError):
        EM.hamming_accuracy(Y_true, Y_pred)


def test_unknown_dataset_raises_valueerror(tmp_path):
    with pytest.raises(ValueError):
        list(read_datasets_from_folder(str(tmp_path), ["not_a_dataset"]))


def test_missing_folder_raises_valueerror():
    with pytest.raises(ValueError):
        list(read_datasets_from_folder("/nonexistent/folder/xyz", ["emotions"]))
