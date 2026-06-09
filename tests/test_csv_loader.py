"""The built-in CSV loader and the DatasetSpec fields that configure it.

CSV datasets are registered exactly like ARFF ones, but with ``_load_csv`` and
optionally a separator / explicit label-column names. These tests exercise the
loader directly via a hand-built spec (no global-registry pollution).
"""
import numpy as np
import pandas as pd
import pytest

from dacaf_mlc.datasets import DatasetSpec, _load_csv


def _write_csv(path, df, sep=","):
    df.to_csv(path, sep=sep, index=False)


def test_positional_labels_trailing(tmp_path):
    # 3 features then 2 trailing label columns.
    df = pd.DataFrame(
        {"f0": [0.1, 0.2], "f1": [1.0, 2.0], "f2": [3.0, 4.0], "l0": [0, 1], "l1": [1, 0]}
    )
    _write_csv(tmp_path / "data.csv", df)
    spec = DatasetSpec("data", 2, _load_csv)
    ds = _load_csv(spec, str(tmp_path))
    assert ds.X.shape == (2, 3)
    assert ds.Y.shape == (2, 2)
    np.testing.assert_array_equal(ds.Y, [[0, 1], [1, 0]])


def test_positional_labels_leading(tmp_path):
    # 2 leading label columns then 3 features.
    df = pd.DataFrame(
        {"l0": [0, 1], "l1": [1, 0], "f0": [0.1, 0.2], "f1": [1.0, 2.0], "f2": [3.0, 4.0]}
    )
    _write_csv(tmp_path / "data.csv", df)
    spec = DatasetSpec("data", 2, _load_csv, target_at_first=True)
    ds = _load_csv(spec, str(tmp_path))
    assert ds.X.shape == (2, 3)
    assert ds.Y.shape == (2, 2)
    np.testing.assert_array_equal(ds.Y, [[0, 1], [1, 0]])


def test_labels_by_name(tmp_path):
    # Labels are interleaved with features; selected by explicit name.
    df = pd.DataFrame(
        {"f0": [0.1, 0.2], "tag_a": [1, 0], "f1": [3.0, 4.0], "tag_b": [0, 1]}
    )
    _write_csv(tmp_path / "data.csv", df)
    spec = DatasetSpec("data", 2, _load_csv, label_columns=("tag_a", "tag_b"))
    ds = _load_csv(spec, str(tmp_path))
    assert ds.X.shape == (2, 2)          # f0, f1
    np.testing.assert_array_equal(ds.Y, [[1, 0], [0, 1]])


def test_custom_separator(tmp_path):
    df = pd.DataFrame({"f0": [0.1, 0.2], "l0": [0, 1]})
    _write_csv(tmp_path / "data.csv", df, sep=";")
    spec = DatasetSpec("data", 1, _load_csv, csv_sep=";")
    ds = _load_csv(spec, str(tmp_path))
    assert ds.X.shape == (2, 1)
    np.testing.assert_array_equal(ds.Y.ravel(), [0, 1])


def test_label_columns_length_must_match_n_labels():
    with pytest.raises(ValueError, match="n_labels"):
        DatasetSpec("data", 3, _load_csv, label_columns=("a", "b"))
