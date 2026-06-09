"""Runtime dataset registration for pip-installed users.

``register_dataset`` lets a user add a ``DatasetSpec`` at runtime (without
editing the package source), after which ``read_datasets_from_folder`` /
``run_single`` can load it. The registry is module-global, so each test cleans
up after itself.
"""
import pandas as pd
import pytest

from dacaf_mlc.datasets import (
    DATASET_REGISTRY,
    DEFAULT_DATASET_NAMES,
    DatasetSpec,
    _load_csv,
    read_datasets_from_folder,
    register_dataset,
)


@pytest.fixture
def csv_folder(tmp_path):
    df = pd.DataFrame(
        {"f0": [0.1, 0.2, 0.3, 0.4], "f1": [1.0, 2.0, 3.0, 4.0],
         "l0": [0, 1, 0, 1], "l1": [1, 0, 1, 0]}
    )
    df.to_csv(tmp_path / "mine.csv", index=False)
    return tmp_path


def test_register_then_load(csv_folder):
    spec = DatasetSpec("mine", 2, _load_csv, in_default_sweep=False)
    try:
        register_dataset(spec)
        assert "mine" in DATASET_REGISTRY
        (ds,) = list(read_datasets_from_folder(str(csv_folder), ["mine"]))
        assert ds.X.shape == (4, 2)
        assert ds.Y.shape == (4, 2)
    finally:
        DATASET_REGISTRY.pop("mine", None)


def test_register_adds_to_sweep_when_flagged():
    spec = DatasetSpec("sweepme", 2, _load_csv, in_default_sweep=True)
    try:
        register_dataset(spec)
        assert "sweepme" in DEFAULT_DATASET_NAMES
    finally:
        DATASET_REGISTRY.pop("sweepme", None)
        if "sweepme" in DEFAULT_DATASET_NAMES:
            DEFAULT_DATASET_NAMES.remove("sweepme")


def test_not_in_sweep_when_flag_false():
    spec = DatasetSpec("quiet", 2, _load_csv, in_default_sweep=False)
    try:
        register_dataset(spec)
        assert "quiet" not in DEFAULT_DATASET_NAMES
    finally:
        DATASET_REGISTRY.pop("quiet", None)


def test_duplicate_raises_without_override():
    spec = DatasetSpec("dup", 2, _load_csv, in_default_sweep=False)
    try:
        register_dataset(spec)
        with pytest.raises(ValueError, match="already registered"):
            register_dataset(spec)
        # override replaces the existing entry.
        register_dataset(DatasetSpec("dup", 3, _load_csv, in_default_sweep=False), override=True)
        assert DATASET_REGISTRY["dup"].n_labels == 3
    finally:
        DATASET_REGISTRY.pop("dup", None)
