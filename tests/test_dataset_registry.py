"""The dataset registry is the single source of truth for dataset config.

These pin the values that were previously scattered across ``_LABEL_COUNTS``,
``DATASET_WHOLE_FILES``, ``DATASET_WHOLE_FILES_TARGET_AT_FIRST``, and
``DEFAULT_DATASET_NAMES`` so the registry refactor is provably
behaviour-preserving: same datasets, same label counts, same orientation,
same default-sweep order.
"""
import pytest

from dacaf_mlc.datasets import (
    DATASET_REGISTRY,
    DEFAULT_DATASET_NAMES,
    read_datasets_from_folder,
)


def test_default_sweep_order_unchanged():
    # The argparse --dataset choices and the no-arg sweep depend on this order.
    assert DEFAULT_DATASET_NAMES == [
        "emotions",
        "scene",
        "CHD_49",
        "Water-quality",
        "yeast",
        "chest_xray_nih__densenet",
        "chest_xray_nih__resnet",
        "chest_xray_nih__resnetae",
    ]


def test_label_counts_unchanged():
    expected = {
        "emotions": 6,
        "scene": 6,
        "CHD_49": 6,
        "Water-quality": 14,
        "yeast": 14,
        "chest_xray_nih__densenet": 14,
        "chest_xray_nih__resnet": 14,
        "chest_xray_nih__resnetae": 14,
    }
    assert {name: DATASET_REGISTRY[name].n_labels for name in expected} == expected


def test_target_at_first_unchanged():
    # Exactly these ARFFs store the labels in the leading columns.
    target_first = {name for name, spec in DATASET_REGISTRY.items() if spec.target_at_first}
    assert target_first == {"Water-quality", "CHD_49", "yeast"}


def test_every_default_name_is_registered():
    for name in DEFAULT_DATASET_NAMES:
        assert name in DATASET_REGISTRY


def test_unknown_dataset_raises(tmp_path):
    with pytest.raises(ValueError, match="not_a_dataset"):
        list(read_datasets_from_folder(str(tmp_path), ["not_a_dataset"]))


def test_bad_folder_raises():
    with pytest.raises(ValueError):
        list(read_datasets_from_folder("/nonexistent/folder/xyz", ["emotions"]))
