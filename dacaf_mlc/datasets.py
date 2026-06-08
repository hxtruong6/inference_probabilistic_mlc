"""Dataset registry and loading.

A single ``DatasetSpec`` registry is the source of truth for every dataset:
its label count, file orientation, loader, and whether it belongs to the
default sweep. To add a dataset, append one ``DatasetSpec`` to
``DATASET_SPECS`` (and, for a new source kind, write a small loader function);
nothing else needs editing.

Tabular datasets are MULAN ARFF files under ``datasets/``. Image datasets use
pre-extracted ChestX-ray feature vectors (``.npy``); see
``dacaf_mlc/chest_xray_dataset/Readme.md`` to regenerate those.
"""
import os
from collections.abc import Callable
from dataclasses import dataclass

from dacaf_mlc.config import BASE_DIR
from dacaf_mlc.arff_dataset import MultiLabelArffDataset
from dacaf_mlc.chest_xray_dataset.chest_xray_utils import load_df_features_from_npy


def _load_arff(spec, folder_path):
    """Load a MULAN ARFF shipped as a single whole file under ``folder_path``."""
    return MultiLabelArffDataset(
        dataset_name=spec.name,
        path=os.path.join(folder_path, f"{spec.name}.arff"),
        target_at_first=spec.target_at_first,
        n_labels=spec.n_labels,
    )


def _load_nih_features(spec, folder_path):
    """Load pre-extracted ChestX-ray feature vectors (``.npy``).

    The backbone (densenet/resnet/resnetae) is the suffix after ``__`` in the
    dataset name, e.g. ``chest_xray_nih__densenet``.
    """
    feature_type = spec.name.split("__")[-1]
    df_feats, df_labels = load_df_features_from_npy(
        features_filename=os.path.join(
            BASE_DIR, "datasets", f"nih_feature_vectors_{feature_type}.npy"
        ),
    )
    return MultiLabelArffDataset(dataset_name=spec.name, X=df_feats, Y=df_labels)


@dataclass(frozen=True)
class DatasetSpec:
    """Everything needed to load one dataset.

    - ``name``           : registry key and (for ARFF) the ``<name>.arff`` stem.
    - ``n_labels``       : number of label columns ``L``.
    - ``loader``         : ``loader(spec, folder_path) -> MultiLabelArffDataset``.
    - ``target_at_first``: ARFF only — labels are the leading columns, not trailing.
    - ``in_default_sweep``: include in ``DEFAULT_DATASET_NAMES`` / the no-arg sweep.
    - ``note``           : free-text hint shown in docs (e.g. ``"L=6"``).
    """

    name: str
    n_labels: int
    loader: Callable
    target_at_first: bool = False
    in_default_sweep: bool = True
    note: str = ""


# The paper datasets, in sweep order. The chest_xray_nih__* entries require
# pre-extracted features at datasets/nih_feature_vectors_{densenet,resnet,resnetae}.npy
# (see dacaf_mlc/chest_xray_dataset/Readme.md).
DATASET_SPECS = (
    DatasetSpec("emotions", 6, _load_arff, note="L=6"),
    DatasetSpec("scene", 6, _load_arff, note="L=6"),
    DatasetSpec("CHD_49", 6, _load_arff, target_at_first=True, note="L=6"),
    DatasetSpec("Water-quality", 14, _load_arff, target_at_first=True, note="L=14"),
    DatasetSpec("yeast", 14, _load_arff, target_at_first=True, note="L=14"),
    DatasetSpec("chest_xray_nih__densenet", 14, _load_nih_features, note="L=14, N≈112k"),
    DatasetSpec("chest_xray_nih__resnet", 14, _load_nih_features, note="L=14, N≈112k"),
    DatasetSpec("chest_xray_nih__resnetae", 14, _load_nih_features, note="L=14, N≈112k"),
)

DATASET_REGISTRY = {spec.name: spec for spec in DATASET_SPECS}

# Names included in the default sweep, in order (used as argparse --dataset
# choices and by the no-arg full sweep).
DEFAULT_DATASET_NAMES = [spec.name for spec in DATASET_SPECS if spec.in_default_sweep]


def read_datasets_from_folder(folder_path, dataset_names):
    """Yield a MultiLabelArffDataset for each requested dataset name."""
    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder path is not valid: {folder_path}")

    for name in dataset_names:
        spec = DATASET_REGISTRY.get(name)
        if spec is None:
            raise ValueError(
                f"Dataset '{name}' is not supported. Known datasets: {list(DATASET_REGISTRY)}"
            )
        yield spec.loader(spec, folder_path)
