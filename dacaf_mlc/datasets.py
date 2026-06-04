"""Dataset registry and loading.

Tabular datasets are MULAN ARFF files under ``datasets/``. Image datasets use
pre-extracted ChestX-ray feature vectors (``.npy``); see
``dacaf_mlc/chest_xray_dataset/Readme.md`` to regenerate those.
"""
import os

from dacaf_mlc.config import BASE_DIR
from dacaf_mlc.arff_dataset import MultiLabelArffDataset
from dacaf_mlc.chest_xray_dataset.chest_xray_utils import load_df_features_from_npy

# ARFF datasets shipped as a single whole file (no separate train/test split).
DATASET_WHOLE_FILES = [
    "VirusGO_sparse",
    "Water-quality",
    "CHD_49",
    "emotions",
    "scene",
    "yeast",
    "flags",
    "PlantPseAAC",
]
# Datasets whose ARFF stores the labels in the FIRST columns rather than the last.
DATASET_WHOLE_FILES_TARGET_AT_FIRST = ["Water-quality", "CHD_49", "yeast", "PlantPseAAC"]

# Full sweep set. The chest_xray_nih__* entries require pre-extracted features at
# datasets/nih_feature_vectors_{densenet,resnet,resnetae}.npy.
DEFAULT_DATASET_NAMES = [
    "flags",                       # L=7
    "emotions",                    # L=6
    "scene",                       # L=6
    "CHD_49",                      # L=6
    "VirusGO_sparse",              # L=6  (sparse ARFF)
    "PlantPseAAC",                 # L=12 (sparse ARFF)
    "Water-quality",               # L=14
    "yeast",                       # L=14
    "chest_xray_nih__densenet",    # L=14, N≈112k
    "chest_xray_nih__resnet",      # L=14, N≈112k
    "chest_xray_nih__resnetae",    # L=14, N≈112k
]


def read_datasets_from_folder(folder_path, dataset_names):
    """Yield a MultiLabelArffDataset for each requested dataset name."""
    if not os.path.isdir(folder_path):
        raise Exception(f"Folder path is not valid: {folder_path}")

    for filename in dataset_names:
        if "chest_xray_nih" in filename:
            feature_type = filename.split("__")[-1]
            df_feats, df_labels = load_df_features_from_npy(
                features_filename=os.path.join(
                    BASE_DIR, "datasets", f"nih_feature_vectors_{feature_type}.npy"
                ),
            )
            yield MultiLabelArffDataset(dataset_name=filename, X=df_feats, Y=df_labels)
        elif filename in DATASET_WHOLE_FILES:
            yield MultiLabelArffDataset(
                dataset_name=filename,
                path=os.path.join(folder_path, f"{filename}.arff"),
                target_at_first=(filename in DATASET_WHOLE_FILES_TARGET_AT_FIRST),
            )
        else:
            raise Exception(f"Dataset '{filename}' is not supported.")
