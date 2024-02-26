# add this package src to sys.path
import numpy as np
import pandas as pd


def save_features_to_npy(extract_features, filename):
    features = np.array([item["features"].numpy() for item in extract_features])
    labels = np.array([item["labels"].numpy() for item in extract_features])
    np.save(filename, features)
    labels_filename = filename.split(".")[0] + "__labels.npy"
    np.save(labels_filename, labels)


def load_features_from_npy(features_filename):
    features = np.load(features_filename)
    labels_filename = features_filename.split(".")[0] + "__labels.npy"
    labels = np.load(labels_filename)
    return features, labels


def load_df_features_from_npy(features_filename):
    """
    Loads features and labels from NumPy arrays, then converts them to DataFrames.

    Args:
      features_filename (str): Path to the NumPy array file containing features.
      labels_filename (str): Path to the NumPy array file containing labels.

    Returns:
      pandas.DataFrame: Features DataFrame
      pandas.DataFrame: Labels DataFrame
    """

    features = np.load(features_filename)

    labels_filename = features_filename.split(".")[0] + "__labels.npy"
    labels = np.load(labels_filename)

    # Create feature DataFrame
    column_names = [
        f"feature_{i}" for i in range(features.shape[1])
    ]  # Generate feature column names
    df_features = pd.DataFrame(features, columns=column_names)

    # Create label DataFrame
    label_names = [
        f"label_{i}" for i in range(labels.shape[1])
    ]  # Generate label column names
    df_labels = pd.DataFrame(labels, columns=label_names)

    return df_features, df_labels
