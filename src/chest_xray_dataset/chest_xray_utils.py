# add this package src to sys.path
import numpy as np


def save_features_to_npy(extract_features, filename="feature_vectors.npy"):
    features = np.array([item["features"].numpy() for item in extract_features])
    labels = np.array([item["labels"].numpy() for item in extract_features])
    np.save(filename, features)
    labels_filename = filename.split(".")[0] + "__labels.npy"
    np.save(labels_filename, labels)


def load_features_from_npy(
    features_filename="feature_vectors.npy",
    labels_filename="feature_vectors__labels.npy",
):
    features = np.load(features_filename)
    labels = np.load(labels_filename)
    return features, labels
