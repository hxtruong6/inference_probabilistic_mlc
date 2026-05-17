from collections import defaultdict
import os
import time
from uuid import uuid4
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from src.arff_dataset import MultiLabelArffDataset
from src.chest_xray_dataset.chest_xray_utils import load_df_features_from_npy
from src.evaluation_metrics import EvaluationMetrics
import numpy as np
from src.probability_classifier_chains import ProbabilisticClassifierChainCustom
from src.utils import add_key_if_missing, save_crosstab, save_result_df
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier


print(f"Numpy version: {np.__version__}")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Base project folder: {BASE_DIR}")

SEED = 6
DATASET_WHOLE_FILES = [
    "VirusGO_sparse",
    "Water-quality",
    "CHD_49",
    "emotions",
    "scene",
    "yeast",
]
DATASET_WHOLE_FILES_TARGET_AT_FIRST = ["Water-quality", "CHD_49", "yeast"]
KFOLD_SPLIT_NUMBER = 10


def read_datasets_from_folder(folder_path, dataset_names):
    """Read datasets from a folder and yield MultiLabelArffDataset instances."""
    if not os.path.isdir(folder_path):
        raise Exception(f"Folder path is not valid: {folder_path}")

    for filename in dataset_names:
        if "chest_xray_nih" in filename:
            feature_type = filename.split("__")[-1]
            df_feats, df_labels = load_df_features_from_npy(
                features_filename=os.path.join(BASE_DIR, "datasets", f"nih_feature_vectors_{feature_type}.npy"),
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


def calculate_metrics(Y_true, Y_pred, metric_funcs):
    """Calculate all metrics for a given (Y_true, Y_pred) pair."""
    scores = []
    for metric in metric_funcs:
        name = metric["name"]
        func = metric["func"]
        opts = metric.get("options", {})
        try:
            value = func(Y_true, Y_pred, **opts) if opts else func(Y_true, Y_pred)
            scores.append({"Metric Name": name, "Metric Function": func.__name__, "Score": f"{value:.5f}"})
        except Exception as e:
            print(f"Error calculating {name}: {e}")
    return scores


def training_model(model, X_train, Y_train, cache_key=None):
    """Train the model and assign a cache key for prediction reuse."""
    if cache_key is None:
        cache_key = uuid4().hex
    try:
        start = time.time()
        print(f"Training {model.base_estimator.__class__.__name__} ...")
        model.set_cache_key(cache_key)
        model.fit(X_train, Y_train)
        print(f"Training time: {time.time() - start:.3f}s")
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        raise


def evaluate_model(model, X_test, Y_test, predict_funcs, metric_funcs):
    """Run all predict_* functions and compute all metrics."""
    results = []
    for pf in predict_funcs:
        start = time.time()
        Y_pred = getattr(model, pf["func"])(X_test)
        print(f"Predict time: {time.time() - start:.3f}s [{pf['name']}]")
        scores = calculate_metrics(Y_test, Y_pred, metric_funcs)
        results.append({"predict_name": pf["name"], "score_metrics": scores})
    return results


def prepare_model_to_evaluate():
    """Return the list of PCC models to evaluate."""
    base_estimators = [
        LogisticRegression(random_state=SEED, max_iter=5_000_000),
        # SGDClassifier(loss="log_loss", random_state=SEED, max_iter=10_000),
        # RandomForestClassifier(random_state=SEED),
        # AdaBoostClassifier(random_state=SEED),
    ]
    return [ProbabilisticClassifierChainCustom(m) for m in base_estimators]


def evaluate_kfold(
    dataset_handler,
    evaluated_models,
    predict_functions,
    metric_functions,
    train_index,
    test_index,
    kfold_index,
):
    """Evaluate all models on one k-fold split with proper per-fold scaling."""
    print(f"\nFold {kfold_index} ...")
    dataset_name = dataset_handler.dataset_name

    X_train_raw = dataset_handler.X[train_index]
    X_test_raw  = dataset_handler.X[test_index]
    y_train = dataset_handler.Y[train_index]
    y_test  = dataset_handler.Y[test_index]

    # Fit scaler on training data only — avoids data leakage into test fold.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    fold_results = {dataset_name: {}}
    for model in evaluated_models:
        model_name = model.base_estimator.__class__.__name__
        fold_results[dataset_name][model_name] = {}

        model = training_model(
            model,
            X_train,
            y_train,
            cache_key=f"{dataset_name}_{model_name}_fold{kfold_index}",
        )
        results = evaluate_model(model, X_test, y_test, predict_functions, metric_functions)

        for result in results:
            pf_name = result["predict_name"]
            fold_results[dataset_name][model_name][pf_name] = {}
            for sm in result["score_metrics"]:
                fold_results[dataset_name][model_name][pf_name][sm["Metric Name"]] = sm["Score"]

    return fold_results


def main():
    """Orchestrate the full evaluation pipeline."""
    evaluated_models = prepare_model_to_evaluate()
    folder_path = os.path.join(BASE_DIR, "datasets")
    result_dir = os.path.join(BASE_DIR, "result")
    os.makedirs(result_dir, exist_ok=True)

    print(f"Dataset folder: {folder_path}")

    # Uncomment the datasets to evaluate:
    dataset_names = [
        "emotions",
        "scene",
        "CHD_49",
        # "VirusGO_sparse",  # sparse ARFF — scipy.io.arff cannot parse

        # "Water-quality",
        # "yeast",
        # "chest_xray_nih__densenet",
        # "chest_xray_nih__resnet",
        # "chest_xray_nih__resnetae",
    ]

    predict_functions = [
        {"name": "Predict Hamming",      "func": "predict_hamming"},
        {"name": "Predict Subset",       "func": "predict_subset"},
        {"name": "Predict Precision",    "func": "predict_precision"},
        {"name": "Predict NPV",          "func": "predict_npv"},
        {"name": "Predict Recall",       "func": "predict_recall"},
        {"name": "Predict Markedness",   "func": "predict_markedness"},
        {"name": "Predict Fmeasure",     "func": "predict_fmeasure"},
        {"name": "Predict Informedness", "func": "predict_informedness"},
    ]

    metric_functions = [
        {"name": "Hamming Accuracy",          "func": EvaluationMetrics.hamming_accuracy},
        {"name": "Subset Accuracy",            "func": EvaluationMetrics.subset_accuracy},
        {"name": "Precision Score",            "func": EvaluationMetrics.precision_score},
        {"name": "Negative Predictive Value",  "func": EvaluationMetrics.negative_predictive_value},
        {"name": "Recall Score",               "func": EvaluationMetrics.recall_score},
        {"name": "Markedness",                 "func": EvaluationMetrics.markedness},
        {"name": "Fmeasure Score",             "func": EvaluationMetrics.f_beta},
        {"name": "Informedness",               "func": EvaluationMetrics.informedness},
    ]

    dataset_results = {}

    for dataset_handler in read_datasets_from_folder(folder_path, dataset_names):
        print(f"\nEvaluating: {dataset_handler.dataset_name}")
        t0 = time.time()

        job_results = Parallel(n_jobs=-1)(
            delayed(evaluate_kfold)(
                dataset_handler,
                evaluated_models,
                predict_functions,
                metric_functions,
                train_index,
                test_index,
                kfold_index,
            )
            for kfold_index, (train_index, test_index) in enumerate(
                dataset_handler.get_cross_validation_folds(
                    n_splits=KFOLD_SPLIT_NUMBER, random_state=SEED
                )
            )
        )

        dataset_results[dataset_handler.dataset_name] = {}
        for fold_result in job_results:
            if fold_result is None:
                continue
            for model in evaluated_models:
                for pf in predict_functions:
                    for mf in metric_functions:
                        add_key_if_missing(
                            dataset_results,
                            dataset_handler.dataset_name,
                            model.base_estimator.__class__.__name__,
                            pf["name"],
                            mf["name"],
                            fold_result[dataset_handler.dataset_name][
                                model.base_estimator.__class__.__name__
                            ][pf["name"]][mf["name"]],
                        )

        output_csv = os.path.join(result_dir, f"result_{dataset_handler.dataset_name}.csv")
        result_df = save_result_df(dataset_results, output_csv)
        save_crosstab(result_df, output_csv)
        print(f"Dataset evaluation time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
