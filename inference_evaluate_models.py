from collections import defaultdict
import os
import time
from uuid import uuid4

from joblib import Parallel, delayed
from src.arff_dataset import MultiLabelArffDataset
from src.evaluation_metrics import EvaluationMetrics

import numpy as np

from src.probability_classifier_chains import ProbabilisticClassifierChainCustom
from src.utils import add_key_if_missing, save_crosstab, save_result_df

print(f"Numpy version: {np.__version__}")

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier

import scipy.io.arff as arff


# Define a constant for seed
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
    """Read datasets from a folder and yield train and test sets."""
    if not os.path.isdir(folder_path):
        raise Exception(f"Folder path is not valid - {folder_path}")

    for filename in dataset_names:
        if filename in DATASET_WHOLE_FILES:
            yield MultiLabelArffDataset(
                path=os.path.join(folder_path, f"{filename}.arff"),
                dataset_name=filename,
                target_at_first=(filename in DATASET_WHOLE_FILES_TARGET_AT_FIRST),
            )
        else:
            raise Exception("Dataset name is not supported")


def calculate_metrics(Y_true, Y_pred, metric_funcs):
    """Calculate metrics based on the provided metric functions."""
    score_metrics = []
    for metric in metric_funcs:
        metric_name, metric_func, options = (
            metric["name"],
            metric["func"],
            metric.get("options", {}),
        )
        try:
            if options:
                score = f"{metric_func(Y_true, Y_pred, **options):.5f}"
            else:
                score = f"{metric_func(Y_true, Y_pred):.5f}"

            score_metrics.append(
                {
                    "Metric Name": metric_name,
                    "Metric Function": metric_func.__name__,
                    "Score": score,
                }
            )
        except Exception as e:
            print(f"Error calculating {metric_name} - {e}")

    return score_metrics


def training_model(model, X_train, Y_train, predicted_store_key=uuid4().__str__):
    """Train the specified model on the training data."""
    try:
        start_time = time.time()
        print(f"⏳ Training {model.base_estimator.__class__.__name__} model...")
        model.set_store_key(predicted_store_key)
        model.fit(X_train, Y_train)

        print(f"🤿 Elapsed training time: {time.time() - start_time:.5f} seconds")
        return model
    except Exception as e:
        print(f"Error training model - {e}")


def evaluate_model(model, X_test, Y_test, predict_funcs, metric_funcs):
    """Evaluate the model using various prediction and metric functions."""
    print(f"{'-' * 50}\nPredicting {model.base_estimator.__class__.__name__} model...")

    loss_score_by_predict_func = []
    for predict_func in predict_funcs:
        start_time = time.time()

        if predict_func["func"] == "predict":
            Y_pred, _, _ = model.predict(X_test)
        else:
            Y_pred = getattr(model, predict_func["func"])(X_test)

        elapsed_time = time.time() - start_time
        print(
            f"🤿 Elapsed predict time: {elapsed_time:.5f} seconds - [{predict_func['name']}]"
        )

        # print("📊 Calculating metrics...")
        score_metrics = calculate_metrics(Y_test, Y_pred, metric_funcs)

        loss_score_by_predict_func.append(
            {"predict_name": predict_func["name"], "score_metrics": score_metrics}
        )

    return loss_score_by_predict_func


def prepare_model_to_evaluate():
    """Prepare a list of models for evaluation."""
    base_estimators = [
        LogisticRegression(random_state=SEED, max_iter=10000),
        # SGDClassifier(loss="log_loss", random_state=SEED, max_iter=10000),
        # RandomForestClassifier(random_state=SEED),
        AdaBoostClassifier(random_state=SEED),
    ]
    return [ProbabilisticClassifierChainCustom(model) for model in base_estimators]


def evaluate_kfold(
    dataset_handler,
    evaluated_models,
    predict_functions,
    metric_functions,
    train_index,
    test_index,
    kfold_index,
):
    """Evaluates all models on a single k-fold split."""
    print(
        f"\n🔁 Cross-validation fold: {kfold_index} ..."
    )  # Replace with fold indicator
    dataset_name = dataset_handler.dataset_name
    fold_results = defaultdict(dict)
    fold_results[dataset_name] = {}

    X_train, X_test = (
        dataset_handler.X[train_index],
        dataset_handler.X[test_index],
    )
    y_train, y_test = (
        dataset_handler.Y[train_index],
        dataset_handler.Y[test_index],
    )

    # For each dataset, iterate over the models and perform evaluation
    for model in evaluated_models:
        model_name = model.base_estimator.__class__.__name__
        if model_name not in fold_results[dataset_name]:
            fold_results[dataset_name][model_name] = {}

        model = training_model(
            model,
            X_train,
            y_train,
            predicted_store_key=f"{dataset_handler.dataset_name}_{model_name}_kfold_{kfold_index}",
        )
        loss_score_by_predict_func = evaluate_model(
            model, X_test, y_test, predict_functions, metric_functions
        )

        # Collect and append evaluation results to the DataFrame
        # print("-" * 10)
        for result in loss_score_by_predict_func:
            # print("❄️ Metric: ", result["predict_name"])

            predict_func_name = result["predict_name"]
            if predict_func_name not in fold_results[dataset_name][model_name]:
                fold_results[dataset_name][model_name][predict_func_name] = {}

            for score_metric in result["score_metrics"]:

                loss_func_name = score_metric["Metric Name"]
                if (
                    loss_func_name
                    not in fold_results[dataset_name][model_name][predict_func_name]
                ):
                    fold_results[dataset_name][model_name][predict_func_name][
                        loss_func_name
                    ] = []

                fold_results[dataset_name][model_name][predict_func_name][
                    loss_func_name
                ] = score_metric["Score"]

    return fold_results


def main():
    """Main function to orchestrate the evaluation process."""
    # Define the list of models you want to evaluate
    evaluated_models = prepare_model_to_evaluate()

    absolute_dir = "/".join(os.path.abspath(__file__).split("/")[:-1])
    # Define the folder path containing JSON datasets and the output CSV file name
    folder_path = os.path.join(absolute_dir, "datasets")
    # check if not exist, create a result folder
    if not os.path.exists(os.path.join(absolute_dir, "result")):
        os.makedirs(os.path.join(absolute_dir, "result"))

    print(f"📂 Dataset folder path: {folder_path}")

    dataset_names = [
        # "emotions",
        "Water-quality",
        # "scene",
        # "VirusGO_sparse",
        # "CHD_49",
        # "yeast",
    ]
    # -----------------  MAIN -----------------
    # func is same name of the predict function in ProbabilisticClassifierChainCustom
    predict_functions = [
        {"name": "Predict Hamming Loss", "func": "predict_Hamming"},
        {"name": "Predict Subset", "func": "predict_Subset"},
        {"name": "Predict Pre", "func": "predict_Precision"},
        {"name": "Predict Neg", "func": "predict_Neg"},
        {"name": "Predict Recall", "func": "predict_Recall"},
        {"name": "Predict Mar", "func": "predict_Mar"},
        {"name": "Predict Fmeasure", "func": "predict_Fmeasure"},
        # {"name": "Predict Inf", "func": "predict_Inf"},
    ]

    metric_functions = [
        {"name": "Hamming Loss", "func": EvaluationMetrics.hamming_loss},
        {"name": "Subset Accuracy", "func": EvaluationMetrics.subset_accuracy},
        {
            "name": "Precision Score",
            "func": EvaluationMetrics.precision_score,
        },
        {
            "name": "Negative Predictive Value",
            "func": EvaluationMetrics.negative_predictive_value,
        },
        {
            "name": "Recall Score",  #
            "func": EvaluationMetrics.recall_score,
        },
        {
            "name": "Markedness",
            "func": EvaluationMetrics.f_markedness,
        },
        {
            "name": "F-beta Score",
            "func": EvaluationMetrics.f_beta_score,
        },
        # TODO: add informedness
    ]

    # Create a DataFrame to store the evaluation results
    dataset_results = {}

    try:

        # Iterate over datasets
        for dataset_handler in read_datasets_from_folder(folder_path, dataset_names):
            print(f"\n🐳 Evaluating on {dataset_handler.dataset_name} dataset...")

            t1 = time.time()

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
                # Use cross-validation for more robust evaluation
                for kfold_index, (train_index, test_index) in enumerate(
                    dataset_handler.get_cross_validation_folds(
                        n_splits=KFOLD_SPLIT_NUMBER, random_state=SEED
                    )
                )
            )

            dataset_results[dataset_handler.dataset_name] = {}
            for fold_result in job_results:
                for model in evaluated_models:
                    for predict_func in predict_functions:
                        for metric_func in metric_functions:
                            add_key_if_missing(
                                dataset_results,
                                dataset_handler.dataset_name,
                                model.base_estimator.__class__.__name__,
                                predict_func["name"],
                                metric_func["name"],
                                fold_result[dataset_handler.dataset_name][
                                    model.base_estimator.__class__.__name__
                                ][predict_func["name"]][metric_func["name"]],
                            )

            output_csv = os.path.join(
                absolute_dir, "result", f"result_{dataset_handler.dataset_name}.csv"
            )
            result_df = save_result_df(dataset_results, output_csv)
            save_crosstab(result_df, output_csv)

            print(
                f"\n ==== 🦈 Dataset evaluation time: {time.time() - t1:.5f} seconds \n"
            )
    except Exception as e:
        print(f"\n Dataset: {dataset_names} \n Error: {e}")
        raise e


if __name__ == "__main__":
    """IDEA:
    - Datasets
        - Models
            - Predict Functions
                - Metric Functions -> Score


    Add all to a DataFrame and save to CSV file
    with format: [
        {
            "Dataset": "emotions",
            "Model": "ProbabilisticClassifierChainCustom",
            "Predict Function of Model": "Predict",
            "Metric Function": "Hamming Loss",
            "Score": 0.0
        },
        {
            "Dataset": "emotions",
            "Model": "ProbabilisticClassifierChainCustom",
            "Predict Function of Model": "Predict Hamming Loss",
            "Metric Function": "Hamming Loss",
            "Score": 0.0
        },
        ...
    ]

    """
    main()
