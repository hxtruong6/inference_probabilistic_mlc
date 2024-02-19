from collections import defaultdict
import os
import time
from uuid import uuid4
from src.arff_dataset import MultiLabelArffDataset
from src.evaluation_metrics import EvaluationMetrics

import numpy as np

from src.probability_classifier_chains import ProbabilisticClassifierChainCustom

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
                score = f"{metric_func(Y_pred, Y_true, **options):.5f}"
            else:
                score = f"{metric_func(Y_pred, Y_true):.5f}"

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


def training_model(model, X_train, Y_train, predicted_store_key=uuid4()):
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

        print("📊 Calculating metrics...")
        score_metrics = calculate_metrics(Y_test, Y_pred, metric_funcs)

        loss_score_by_predict_func.append(
            {"predict_name": predict_func["name"], "score_metrics": score_metrics}
        )

    return loss_score_by_predict_func


def prepare_model_to_evaluate():
    """Prepare a list of models for evaluation."""
    base_estimators = [
        LogisticRegression(random_state=SEED, max_iter=10000),
        # SGDClassifier(
        #     loss="log_loss", random_state=SEED
        # ),
        # RandomForestClassifier(random_state=SEED),
        # AdaBoostClassifier(random_state=SEED),
    ]
    return [ProbabilisticClassifierChainCustom(model) for model in base_estimators]


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

    output_csv = os.path.join(absolute_dir, "result/evaluation_results.csv")
    print(f"📂 Dataset folder path: {folder_path}")

    dataset_names = [
        "emotions",
        # "Water-quality",
        # "yeast",
        # "scene",
        # "VirusGO_sparse"
        "CHD_49",
    ]
    # -----------------  MAIN -----------------
    # func is same name of the predict function in ProbabilisticClassifierChainCustom
    predict_functions = [
        {"name": "Predict Hamming Loss", "func": "predict_Hamming"},
        {"name": "Predict Subset", "func": "predict_Subset"},
        # {"name": "Predict Pre", "func": "predict_Pre"},
        # {"name": "Predict Neg", "func": "predict_Neg"},
        # {"name": "Predict Recall", "func": "predict_Recall"},
        # {"name": "Predict Mar", "func": "predict_Mar"},
        # {"name": "Predict Fmeasure", "func": "predict_Fmeasure"},
        # {"name": "Predict Inf", "func": "predict_Inf"},
    ]

    metric_functions = [
        {"name": "Hamming Loss", "func": EvaluationMetrics.hamming_loss},
        {"name": "Subset Accuracy", "func": EvaluationMetrics.subset_accuracy},
        # {
        #     "name": "Precision Score",
        #     "func": EvaluationMetrics.precision_score,
        # },
        # {
        #     "name": "Negative Predictive Value",
        #     "func": EvaluationMetrics.negative_predictive_value,
        # },
        # {
        #     "name": "Recall Score",  #
        #     "func": EvaluationMetrics.recall_score,
        # },
        # {
        #     "name": "Markedness",
        #     "func": EvaluationMetrics.f_markedness,
        # },
        # {
        #     "name": "F-beta Score",
        #     "func": EvaluationMetrics.f_beta_score,
        # },
        # TODO:add informedness
    ]

    # Create a DataFrame to store the evaluation results
    data = {
        "Dataset": [],
        "Model": [],
        "Predict Function of Model": [],
        "Metric Function": [],
        "Score": [],
    }

    # Create a structure to store scores per dataset and metric
    scores_by_dataset = defaultdict(lambda: defaultdict(list))

    # Iterate over datasets
    for dataset_handler in read_datasets_from_folder(folder_path, dataset_names):
        print(f"\n🐳 Evaluating on {dataset_handler.dataset_name} dataset...")

        # Use cross-validation for more robust evaluation
        kfold_count = 0
        for train_index, test_index in dataset_handler.get_cross_validation_folds(
            n_splits=5, random_state=SEED
        ):
            kfold_count += 1
            print(f"\n🔁 Cross-validation fold {kfold_count}...")
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
                model = training_model(
                    model,
                    X_train,
                    y_train,
                    predicted_store_key=f"{dataset_handler.dataset_name}_kfold_{kfold_count}",
                )
                loss_score_by_predict_func = evaluate_model(
                    model, X_test, y_test, predict_functions, metric_functions
                )

                # Collect and append evaluation results to the DataFrame
                print("-" * 10)
                for result in loss_score_by_predict_func:
                    print("❄️ Metric: ", result["predict_name"])

                    for score_metric in result["score_metrics"]:
                        data["Dataset"].append(dataset_handler.dataset_name)
                        data["Model"].append(model.base_estimator.__class__.__name__)
                        data["Predict Function of Model"].append(result["predict_name"])

                        data["Metric Function"].append(score_metric["Metric Name"])
                        data["Score"].append(score_metric["Score"])

                        # Store the scores in a dictionary for later use
                        scores_by_dataset[dataset_handler.dataset_name][
                            score_metric["Metric Name"]
                        ].append(float(score_metric["Score"]))

    result_df = pd.DataFrame(data)

    average_scores = {}
    for dataset, metric_scores in scores_by_dataset.items():
        average_scores[dataset] = {}
        for metric, scores in metric_scores.items():
            average_scores[dataset][metric] = sum(float(s) for s in scores) / len(
                scores
            )

    print("Average Evaluation Scores:")
    for dataset, scores in average_scores.items():
        print(f"Dataset: {dataset}")
        for metric, score in scores.items():
            print(f"  {metric}: {score:.5f}")

    # Save the evaluation results to a CSV file
    result_df.to_csv(output_csv, index=False)
    print(f"\n✅Evaluation results saved to {output_csv}")


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
