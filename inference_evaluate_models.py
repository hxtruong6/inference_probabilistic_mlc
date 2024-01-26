import os
import time

import numpy as np

print(f"Numpy version: {np.__version__}")

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier

import scipy.io.arff as arff
from probability_classifier_chains import ProbabilisticClassifierChainCustom

# Define a constant for seed
SEED = 6
DATASET_WHOLE_FILES = ["VirusGO_sparse", "Water-quality", "CHD_49"]
DATASET_WHOLE_FILES_TARGET_AT_FIRST = ["Water-quality", "CHD_49"]


class EvaluationMetrics:
    """A class for various evaluation metrics."""

    @staticmethod
    def _check_dimensions(Y_true, Y_pred):
        """Check if the dimensions of Y_true and Y_pred are the same."""
        if Y_true.shape != Y_pred.shape:
            raise Exception("Y_true and Y_pred have different shapes")

    @staticmethod
    def get_loss(Y_true, Y_pred, loss_func):
        """Get the loss using the specified loss function."""
        EvaluationMetrics._check_dimensions(Y_true, Y_pred)
        return loss_func(Y_true, Y_pred)

    @staticmethod
    def hamming_loss(Y_true, Y_pred):
        """
        Calculate Hamming Loss for multilabel classification.

        Parameters:
        - y_true: NumPy array, true labels (2D array with shape [n_samples, n_labels]).
        - y_pred: NumPy array, predicted labels (2D array with shape [n_samples, n_labels]).

        Returns:
        - float: Hamming Loss.
        """
        EvaluationMetrics._check_dimensions(Y_true, Y_pred)

        # Calculate Hamming Loss
        loss = np.mean(np.not_equal(Y_true, Y_pred))
        return loss

    @staticmethod
    def precision_score(y_true, y_pred):
        """
        Calculate Precision score for binary or multiclass classification.

        Parameters:
        - y_true: NumPy array, true labels.
        - y_pred: NumPy array, predicted labels.

        Returns:
        - float: Precision score.
        """
        # Ensure y_true and y_pred have the same shape
        EvaluationMetrics._check_dimensions(y_true, y_pred)

        # Calculate True Positives and False Positives
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        # print(
        #     f"true_positives:\t{true_positives}\t\t| false_positives:\t{false_positives}"
        # )

        # Calculate Precision
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )

        return precision

    @staticmethod
    def recall_score(y_true, y_pred):
        """
        Calculate Recall score for binary or multiclass classification.

        Parameters:
        - y_true: NumPy array, true labels.
        - y_pred: NumPy array, predicted labels.

        Returns:
        - float: Recall score.
        """
        # Ensure y_true and y_pred have the same shape
        EvaluationMetrics._check_dimensions(y_true, y_pred)

        # Calculate True Positives and False Negatives
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))

        # Calculate Recall
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )

        return recall

    @staticmethod
    def subset_accuracy(y_true, y_pred):
        """
        Calculate Subset Accuracy for multilabel classification.

        Parameters:
        - y_true: NumPy array, true labels (2D array with shape [n_samples, n_labels]).
        - y_pred: NumPy array, predicted labels (2D array with shape [n_samples, n_labels]).

        Returns:
        - float: Subset Accuracy.
        """
        # Ensure y_true and y_pred have the same shape
        EvaluationMetrics._check_dimensions(y_true, y_pred)

        # Calculate subset accuracy
        correct_samples = np.sum(np.all(y_true == y_pred, axis=1))
        subset_accuracy_value = correct_samples / len(y_true)

        return subset_accuracy_value

    @staticmethod
    def negative_predictive_value(y_true, y_pred):
        """
        Calculate Negative Predictive Value for binary or multiclass classification.

        Parameters:
        - y_true: NumPy array, true labels.
        - y_pred: NumPy array, predicted labels.

        Returns:
        - float: Negative Predictive Value.
        """
        # Ensure y_true and y_pred have the same shape
        EvaluationMetrics._check_dimensions(y_true, y_pred)

        # Calculate True Negatives and False Negatives
        true_negatives = np.sum((y_true == 0) & (y_pred == 0))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))

        # Calculate Negative Predictive Value
        npv = (
            true_negatives / (true_negatives + false_negatives)
            if (true_negatives + false_negatives) > 0
            else 0
        )

        return npv

    @staticmethod
    def f1_score(y_true, y_pred):
        """
        Calculate F1 score for binary or multiclass classification.

        Parameters:
        - y_true: NumPy array, true labels.
        - y_pred: NumPy array, predicted labels.

        Returns:
        - float: F1 score.
        """
        # Ensure y_true and y_pred have the same shape
        EvaluationMetrics._check_dimensions(y_true, y_pred)

        # Calculate Precision and Recall
        precision = EvaluationMetrics.precision_score(y_true, y_pred)
        recall = EvaluationMetrics.recall_score(y_true, y_pred)

        # Calculate F1 score
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return f1

    @staticmethod
    def f_beta_score(y_true, y_pred, beta=1):
        """
        Calculate F-beta score for binary or multiclass classification.

        Parameters:
        - y_true: NumPy array, true labels.
        - y_pred: NumPy array, predicted labels.
        - beta: float, beta value. Default value is 1.

        Returns:
        - float: F-beta score.
        """
        # Ensure y_true and y_pred have the same shape
        EvaluationMetrics._check_dimensions(y_true, y_pred)

        # Calculate Precision and Recall
        precision = EvaluationMetrics.precision_score(y_true, y_pred)
        recall = EvaluationMetrics.recall_score(y_true, y_pred)

        # Calculate F-beta score
        f_beta = (
            (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
            if (beta**2 * precision + recall) > 0
            else 0
        )

        return f_beta

    @staticmethod
    def f_informedness(Y_true, Y_pred):
        """
        Calculate Informedness for binary or multiclass classification.

        Parameters:
        - y_true: NumPy array, true labels.
        - y_pred: NumPy array, predicted labels.

        Returns:
        - float: Informedness.
        """
        # Ensure y_true and y_pred have the same shape
        EvaluationMetrics._check_dimensions(Y_true, Y_pred)

        sum_not_y_true_and_not_y_pred = np.sum((1 - Y_true) * (1 - Y_pred), axis=0)
        sum_y_true_and_y_pred = np.sum(Y_true * Y_pred, axis=0)
        sum_not_y_true = np.sum(1 - Y_true, axis=0)
        sum_y_true = np.sum(Y_pred, axis=0)

        f_spec = sum_not_y_true_and_not_y_pred / sum_not_y_true
        f_rec = sum_y_true_and_y_pred / sum_y_true

        f_inf = 0.5 * (f_spec + f_rec)

        return f_inf.mean()

    @staticmethod
    def f_markedness(Y_true, Y_pred):
        """
        Calculate Markedness for binary or multiclass classification.

        Parameters:
        - y_true: NumPy array, true labels.
        - y_pred: NumPy array, predicted labels.

        Returns:
        - float: Markedness.
        """
        # Ensure y_true and y_pred have the same shape
        EvaluationMetrics._check_dimensions(Y_true, Y_pred)

        sum_not_y_true_and_not_y_pred = np.sum((1 - Y_true) * (1 - Y_pred), axis=0)
        sum_not_y_pred = np.sum(1 - Y_pred, axis=0)
        sum_y_true_and_y_pred = np.sum(Y_true * Y_pred, axis=0)
        sum_y_pred = np.sum(Y_pred, axis=0)

        f_neg = sum_not_y_true_and_not_y_pred / sum_not_y_pred
        f_pre = sum_y_true_and_y_pred / sum_y_pred

        f_mar = 0.5 * (f_neg + f_pre)

        return f_mar.mean()


class HandleMulanDatasetForMultiLabelArffFile:
    """Class to handle Mulan datasets stored in ARFF files."""

    def __init__(
        self,
        path,
        dataset_name,
        target_at_first=False,
        is_train=False,
        is_test=False,
        train_index=None,
    ):
        """Initialize the dataset handler."""
        self.path = path
        self.data = arff.loadarff(self.path)
        self.df = pd.DataFrame(self.data[0])

        # Handle train-test split for some datasets
        if is_train:
            self.df = self.df.sample(frac=0.8, random_state=SEED)
            self.train_index = self.df.index
        elif is_test:
            self.df = self.df.drop(train_index)

        self.dataset_name = dataset_name
        y_split_index = self._get_Y_split_index()

        if target_at_first:
            self.Y = self.df.iloc[:, :y_split_index].astype(int)
            self.X = self.df.iloc[:, y_split_index:]
        else:
            self.X = self.df.iloc[:, :-y_split_index]
            self.Y = self.df.iloc[:, -y_split_index:].astype(int)

    def _get_Y_split_index(self):
        """Get the index for splitting Y from X based on the dataset name."""
        if self.dataset_name == "emotions":
            return 6
        elif self.dataset_name == "corel5k":
            return 374
        elif self.dataset_name == "bitex":
            return 159
        elif self.dataset_name == "scene":
            return 6
        elif self.dataset_name == "yeast":
            return 14
        elif self.dataset_name == "CAL500":
            return 174
        elif self.dataset_name == "mediaMill":
            return 101
        elif self.dataset_name == "VirusGO_sparse":
            return 6
        elif self.dataset_name == "Water-quality":
            return 14
        elif self.dataset_name == "CHD_49":
            return 6

        else:
            raise Exception("Dataset name is not supported")


def read_datasets_from_folder(folder_path, dataset_names):
    """Read datasets from a folder and yield train and test sets."""
    if not os.path.isdir(folder_path):
        raise Exception("Folder path is not valid")

    def _get_result(filename, target_at_first=False):
        df_train = HandleMulanDatasetForMultiLabelArffFile(
            os.path.join(folder_path, f"{filename}.arff"),
            filename,
            target_at_first,
            is_train=True,
        )
        df_test = HandleMulanDatasetForMultiLabelArffFile(
            os.path.join(folder_path, f"{filename}.arff"),
            filename,
            target_at_first,
            is_test=True,
            train_index=df_train.train_index,
        )
        return df_train, df_test

    for filename in dataset_names:
        if filename in DATASET_WHOLE_FILES:
            yield _get_result(
                filename,
                target_at_first=(filename in DATASET_WHOLE_FILES_TARGET_AT_FIRST),
            )
        elif os.path.isdir(os.path.join(folder_path, filename)):
            # Handle individual datasets in subfolders
            print(f"Reading {filename} dataset...")
            df_train = HandleMulanDatasetForMultiLabelArffFile(
                os.path.join(folder_path, filename, f"{filename}-train.arff"), filename
            )
            df_test = HandleMulanDatasetForMultiLabelArffFile(
                os.path.join(folder_path, filename, f"{filename}-test.arff"), filename
            )
            yield df_train, df_test
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


def training_model(model, X_train, Y_train):
    """Train the specified model on the training data."""
    start_time = time.time()
    print(f"⏳ Training {model.base_estimator.__class__.__name__} model...")
    model.fit(X_train, Y_train)

    print(f"🤿 Elapsed training time: {time.time() - start_time:.5f} seconds")
    return model


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
        print(f"🤿 Elapsed predict time: {elapsed_time:.5f} seconds")

        score_metrics = calculate_metrics(Y_test, Y_pred, metric_funcs)
        loss_score_by_predict_func.append(
            {"predict_name": predict_func["name"], "score_metrics": score_metrics}
        )

    return loss_score_by_predict_func


def prepare_model_to_evaluate():
    """Prepare a list of models for evaluation."""
    base_estimators = [
        LogisticRegression(random_state=SEED),
        SGDClassifier(loss="log_loss", random_state=SEED),
        # RandomForestClassifier(random_state=SEED),
        # AdaBoostClassifier(random_state=SEED),
    ]
    return [ProbabilisticClassifierChainCustom(model) for model in base_estimators]


def main():
    """Main function to orchestrate the evaluation process."""
    # Define the list of models you want to evaluate
    evaluated_models = prepare_model_to_evaluate()

    absolute_path = os.path.abspath(__file__)
    # Define the folder path containing JSON datasets and the output CSV file name
    folder_path = os.path.join(absolute_path, "datasets")
    output_csv = os.path.join(absolute_path, "result/evaluation_results.csv")

    dataset_names = [
        "emotions",
        # "yeast",
        # "scene",
        # "VirusGO_sparse"
        # "CHD_49"
    ]
    # -----------------  MAIN -----------------
    # func is same name of the predict function in ProbabilisticClassifierChainCustom
    predict_functions = [
        {"name": "Predict Hamming Loss", "func": "predict_Hamming"},
        {"name": "Predict Subset", "func": "predict_Subset"},
        {"name": "Predict Pre", "func": "predict_Pre"},
        # {"name": "Predict Neg", "func": "predict_Neg"},
        # {"name": "Predict Recall", "func": "predict_Recall"},
        # {"name": "Predict Mar", "func": "predict_Mar"},
        # {"name": "Predict Fmeasure", "func": "predict_Fmeasure"},
        # {"name": "Predict Inf", "func": "predict_Inf"},
    ]

    metric_functions = [
        {"name": "Hamming Loss", "func": EvaluationMetrics.hamming_loss},
        # {"name": "Subset Accuracy", "func": EvaluationMetrics.subset_accuracy},
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

    # Iterate over datasets
    for df_train, df_test in read_datasets_from_folder(folder_path, dataset_names):
        print(f"\n🐳Evaluating on {df_train.dataset_name} dataset...")

        # For each dataset, iterate over the models and perform evaluation
        for model in evaluated_models:
            model = training_model(model, df_train.X, df_train.Y)
            loss_score_by_predict_func = evaluate_model(
                model, df_test.X, df_test.Y, predict_functions, metric_functions
            )

            # Collect and append evaluation results to the DataFrame
            print("-" * 10)
            for result in loss_score_by_predict_func:
                print("❄️ Metric: ", result["predict_name"])

                for score_metric in result["score_metrics"]:
                    data["Dataset"].append(df_train.dataset_name)
                    data["Model"].append(model.base_estimator.__class__.__name__)
                    data["Predict Function of Model"].append(result["predict_name"])

                    data["Metric Function"].append(score_metric["Metric Name"])
                    data["Score"].append(score_metric["Score"])

    result_df = pd.DataFrame(data)

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
