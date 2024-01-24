import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

import scipy.io.arff as arff
import pandas as pd

from probability_classifier_chains import ProbabilisticClassifierChainCustom

# # Load the ARFF file
# data = arff.loadarff('your_dataset.arff')
# # Convert the ARFF data to a Pandas DataFrame
# df = pd.DataFrame(data[0])

# Now, df contains your data as a Pandas DataFrame
# You can perform various data analysis tasks with df

SEED = 6


# Create static class of EvaluationMetrics
class EvaluationMetrics:
    @staticmethod
    def _check_dimensions(Y_true, Y_pred):
        if Y_true.shape != Y_pred.shape:
            raise Exception("Y_true and Y_pred have different shapes")

    @staticmethod
    def get_loss(Y_true, Y_pred, loss_func):
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
    def __init__(
        self,
        path,
        dataset_name,
        target_at_first=False,
        is_train=False,
        is_test=False,
        train_index=None,
    ):
        self.path = path
        self.data = arff.loadarff(self.path)
        self.df = pd.DataFrame(self.data[0])

        # just for some dataset need to split train test 0% on dataframes randomly
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

    # Handle custom dataset name to get Y column which is multi-label
    def _get_Y_split_index(self):
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


# Define a function to read datasets from JSON files in a folder using a generator function (yield)
def read_datasets_from_folder(folder_path, dataset_names):
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
        if filename == "VirusGO_sparse":
            yield _get_result(filename)
        elif filename == "Water-quality":
            yield _get_result(filename, target_at_first=True)
        elif filename == "CHD_49":
            yield _get_result(filename, target_at_first=True)
        # check is folder
        elif os.path.isdir(os.path.join(folder_path, filename)):
            # TODO: check if file exist and flexible with testing file

            # Training data
            print(f"Reading {filename} dataset...")
            df_train = HandleMulanDatasetForMultiLabelArffFile(
                os.path.join(folder_path, filename, f"{filename}-train.arff"), filename
            )

            # Testing data
            df_test = HandleMulanDatasetForMultiLabelArffFile(
                os.path.join(folder_path, filename, f"{filename}-test.arff"), filename
            )
            yield df_train, df_test
        else:
            raise Exception("Dataset name is not supported")


def calculate_metrics(Y_true, Y_pred, metric_funcs):
    try:
        score_metrics = []

        for metric in metric_funcs:
            # print(f"\n{metric}\n")
            # Check options in metric function
            if "options" in metric:
                metric_name, metric_func, options = (
                    metric["name"],
                    metric["func"],
                    metric["options"],
                )
                score = f"{metric_func(Y_pred,Y_true, **options):.5f}"
            else:
                metric_name, metric_func = metric["name"], metric["func"]
                score = f"{metric_func(Y_pred,Y_true):.5f}"

            score_metrics.append(
                {
                    "Metric Name": metric_name,
                    "Metric Function": metric_func.__name__,
                    "Score": score,
                }
            )

        # print(f"score_metrics:\t{score_metrics}")
    except Exception as e:
        print("-" * 10)

        print(f"Error: {e}")
        print(f"Y_true:\t{Y_true.shape}\nY_pred:\t{Y_pred.shape}")
        print(f"Metric: \t{metric}")

        print("-" * 10)

    return score_metrics


def training_model(model, X_train, Y_train):
    # ----------------- Fit -----------------
    start_time = time.time()
    print(f"⏳ Training {model.base_estimator.__class__.__name__} model...")
    model.fit(X_train, Y_train)

    print(f"🤿 Elapsed training time: {time.time() - start_time:.5f} seconds")
    return model


# Define a function to perform model evaluation on each dataset
def evaluate_model(
    model: ProbabilisticClassifierChainCustom,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
    predict_funcs: list,
    metric_funcs: list,
) -> list:
    """_summary_

    Args:
        model (ProbabilisticClassifierChainCustom): _description_
        X_test (pd.DataFrame): _description_
        Y_test (pd.DataFrame): _description_
        predict_funcs (list): [{'name': 'Hamming Loss', 'func': predict_HammingLoss}, ...}]
        metric_funcs (list): [{'name': "Predict", 'func': "predict"}, {'name': "Predict Hamming Lost", 'func': 'predict_Hamming'}, ...]

    Returns:
        list: [{'predict_name': "Predict", 'score_metrics': [{'Metric Name': 'Hamming Loss', 'Metric Function': 'hamming_loss', 'Score': 0.0}, ...]}]
    """

    # ----------------- Predict -----------------
    print(f"{'-'*50}\nPredicting {model.base_estimator.__class__.__name__} model...")
    # [{'name': "Predict", 'score_metrics': [
    # {'Metric Name': 'Hamming Loss', 'Metric Function': 'hamming_loss', 'Score': 0.0},
    # {'Metric Name': 'Accuracy Score', 'Metric Function': 'accuracy_score', 'Score': 1.0},
    # {'Metric Name': 'Precision Score', 'Metric Function': 'precision_score', 'Score': 1.0},
    # {'Metric Name': 'Recall Score', 'Metric Function': 'recall_score', 'Score': 1.0},
    # {'Metric Name': 'F1 Score', 'Metric Function': 'f1_score', 'Score': 1.0}]}
    # ]
    loss_score_by_predict_func = []
    # TODO: predict_HammingLoss, predict_Inf, predict_Mar, predict_Neg, predict_Pre, predict_Subset,...
    for predict_func in predict_funcs:
        # Measure time for evaluation
        start_time = time.time()

        if predict_func["func"] == "predict":
            Y_pred, _, _ = model.predict(X_test)
        else:
            Y_pred = getattr(model, predict_func["func"])(X_test)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"🤿 Elapsed predict time: {elapsed_time:.5f} seconds")

        # print(f"🍓 Calculating metrics for {predict_func['name']}...")
        score_metrics = calculate_metrics(Y_test, Y_pred, metric_funcs)

        # end_time2 = time.time()
        # elapsed_time2 = end_time2 - end_time
        # print(f"🍓 Elapsed calculate metrics time: {elapsed_time2:.5f} seconds")

        loss_score_by_predict_func.append(
            {"predict_name": predict_func["name"], "score_metrics": score_metrics}
        )
    return loss_score_by_predict_func


def prepare_model_to_evaluate():
    # TODO: add more models
    pcc = [
        LogisticRegression(random_state=SEED),
        SGDClassifier(loss="log_loss", random_state=SEED),
        RandomForestClassifier(random_state=SEED),
        AdaBoostClassifier(random_state=SEED),
    ]

    # Add more models here if you want to evaluate them
    # Iterate all BOP methods per each classifier such as SGD, etc.
    return [ProbabilisticClassifierChainCustom(model) for model in pcc]


def main():
    # Define the list of models you want to evaluate
    evaluated_models = prepare_model_to_evaluate()

    # Define the folder path containing JSON datasets and the output CSV file name
    folder_path = (
        "/Users/xuantruong/Documents/JAIST/scikit-multiflow/tests/evaluation/dataset"
    )
    output_csv = "/Users/xuantruong/Documents/JAIST/scikit-multiflow/tests/evaluation/result/evaluation_results.csv"

    dataset_names = [
        "emotions",
        # "yeast",
        # "scene",
        # "VirusGO_sparse"
        # "CHD_49"
    ]
    # -----------------  MAIN -----------------
    # func is same name of the predict function in ProbabilisticClassifierChainCustom
    predict_funcs = [
        {"name": "Predict Hamming Loss", "func": "predict_Hamming"},
        {"name": "Predict Subset", "func": "predict_Subset"},
        {"name": "Predict Pre", "func": "predict_Pre"},
        # {"name": "Predict Neg", "func": "predict_Neg"},
        # {"name": "Predict Recall", "func": "predict_Recall"},
        # {"name": "Predict Mar", "func": "predict_Mar"},
        # {"name": "Predict Fmeasure", "func": "predict_Fmeasure"},
        # {"name": "Predict Inf", "func": "predict_Inf"},
    ]

    metric_funcs = [
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

    # Iterate over the datasets and models, perform evaluation, and append the results to the DataFrame
    for dataset in read_datasets_from_folder(folder_path, dataset_names):
        print(f"{'-'*50}\n🐳 Dataset: {dataset[0].dataset_name}")
        df_train, df_test = dataset

        # convert df to numpy array
        X_train, Y_train = df_train.X.to_numpy(), df_train.Y.to_numpy()
        X_test, Y_test = df_test.X.to_numpy(), df_test.Y.to_numpy()
        print(f"X_train:\t{X_train.shape}\nY_train:\t{Y_train.shape}")

        # For each dataset, iterate over the models and perform evaluation
        for model in evaluated_models:
            trained_model = training_model(model, X_train, Y_train)

            loss_score_by_predict_funcs = evaluate_model(
                trained_model, X_test, Y_test, predict_funcs, metric_funcs
            )
            # Format of loss_score_by_predict_funcs: [
            #   { 'predict_name': "Predict",
            #       'score_metrics': [{'Metric Name': 'Hamming Loss', 'Metric Function': 'hamming_loss', 'Score': 0.0}, ...]
            #   }
            # ]

            # Add the results to the DataFrame.
            print("-" * 10)
            for loss_score_by_predict_func in loss_score_by_predict_funcs:
                print("❄️ Metric: ", loss_score_by_predict_func["predict_name"])

                for score_metric in loss_score_by_predict_func["score_metrics"]:
                    data["Dataset"].append(dataset[0].dataset_name)
                    data["Model"].append(model.base_estimator.__class__.__name__)

                    data["Predict Function of Model"].append(
                        loss_score_by_predict_func["predict_name"]
                    )
                    data["Metric Function"].append(score_metric["Metric Name"])
                    data["Score"].append(score_metric["Score"])

    results = pd.DataFrame(data)

    results.to_csv(output_csv, index=False)


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

    # TODO:
    # [X] 5 datasets (<20 labels)
    #   [X] emotions
    #   [X] yeast
    #   [X] scene
    #   [] mediaMill
    #   [] CAL500

    # [] Models:
    #   [X] LogisticRegression
    #   [X] SGDClassifier
    #   [] RandomForestClassifier
    #   [] AdaBoostClassifier
    # [] F-measure, Informedness
    #   [] F-measure
    #   [] Informedness
    # [] Implement loss functions
    #   [X] Hamming Loss
    #   [X] Subset Accuracy
    #   [X] Precision Score
    #   [X] Recall Score
    #   [X] Negative Predictive Value
    #   [X] F1 Score
    #   [X] F Beta Score
    #   [ ] Informedness
    #   [ ] Markedness
