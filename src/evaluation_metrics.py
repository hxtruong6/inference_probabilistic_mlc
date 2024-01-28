import numpy as np


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
