import numpy as np


class EvaluationMetrics:
    """A class for various evaluation metrics for multi-label classification.

    All metrics are reported in "higher is better" form (i.e., values in [0,1]
    where 1 is perfect). This includes hamming_accuracy (= 1 - Hamming Loss).
    """

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
    def hamming_accuracy(Y_true, Y_pred):
        """
        Calculate Hamming Accuracy (1 - Hamming Loss) for multilabel classification.

        Hamming Accuracy = fraction of correctly predicted (label, sample) pairs.
        Higher is better; perfect prediction = 1.0.

        Note: the original Hamming Loss metric is the complement (fraction of
        incorrect predictions). We report accuracy here for a uniform higher-is-better
        convention across all metrics.

        Parameters:
        - Y_true: NumPy array, true labels (2D array with shape [n_samples, n_labels]).
        - Y_pred: NumPy array, predicted labels (2D array with shape [n_samples, n_labels]).

        Returns:
        - float: Hamming Accuracy in [0, 1].
        """
        EvaluationMetrics._check_dimensions(Y_true, Y_pred)
        return 1 - np.mean(np.not_equal(Y_true, Y_pred))

    @staticmethod
    def precision_score(y_true, y_pred):
        """
        Calculate example-based Precision for multilabel classification.

        Per-sample: TP / (TP + FP). Averaged over all samples.
        Edge cases:
        - Both true and pred are all-zero → precision = 1 (vacuously correct).
        - Pred is all-zero but true has positives → precision = 0 (nothing predicted).

        Parameters:
        - y_true: NumPy array, true labels.
        - y_pred: NumPy array, predicted labels.

        Returns:
        - float: Precision score.
        """
        EvaluationMetrics._check_dimensions(y_true, y_pred)
        n_samples, _ = y_true.shape
        precision = np.zeros(n_samples)
        for i in range(n_samples):
            sum_true = np.sum(y_true[i])
            sum_pred = np.sum(y_pred[i])
            if sum_true == 0 and sum_pred == 0:
                precision[i] = 1
            elif sum_pred == 0:
                precision[i] = 0
            else:
                precision[i] = np.dot(y_true[i], y_pred[i]) / sum_pred
        return precision.mean()

    @staticmethod
    def recall_score(y_true, y_pred):
        """
        Calculate example-based Recall for multilabel classification.

        Per-sample: TP / (TP + FN). Averaged over all samples.
        Edge cases:
        - True has no positives and pred has no positives → recall = 1.
        - True has no positives but pred has positives → recall = 0.

        Parameters:
        - y_true: NumPy array, true labels.
        - y_pred: NumPy array, predicted labels.

        Returns:
        - float: Recall score.
        """
        EvaluationMetrics._check_dimensions(y_true, y_pred)
        n_samples, _ = y_true.shape
        recall = np.zeros(n_samples)
        for i in range(n_samples):
            sum_true = np.sum(y_true[i])
            if sum_true == 0:
                recall[i] = 1 if np.sum(y_pred[i]) == 0 else 0
            else:
                recall[i] = np.dot(y_true[i], y_pred[i]) / sum_true
        return recall.mean()

    @staticmethod
    def subset_accuracy(y_true, y_pred):
        """
        Calculate Subset Accuracy (exact match ratio) for multilabel classification.

        A sample is correct only if all its labels are predicted exactly.

        Parameters:
        - y_true: NumPy array, true labels (2D array with shape [n_samples, n_labels]).
        - y_pred: NumPy array, predicted labels (2D array with shape [n_samples, n_labels]).

        Returns:
        - float: Subset Accuracy in [0, 1].
        """
        EvaluationMetrics._check_dimensions(y_true, y_pred)
        correct_samples = np.sum(np.all(y_true == y_pred, axis=1))
        return correct_samples / len(y_true)

    @staticmethod
    def negative_predictive_value(y_true, y_pred):
        """
        Calculate example-based Negative Predictive Value (NPV) for multilabel classification.

        Per-sample: TN / (TN + FN). Averaged over all samples.
        Edge cases:
        - All labels predicted positive (no negative predictions):
            - If all true labels are also positive → NPV = 1 (vacuously correct).
            - Otherwise (some true negatives exist but missed) → NPV = 0.

        Parameters:
        - y_true: NumPy array, true labels.
        - y_pred: NumPy array, predicted labels.

        Returns:
        - float: NPV in [0, 1].
        """
        EvaluationMetrics._check_dimensions(y_true, y_pred)
        n_samples, n_labels = y_true.shape
        npv = np.zeros(n_samples)
        for i in range(n_samples):
            sum_pred = np.sum(y_pred[i])
            if sum_pred == n_labels:
                # All positive predictions: no negatives predicted.
                # TN = 0; FN = number of true negatives (1 - y_true).
                npv[i] = 1 if np.sum(y_true[i]) == n_labels else 0
            else:
                npv[i] = np.dot(1 - y_true[i], 1 - y_pred[i]) / np.sum(1 - y_pred[i])
        return npv.mean()

    @staticmethod
    def f1_score(y_true, y_pred):
        """
        Calculate example-based F1 score for multilabel classification.

        Per-sample Dice coefficient: 2*TP / (2*TP + FP + FN). Averaged over samples.
        Edge case: both true and pred are all-zero → F1 = 1.

        Parameters:
        - y_true: NumPy array, true labels.
        - y_pred: NumPy array, predicted labels.

        Returns:
        - float: F1 score in [0, 1].
        """
        EvaluationMetrics._check_dimensions(y_true, y_pred)
        n_samples, _ = y_true.shape
        f1 = np.zeros(n_samples)
        for i in range(n_samples):
            denom = np.sum(y_true[i]) + np.sum(y_pred[i])
            if denom == 0:
                f1[i] = 1
            else:
                f1[i] = (2 * np.dot(y_true[i], y_pred[i])) / denom
        return f1.mean()

    @staticmethod
    def f_beta_score(y_true, y_pred, beta=1):
        """
        Calculate example-based F-beta score for multilabel classification.

        Per-sample: (1+β²)*TP / (β²*(TP+FN) + (TP+FP)). Averaged over samples.
        Edge case: both true and pred are all-zero → F_β = 1.

        Parameters:
        - y_true: NumPy array, true labels.
        - y_pred: NumPy array, predicted labels.
        - beta: float, controls precision/recall trade-off. Default = 1 (F1).

        Returns:
        - float: F-beta score in [0, 1].
        """
        EvaluationMetrics._check_dimensions(y_true, y_pred)
        n_samples, _ = y_true.shape
        f_beta = np.zeros(n_samples)
        for i in range(n_samples):
            denom = (beta**2) * np.sum(y_true[i]) + np.sum(y_pred[i])
            if denom == 0:
                f_beta[i] = 1
            else:
                f_beta[i] = (1 + beta**2) * np.dot(y_true[i], y_pred[i]) / denom
        return f_beta.mean()

    @staticmethod
    def f_informedness(Y_true, Y_pred):
        """
        Calculate label-averaged Informedness (Balanced Accuracy per label) for multilabel classification.

        Per-label: 0.5 * (Specificity + Sensitivity)
            = 0.5 * (TN/N_true_negative + TP/N_true_positive)
        Averaged over labels.

        Parameters:
        - Y_true: NumPy array, true labels (shape [n_samples, n_labels]).
        - Y_pred: NumPy array, predicted labels (shape [n_samples, n_labels]).

        Returns:
        - float: Informedness in [0, 1].
        """
        EvaluationMetrics._check_dimensions(Y_true, Y_pred)

        tn = np.sum((1 - Y_true) * (1 - Y_pred), axis=0)
        tp = np.sum(Y_true * Y_pred, axis=0)
        n_true_neg = np.sum(1 - Y_true, axis=0)
        n_true_pos = np.sum(Y_true, axis=0)  # fixed: was np.sum(Y_pred, axis=0)

        f_spec = np.where(n_true_neg > 0, tn / n_true_neg, 1.0)
        f_sens = np.where(n_true_pos > 0, tp / n_true_pos, 1.0)

        return (0.5 * (f_spec + f_sens)).mean()

    @staticmethod
    def f_markedness(Y_true, Y_pred):
        """
        Calculate example-based Markedness for multilabel classification.

        Per-sample: 0.5 * (NPV + Precision). Averaged over samples.
        Edge cases for NPV and Precision follow the same conventions as
        negative_predictive_value() and precision_score() respectively.

        Parameters:
        - Y_true: NumPy array, true labels (shape [n_samples, n_labels]).
        - Y_pred: NumPy array, predicted labels (shape [n_samples, n_labels]).

        Returns:
        - float: Markedness in [0, 1].
        """
        EvaluationMetrics._check_dimensions(Y_true, Y_pred)
        n_samples, n_labels = Y_true.shape
        npv = np.zeros(n_samples)
        precision = np.zeros(n_samples)

        for i in range(n_samples):
            sum_pred = np.sum(Y_pred[i])
            sum_true = np.sum(Y_true[i])

            # NPV: TN / (TN + FN)
            if sum_pred == n_labels:
                npv[i] = 1 if sum_true == n_labels else 0
            else:
                npv[i] = np.dot(1 - Y_true[i], 1 - Y_pred[i]) / np.sum(1 - Y_pred[i])

            # Precision: TP / (TP + FP)
            if sum_true == 0 and sum_pred == 0:
                precision[i] = 1
            elif sum_pred == 0:
                precision[i] = 0  # fixed: was 1 when pred=0 and true>0
            else:
                precision[i] = np.dot(Y_true[i], Y_pred[i]) / sum_pred

        return (0.5 * (npv + precision)).mean()
