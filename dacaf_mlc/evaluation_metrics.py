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
        Edge case (paper convention, Corollary 2):
        - All labels predicted positive (no predicted negatives) → NPV is
          undefined; the paper assigns Fneg(y, 1_K) = 1 (vacuously perfect),
          regardless of y_true.

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
                # No predicted negatives → NPV undefined; paper assigns 1 (vacuous).
                npv[i] = 1
            else:
                npv[i] = np.dot(1 - y_true[i], 1 - y_pred[i]) / np.sum(1 - y_pred[i])
        return npv.mean()

    @staticmethod
    def f_beta(y_true, y_pred, beta=1):
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
    def markedness(Y_true, Y_pred):
        """
        Calculate example-based Markedness for multilabel classification.

        Per-sample: 0.5 * (NPV + Precision). Averaged over samples.

        Paper convention (Proposition 6 / Algorithm 4): the vacuous cases are
        assigned Fpre(y, 0_K) = 1 and Fneg(y, 1_K) = 1, unconditionally. This is
        what the published markedness results use, and it intentionally DIFFERS
        from the standalone precision_score() (which follows the scikit-learn
        convention of 0 when nothing is predicted but true positives exist).

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

            # NPV: TN / (TN + FN); no predicted negatives → 1 (vacuous, Fneg(.,1_K)=1).
            if sum_pred == n_labels:
                npv[i] = 1
            else:
                npv[i] = np.dot(1 - Y_true[i], 1 - Y_pred[i]) / np.sum(1 - Y_pred[i])

            # Precision: TP / (TP + FP); empty prediction → 1 (vacuous, Fpre(.,0_K)=1).
            if sum_pred == 0:
                precision[i] = 1
            else:
                precision[i] = np.dot(Y_true[i], Y_pred[i]) / sum_pred

        return (0.5 * (npv + precision)).mean()
