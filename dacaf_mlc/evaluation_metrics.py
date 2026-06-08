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
            raise ValueError(
                f"Y_true and Y_pred have different shapes: {Y_true.shape} vs {Y_pred.shape}"
            )

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
        # Per-sample Precision = TP / (TP + FP) = TP / sum_pred, averaged over samples.
        # Binary inputs ⇒ every reduction is an exact integer sum (vectorized = loop).
        sum_true = y_true.sum(axis=1)
        sum_pred = y_pred.sum(axis=1)
        tp = (y_true * y_pred).sum(axis=1)
        # Safe denominator avoids 0/0; the sum_pred == 0 rows are overwritten next.
        precision = tp / np.where(sum_pred == 0, 1, sum_pred)
        # Vacuous cases: pred empty & true empty → 1; pred empty & true non-empty → 0.
        precision = np.where(sum_pred == 0, np.where(sum_true == 0, 1.0, 0.0), precision)
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
        # Per-sample Recall = TP / (TP + FN) = TP / sum_true, averaged over samples.
        sum_true = y_true.sum(axis=1)
        sum_pred = y_pred.sum(axis=1)
        tp = (y_true * y_pred).sum(axis=1)
        recall = tp / np.where(sum_true == 0, 1, sum_true)
        # Vacuous: true empty → 1 if pred also empty, else 0.
        recall = np.where(sum_true == 0, np.where(sum_pred == 0, 1.0, 0.0), recall)
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
        # Per-sample NPV = TN / (TN + FN) = TN / (#predicted negatives), averaged.
        n_labels = y_true.shape[1]
        sum_pred = y_pred.sum(axis=1)
        pred_neg = n_labels - sum_pred                      # = sum(1 - y_pred)
        tn = ((1 - y_true) * (1 - y_pred)).sum(axis=1)
        npv = tn / np.where(pred_neg == 0, 1, pred_neg)
        # Vacuous (paper Corollary 2): no predicted negatives (pred all-ones) → 1.
        npv = np.where(sum_pred == n_labels, 1.0, npv)
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
        # Per-sample F_β = (1+β²)·TP / (β²·sum_true + sum_pred), averaged over samples.
        sum_true = y_true.sum(axis=1)
        sum_pred = y_pred.sum(axis=1)
        tp = (y_true * y_pred).sum(axis=1)
        denom = (beta**2) * sum_true + sum_pred
        f_beta = (1 + beta**2) * tp / np.where(denom == 0, 1, denom)
        # Vacuous: denom == 0 ⇔ true and pred both empty → 1.
        f_beta = np.where(denom == 0, 1.0, f_beta)
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
        # Per-sample Markedness = 0.5·(NPV + Precision), averaged over samples.
        # NB: these vacuous conventions (Fpre(.,0_K)=Fneg(.,1_K)=1) follow the paper's
        # markedness definition and intentionally differ from precision_score()'s.
        n_labels = Y_true.shape[1]
        sum_pred = Y_pred.sum(axis=1)
        tp = (Y_true * Y_pred).sum(axis=1)
        tn = ((1 - Y_true) * (1 - Y_pred)).sum(axis=1)
        pred_neg = n_labels - sum_pred

        # NPV: TN / (#predicted negatives); pred all-ones → 1 (vacuous, Fneg(.,1_K)=1).
        npv = np.where(sum_pred == n_labels, 1.0, tn / np.where(pred_neg == 0, 1, pred_neg))
        # Precision: TP / sum_pred; empty prediction → 1 (vacuous, Fpre(.,0_K)=1).
        precision = np.where(sum_pred == 0, 1.0, tp / np.where(sum_pred == 0, 1, sum_pred))
        return (0.5 * (npv + precision)).mean()
