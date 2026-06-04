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
    def macro_f1(Y_true, Y_pred):
        """Macro-averaged F1: per-label F1 (aggregated across samples), then mean over labels.

        Per-label: F1_j = 2*TP_j / (2*TP_j + FP_j + FN_j).
        Edge case: label with no positives in both Y_true and Y_pred → 1.0 (vacuous);
        label with positives in Y_true but none predicted → 0.0.
        """
        EvaluationMetrics._check_dimensions(Y_true, Y_pred)
        tp = np.sum(Y_true * Y_pred, axis=0).astype(float)
        fp = np.sum((1 - Y_true) * Y_pred, axis=0).astype(float)
        fn = np.sum(Y_true * (1 - Y_pred), axis=0).astype(float)
        denom = 2 * tp + fp + fn
        f1 = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 1.0)
        return float(f1.mean())

    @staticmethod
    def micro_f1(Y_true, Y_pred):
        """Micro-averaged F1: aggregate TP/FP/FN globally, then single F1.

        Equivalent to weighting each (sample, label) pair equally.
        Edge case: both Y_true and Y_pred all zeros → 1.0.
        """
        EvaluationMetrics._check_dimensions(Y_true, Y_pred)
        tp = float(np.sum(Y_true * Y_pred))
        fp = float(np.sum((1 - Y_true) * Y_pred))
        fn = float(np.sum(Y_true * (1 - Y_pred)))
        denom = 2 * tp + fp + fn
        return 1.0 if denom == 0 else 2 * tp / denom

    @staticmethod
    def informedness(Y_true, Y_pred):
        """
        Example-based Informedness F_Inf = 1/2 (Specificity + Recall), averaged
        over samples (paper eq. informedness).

        Per sample:
            Recall      = TP / (TP + FN) = Σ ŷ·y / Σ y                (= 1 if Σ y = 0)
            Specificity = TN / (TN + FP) = Σ (1-ŷ)(1-y) / Σ (1-y)     (= 1 if Σ (1-y) = 0)
            F_Inf       = 1/2 (Specificity + Recall)
        The vacuous conventions are on the true label vector (a sample with no
        true positives → Recall = 1; with no true negatives → Specificity = 1),
        matching the appendix derivation of the Informedness BOP.

        Parameters:
        - Y_true: NumPy array, true labels (shape [n_samples, n_labels]).
        - Y_pred: NumPy array, predicted labels (shape [n_samples, n_labels]).

        Returns:
        - float: Informedness in [0, 1].
        """
        EvaluationMetrics._check_dimensions(Y_true, Y_pred)
        n_samples = Y_true.shape[0]
        vals = np.zeros(n_samples)
        for i in range(n_samples):
            yt = Y_true[i]
            sum_pos = np.sum(yt)
            sum_neg = np.sum(1 - yt)
            recall = 1.0 if sum_pos == 0 else np.dot(yt, Y_pred[i]) / sum_pos
            spec = 1.0 if sum_neg == 0 else np.dot(1 - yt, 1 - Y_pred[i]) / sum_neg
            vals[i] = 0.5 * (spec + recall)
        return vals.mean()

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

    # ────────────────────────────────────────────────────────────────────
    # Ranking-based metrics (Schapire & Singer 2000).
    # These take continuous SCORES per label (e.g. marginal P(y_j=1|x))
    # rather than binary predictions. All are reported in higher-is-better
    # form: 1 - raw_metric, scaled to [0, 1] where applicable.
    # ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _rank_descending(scores):
        """Return rank (1-indexed) of each entry along last axis, ties broken arbitrarily.

        rank 1 = highest score. Shape preserved.
        """
        order = np.argsort(-scores, axis=-1)  # indices of descending sort
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.arange(1, scores.shape[-1] + 1), axis=-1)
        return ranks

    @staticmethod
    def one_error_score(Y_true, scores):
        """1 - OneError. Top-scored label IS in the true label set → 1, else 0.

        Higher is better. Range [0, 1].
        """
        EvaluationMetrics._check_dimensions(Y_true, scores)
        top = np.argmax(scores, axis=1)
        hits = Y_true[np.arange(len(Y_true)), top]
        return float(hits.mean())

    @staticmethod
    def coverage_score(Y_true, scores):
        """1 - normalised Coverage error.

        Coverage_raw = mean over samples of: how far down the ranked list (1-indexed)
        we must go to include all true labels. Normalised to [0, 1] by (L-1).
        Returned as 1 - normalised (higher better).
        Samples with no positive labels contribute 0 to raw coverage (vacuous).
        """
        EvaluationMetrics._check_dimensions(Y_true, scores)
        N, L = Y_true.shape
        if L <= 1:
            return 1.0
        ranks = EvaluationMetrics._rank_descending(scores)  # (N, L)
        # Per sample: max rank among labels where Y_true=1; if no positives, 0.
        masked = np.where(Y_true.astype(bool), ranks, 0)
        max_rank = masked.max(axis=1)  # (N,)
        # Normalise: rank ∈ [1, L] for samples with positives; map to [0, 1] via (rank-1)/(L-1)
        # For samples with no positives, max_rank = 0; convention: 0 normalised (perfect).
        norm = np.where(max_rank > 0, (max_rank - 1) / (L - 1), 0.0)
        return float(1.0 - norm.mean())

    @staticmethod
    def ranking_loss_score(Y_true, scores):
        """1 - Ranking Loss.

        Ranking Loss = avg fraction of (relevant, irrelevant) label pairs that are
        misordered (score of irrelevant >= score of relevant). Lower is better → we
        return 1 - RL. Samples with all-positive or all-negative labels contribute 0
        loss (vacuous).
        """
        EvaluationMetrics._check_dimensions(Y_true, scores)
        N, L = Y_true.shape
        rl = np.zeros(N)
        for n in range(N):
            pos = np.where(Y_true[n] == 1)[0]
            neg = np.where(Y_true[n] == 0)[0]
            if len(pos) == 0 or len(neg) == 0:
                rl[n] = 0.0
                continue
            s_pos = scores[n, pos][:, None]    # (|pos|, 1)
            s_neg = scores[n, neg][None, :]    # (1, |neg|)
            # Pair misorder: s_pos <= s_neg (strict tie counted as half-violation)
            mis = (s_pos < s_neg).sum() + 0.5 * (s_pos == s_neg).sum()
            rl[n] = mis / (len(pos) * len(neg))
        return float(1.0 - rl.mean())

    @staticmethod
    def average_precision_score(Y_true, scores):
        """Label-ranking Average Precision (LRAP).

        For each true label, precision @ its rank (fraction of labels ranked
        at-or-above it that are also true). Averaged within sample, then across.
        Samples with no positive labels contribute 1 (vacuous).
        Higher is better. Range [0, 1].
        """
        EvaluationMetrics._check_dimensions(Y_true, scores)
        N, L = Y_true.shape
        ranks = EvaluationMetrics._rank_descending(scores)  # (N, L)
        ap = np.zeros(N)
        for n in range(N):
            pos = np.where(Y_true[n] == 1)[0]
            if len(pos) == 0:
                ap[n] = 1.0
                continue
            pos_ranks = ranks[n, pos]  # ranks of true labels
            # For each true label at rank r: count how many of the true labels have rank ≤ r.
            sorted_pr = np.sort(pos_ranks)
            counts = np.arange(1, len(pos) + 1)  # 1, 2, ..., |pos|
            ap[n] = float(np.mean(counts / sorted_pr))
        return float(ap.mean())
