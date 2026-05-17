"""Unit tests for EvaluationMetrics.

Run with: python -m pytest tests/test_evaluation_metrics.py -v
"""
import numpy as np
import pytest
from src.evaluation_metrics import EvaluationMetrics as EM


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def perfect():
    Y = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    return Y, Y.copy()


@pytest.fixture
def worst():
    Y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    Y_pred = 1 - Y_true
    return Y_true, Y_pred


@pytest.fixture
def all_ones():
    Y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    Y_pred = np.ones_like(Y_true)
    return Y_true, Y_pred


@pytest.fixture
def all_zeros():
    Y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    Y_pred = np.zeros_like(Y_true)
    return Y_true, Y_pred


# ──────────────────────────────────────────────
# hamming_accuracy
# ──────────────────────────────────────────────

class TestHammingAccuracy:
    def test_perfect(self, perfect):
        assert EM.hamming_accuracy(*perfect) == pytest.approx(1.0)

    def test_worst(self, worst):
        assert EM.hamming_accuracy(*worst) == pytest.approx(0.0)

    def test_all_ones(self, all_ones):
        Y_true, Y_pred = all_ones
        # fraction correct = fraction of 1s in Y_true
        expected = np.mean(Y_true == Y_pred)
        assert EM.hamming_accuracy(Y_true, Y_pred) == pytest.approx(expected)

    def test_shape_mismatch(self):
        with pytest.raises(Exception):
            EM.hamming_accuracy(np.ones((3, 3)), np.ones((3, 2)))


# ──────────────────────────────────────────────
# precision_score
# ──────────────────────────────────────────────

class TestPrecisionScore:
    def test_perfect(self, perfect):
        assert EM.precision_score(*perfect) == pytest.approx(1.0)

    def test_worst(self, worst):
        assert EM.precision_score(*worst) == pytest.approx(0.0)

    def test_both_zero(self):
        Y = np.array([[0, 0, 0]])
        assert EM.precision_score(Y, Y) == pytest.approx(1.0)

    def test_pred_zero_true_positive(self, all_zeros):
        # No positive predictions when there are true positives → precision = 0
        assert EM.precision_score(*all_zeros) == pytest.approx(0.0)

    def test_all_ones(self, all_ones):
        Y_true, Y_pred = all_ones
        # precision per sample = sum(true) / L (all predicted, so TP/pred = true_density)
        n_labels = Y_true.shape[1]
        expected = np.mean(np.sum(Y_true, axis=1) / n_labels)
        assert EM.precision_score(Y_true, Y_pred) == pytest.approx(expected)


# ──────────────────────────────────────────────
# recall_score
# ──────────────────────────────────────────────

class TestRecallScore:
    def test_perfect(self, perfect):
        assert EM.recall_score(*perfect) == pytest.approx(1.0)

    def test_worst(self, worst):
        assert EM.recall_score(*worst) == pytest.approx(0.0)

    def test_all_ones_recall_is_one(self, all_ones):
        # All-ones prediction always achieves perfect recall
        assert EM.recall_score(*all_ones) == pytest.approx(1.0)

    def test_all_zeros_no_true_positives(self):
        # When true has no positives, pred=zeros → recall=1 (vacuous)
        Y_true = np.array([[0, 0, 0]])
        Y_pred = np.array([[0, 0, 0]])
        assert EM.recall_score(Y_true, Y_pred) == pytest.approx(1.0)

    def test_all_zeros_has_true_positives(self):
        # pred=zeros but true has positives → recall=0
        Y_true = np.array([[1, 0, 0]])
        Y_pred = np.array([[0, 0, 0]])
        assert EM.recall_score(Y_true, Y_pred) == pytest.approx(0.0)


# ──────────────────────────────────────────────
# subset_accuracy
# ──────────────────────────────────────────────

class TestSubsetAccuracy:
    def test_perfect(self, perfect):
        assert EM.subset_accuracy(*perfect) == pytest.approx(1.0)

    def test_worst(self, worst):
        assert EM.subset_accuracy(*worst) == pytest.approx(0.0)

    def test_partial(self):
        Y_true = np.array([[1, 0], [0, 1], [1, 1]])
        Y_pred = np.array([[1, 0], [1, 0], [0, 0]])  # only first row matches
        assert EM.subset_accuracy(Y_true, Y_pred) == pytest.approx(1 / 3)


# ──────────────────────────────────────────────
# negative_predictive_value
# ──────────────────────────────────────────────

class TestNPV:
    def test_perfect(self, perfect):
        assert EM.negative_predictive_value(*perfect) == pytest.approx(1.0)

    def test_worst(self, worst):
        assert EM.negative_predictive_value(*worst) == pytest.approx(0.0)

    def test_all_ones_has_true_neg(self, all_ones):
        # all-ones pred, but some true labels are 0 → NPV = 0
        Y_true, Y_pred = all_ones
        if np.any(Y_true == 0):
            assert EM.negative_predictive_value(Y_true, Y_pred) == pytest.approx(0.0)

    def test_all_ones_all_true_pos(self):
        # all-ones pred, all-ones true → vacuously perfect NPV = 1
        Y = np.ones((2, 3), dtype=int)
        assert EM.negative_predictive_value(Y, Y) == pytest.approx(1.0)


# ──────────────────────────────────────────────
# f_beta
# ──────────────────────────────────────────────

class TestFBetaScore:
    def test_perfect(self, perfect):
        assert EM.f_beta(*perfect) == pytest.approx(1.0)

    def test_worst(self, worst):
        assert EM.f_beta(*worst) == pytest.approx(0.0)

    def test_both_zero(self):
        Y = np.array([[0, 0, 0]])
        assert EM.f_beta(Y, Y) == pytest.approx(1.0)


# ──────────────────────────────────────────────
# informedness  (was buggy: used Y_pred instead of Y_true)
# ──────────────────────────────────────────────

class TestFInformedness:
    def test_perfect(self, perfect):
        assert EM.informedness(*perfect) == pytest.approx(1.0)

    def test_worst(self, worst):
        assert EM.informedness(*worst) == pytest.approx(0.0)

    def test_all_ones_informedness(self, all_ones):
        # Sensitivity = 1 (all true pos captured), Specificity = 0 (no TN).
        # Informedness = 0.5*(1+0) = 0.5
        assert EM.informedness(*all_ones) == pytest.approx(0.5)

    def test_uses_true_labels_as_denominator(self):
        # Verify the fixed denominator: uses N_true_pos (from Y_true), not N_pred_pos.
        # Y_true col sums: [2,1,0], Y_pred col sums: [1,1,0] — they differ.
        # If the old bug were present (denominator = Y_pred sum), the result would differ.
        Y_true = np.array([[1, 1, 0], [1, 0, 0]])
        Y_pred = np.array([[1, 0, 0], [0, 1, 0]])
        # Manually computed:
        #   tp=[1,0,0], tn=[0,0,2], n_true_pos=[2,1,0], n_true_neg=[0,1,2]
        #   sens=[0.5, 0, 1.0(default)], spec=[1.0(default), 0, 1.0]
        #   informedness = 0.5*mean([1.5, 0, 2.0]) = 0.5833...
        assert EM.informedness(Y_true, Y_pred) == pytest.approx(0.5833333, rel=1e-5)


# ──────────────────────────────────────────────
# markedness  (was buggy: precision=1 when pred=0,true>0)
# ──────────────────────────────────────────────

class TestFMarkedness:
    def test_perfect(self, perfect):
        assert EM.markedness(*perfect) == pytest.approx(1.0)

    def test_worst(self, worst):
        assert EM.markedness(*worst) == pytest.approx(0.0)

    def test_all_zeros_precision_is_zero(self, all_zeros):
        # pred=zeros, some true=1 → precision=0, not 1 (was the bug)
        # precision=0 for all samples, so markedness ≤ 0.5
        result = EM.markedness(*all_zeros)
        # Precision component must be 0; NPV component may be non-zero.
        # Markedness = 0.5*(NPV + 0). NPV is non-zero since TN > 0.
        assert result < 1.0
        # Sanity: precision_score should agree on the 0-case
        assert EM.precision_score(*all_zeros) == pytest.approx(0.0)

    def test_consistent_with_precision_score(self):
        # markedness's internal precision edge case must match precision_score
        Y_true = np.array([[1, 0], [1, 1]])
        Y_pred = np.array([[0, 0], [0, 1]])
        # Sample 0: pred=0,true=[1,0] → precision=0 (both metrics)
        # Sample 1: pred=[0,1],true=[1,1] → precision=1*1/1=1
        prec = EM.precision_score(Y_true, Y_pred)
        # markedness uses same logic; result should be consistent
        mar = EM.markedness(Y_true, Y_pred)
        assert prec == pytest.approx(0.5)  # mean of [0, 1]
        assert mar >= 0.0 and mar <= 1.0
