"""Unit tests for EvaluationMetrics.

Run with: python -m pytest tests/test_evaluation_metrics.py -v
"""
import numpy as np
import pytest
from dacaf_mlc.evaluation_metrics import EvaluationMetrics as EM


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

    def test_all_ones_is_vacuous_one(self, all_ones):
        # all-ones pred → no predicted negatives → NPV vacuously 1 (paper Corollary 2),
        # regardless of y_true. This is the convention the published results use.
        Y_true, Y_pred = all_ones
        assert EM.negative_predictive_value(Y_true, Y_pred) == pytest.approx(1.0)

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
# informedness  — example-based F_Inf = 1/2 (Specificity + Recall) (paper eq.)
# ──────────────────────────────────────────────

class TestFInformedness:
    def test_perfect(self, perfect):
        assert EM.informedness(*perfect) == pytest.approx(1.0)

    def test_worst(self, worst):
        assert EM.informedness(*worst) == pytest.approx(0.0)

    def test_all_ones_informedness(self, all_ones):
        # all-ones pred: Recall = 1 (all true pos captured), Specificity = 0 (no TN
        # since nothing predicted negative). F_Inf = 0.5*(0+1) = 0.5 per sample.
        assert EM.informedness(*all_ones) == pytest.approx(0.5)

    def test_example_based_known_case(self):
        # Example-based: mean over samples of 0.5*(specificity + recall).
        Y_true = np.array([[1, 1, 0], [1, 0, 0]])
        Y_pred = np.array([[1, 0, 0], [0, 1, 0]])
        # sample 0: recall = 1/2, spec = 1   → 0.75
        # sample 1: recall = 0,   spec = 1/2 → 0.25
        assert EM.informedness(Y_true, Y_pred) == pytest.approx(0.5)


# ──────────────────────────────────────────────
# macro_f1 / micro_f1
# ──────────────────────────────────────────────

class TestMacroMicroF1:
    def test_perfect(self, perfect):
        assert EM.macro_f1(*perfect) == pytest.approx(1.0)
        assert EM.micro_f1(*perfect) == pytest.approx(1.0)

    def test_worst(self, worst):
        assert EM.macro_f1(*worst) == pytest.approx(0.0)
        assert EM.micro_f1(*worst) == pytest.approx(0.0)

    def test_all_zero(self):
        Y = np.zeros((3, 3), dtype=int)
        assert EM.macro_f1(Y, Y) == pytest.approx(1.0)
        assert EM.micro_f1(Y, Y) == pytest.approx(1.0)

    def test_known_case_macro_differs_from_micro(self):
        # Y_true col sums: [2,2,1]; Y_pred col sums: [1,1,0]
        # Per-label F1: [2/3, 2/3, 0] → macro = 4/9
        # Global: TP=2, FP=0, FN=3 → micro = 4/7
        Y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        Y_pred = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        assert EM.macro_f1(Y_true, Y_pred) == pytest.approx(4 / 9)
        assert EM.micro_f1(Y_true, Y_pred) == pytest.approx(4 / 7)


# ──────────────────────────────────────────────
# markedness  (was buggy: precision=1 when pred=0,true>0)
# ──────────────────────────────────────────────

class TestFMarkedness:
    def test_perfect(self, perfect):
        assert EM.markedness(*perfect) == pytest.approx(1.0)

    def test_worst(self, worst):
        assert EM.markedness(*worst) == pytest.approx(0.0)

    def test_all_zeros_vacuous_precision(self, all_zeros):
        # Paper convention: pred=0_K → internal precision = 1 (vacuous, Fpre(.,0_K)=1).
        # markedness = 0.5*(NPV + 1); NPV = mean fraction of true negatives.
        Y_true, Y_pred = all_zeros
        n = Y_true.shape[1]
        npv_expected = np.mean(np.sum(1 - Y_true, axis=1) / n)
        expected = 0.5 * (npv_expected + 1.0)
        assert EM.markedness(Y_true, Y_pred) == pytest.approx(expected)

    def test_internal_precision_differs_from_precision_score(self):
        # markedness's internal precision uses the paper's vacuous convention
        # (Fpre(.,0_K)=1), which DIFFERS from standalone precision_score (sklearn).
        Y_true = np.array([[1, 0], [1, 1]])
        Y_pred = np.array([[0, 0], [0, 1]])
        # standalone precision: sample0 (empty pred, true>0) → 0; sample1 → 1 → mean 0.5
        assert EM.precision_score(Y_true, Y_pred) == pytest.approx(0.5)
        # markedness internal precision: sample0 → 1 (vacuous), sample1 → 1 → mean 1.0
        # markedness internal NPV:        sample0 → 0.5,        sample1 → 0.0 → mean 0.25
        # markedness = 0.5 * (0.25 + 1.0) = 0.625
        assert EM.markedness(Y_true, Y_pred) == pytest.approx(0.625)


# ──────────────────────────────────────────────
# Ranking metrics: one_error, coverage, ranking_loss, average_precision
# All take continuous scores (not binary preds). Higher = better.
# ──────────────────────────────────────────────

class TestRankingMetrics:
    def test_perfect_single_positive(self):
        Y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        S = Y.astype(float)
        assert EM.one_error_score(Y, S) == pytest.approx(1.0)
        assert EM.coverage_score(Y, S) == pytest.approx(1.0)
        assert EM.ranking_loss_score(Y, S) == pytest.approx(1.0)
        assert EM.average_precision_score(Y, S) == pytest.approx(1.0)

    def test_worst_single_positive(self):
        Y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        S = (1 - Y).astype(float)
        assert EM.one_error_score(Y, S) == pytest.approx(0.0)

    def test_known_case_correct_top(self):
        # Top-1 is true positive; pos ranks [1,2] of L=4 → cov norm = 1/3
        Y = np.array([[1, 0, 1, 0]])
        S = np.array([[0.9, 0.1, 0.5, 0.2]])
        assert EM.one_error_score(Y, S) == pytest.approx(1.0)
        assert EM.coverage_score(Y, S) == pytest.approx(2 / 3)
        assert EM.ranking_loss_score(Y, S) == pytest.approx(1.0)
        assert EM.average_precision_score(Y, S) == pytest.approx(1.0)

    def test_known_case_misranked(self):
        # Pos ranks [4, 2] of L=4 → cov norm=1
        # Pair misorder: 3/4 → RL_score=0.25
        # AP: sorted pos ranks [2,4] → mean(1/2, 2/4) = 0.5
        Y = np.array([[1, 0, 1, 0]])
        S = np.array([[0.1, 0.9, 0.5, 0.2]])
        assert EM.one_error_score(Y, S) == pytest.approx(0.0)
        assert EM.coverage_score(Y, S) == pytest.approx(0.0)
        assert EM.ranking_loss_score(Y, S) == pytest.approx(0.25)
        assert EM.average_precision_score(Y, S) == pytest.approx(0.5)

    def test_all_negative_sample_is_vacuous(self):
        # No positive labels — should not crash; AP=1, coverage perfect.
        Y = np.array([[0, 0, 0]])
        S = np.array([[0.1, 0.2, 0.3]])
        assert EM.coverage_score(Y, S) == pytest.approx(1.0)
        assert EM.ranking_loss_score(Y, S) == pytest.approx(1.0)
        assert EM.average_precision_score(Y, S) == pytest.approx(1.0)
