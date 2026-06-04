"""Brute-force optimality tests for the Bayes-optimal inference rules.

For a fitted PCC and a query x, the exact joint P(y | x) is computed over all
2^L label vectors. Each predict_* rule must attain the maximum *expected*
value of its target metric over all 2^L candidate predictions, where the
expectation uses the paper's metric conventions (EvaluationMetrics).

This is the ground-truth check that the closed-form rules optimise the right
quantity. Run with: python -m pytest tests/test_inference_optimality.py -v
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.probability_classifier_chains import (
    ProbabilisticClassifierChainCustom,
    joint_probability,
)
from src.evaluation_metrics import EvaluationMetrics as EM


def _fit_small_pcc(L, N=30, D=5, seed=0):
    """Fit a PCC on small random (but learnable) data. cache_key stays None so
    each predict_* call recomputes for the given X (no stale cache)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(N, D)
    logits = X[:, :L] if D >= L else rng.randn(N, L)
    Y = (rng.rand(N, L) < 1.0 / (1.0 + np.exp(-logits))).astype(int)
    cc = ProbabilisticClassifierChainCustom(
        base_estimator=LogisticRegression(max_iter=2000)
    )
    cc.fit(X, Y)
    return cc, X


def _joint(cc, x, L):
    """Return (vecs (2^L, L) int, P (2^L,)) — full joint P(y|x) under cc."""
    K = 1 << L
    vecs = np.array(
        [list(map(int, np.binary_repr(b, width=L))) for b in range(K)], dtype=int
    )
    P = np.array([joint_probability(vecs[b], x, cc) for b in range(K)])
    return vecs, P


def _expected(vecs, P, yhat, metric_fn):
    """E[ metric(y, yhat) ] = Σ_y P(y) · metric(y, yhat)."""
    yh = np.asarray(yhat, dtype=int)[None, :]
    return sum(P[b] * metric_fn(vecs[b][None, :], yh) for b in range(len(vecs)))


def _assert_optimal(cc, X, L, predict_fn, metric_fn):
    """predict_fn(X) must achieve, for every sample, the best possible
    expected metric over all 2^L candidate predictions."""
    preds = predict_fn(X)
    for n in range(len(X)):
        vecs, P = _joint(cc, X[n], L)
        best = max(_expected(vecs, P, vecs[c], metric_fn) for c in range(len(vecs)))
        got = _expected(vecs, P, preds[n].astype(int), metric_fn)
        assert got == pytest.approx(best, abs=1e-9), (
            f"sample {n}: rule achieved {got:.6f} but optimum is {best:.6f}"
        )


L = 4
NTEST = 6  # samples checked per rule (brute force is O(2^L · 2^L) per sample)


def test_hamming_optimal():
    cc, X = _fit_small_pcc(L, seed=1)
    _assert_optimal(cc, X[:NTEST], L, cc.predict_hamming, EM.hamming_accuracy)


def test_subset_optimal():
    cc, X = _fit_small_pcc(L, seed=2)
    _assert_optimal(cc, X[:NTEST], L, cc.predict_subset, EM.subset_accuracy)


def test_recall_optimal():
    cc, X = _fit_small_pcc(L, seed=3)
    _assert_optimal(cc, X[:NTEST], L, cc.predict_recall, EM.recall_score)


def test_npv_optimal_and_all_ones():
    cc, X = _fit_small_pcc(L, seed=4)
    assert np.all(cc.predict_npv(X) == 1), "NPV BOP must be the all-ones vector"
    _assert_optimal(cc, X[:NTEST], L, cc.predict_npv, EM.negative_predictive_value)


def test_markedness_optimal():
    cc, X = _fit_small_pcc(L, seed=5)
    _assert_optimal(cc, X[:NTEST], L, cc.predict_markedness, EM.markedness)


def test_fmeasure_optimal():
    cc, X = _fit_small_pcc(L, seed=6)
    _assert_optimal(
        cc, X[:NTEST], L, lambda x: cc.predict_fmeasure(x, beta=1),
        lambda yt, yp: EM.f_beta(yt, yp, beta=1),
    )


@pytest.mark.parametrize("seed", [8, 9, 11])
def test_informedness_optimal(seed):
    # The per-label-threshold rule must attain the true argmax of E[F_Inf] over
    # all 2^L predictions. (The paper appendix's size-l/3-candidate rule does
    # NOT — this test is what flags that; hence the corrected derivation in code.)
    cc, X = _fit_small_pcc(L, seed=seed)
    _assert_optimal(cc, X[:12], L, cc.predict_informedness, EM.informedness)


def test_precision_predicts_single_top_label():
    # Paper Corollary 1: the (table-reproducing) precision BOP predicts exactly
    # one label, the highest-marginal one. (The standalone precision metric uses
    # the sklearn convention, under which the single-top-label rule is used.)
    cc, X = _fit_small_pcc(L, seed=7)
    pred = cc.predict_precision(X)
    assert np.all(pred.sum(axis=1) == 1)
    _, marg, _ = cc.predict(X, marginal=True)
    assert np.array_equal(pred.argmax(axis=1), marg.argmax(axis=1))
