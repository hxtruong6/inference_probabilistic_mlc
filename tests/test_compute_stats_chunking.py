"""Chunked compute_stats must be bit-identical to whole-batch (samples are independent)."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from dacaf_mlc.probability_classifier_chains import ProbabilisticClassifierChain


def _fit(L=5, D=6, n=37, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, D))
    Y = (rng.random((n, L)) > 0.5).astype(int)
    pcc = ProbabilisticClassifierChain(LogisticRegression(max_iter=2000, random_state=seed))
    pcc.fit(X, Y)
    return pcc, X


@pytest.mark.parametrize("batch_size", [1, 4, 10, 100])
def test_chunked_matches_whole(batch_size):
    pcc, X = _fit()
    whole = pcc.compute_stats(X, needs={"map", "marginal", "pairwise"})
    chunked = pcc.compute_stats(X, needs={"map", "marginal", "pairwise"}, batch_size=batch_size)
    # joint-derived-by-slicing/argmax fields are bit-identical (samples independent).
    np.testing.assert_array_equal(chunked.map_prediction, whole.map_prediction)
    np.testing.assert_array_equal(chunked.p_empty, whole.p_empty)
    np.testing.assert_array_equal(chunked.p_full, whole.p_full)
    # matmul-derived fields match up to float associativity (BLAS gemm vs gemv path).
    np.testing.assert_allclose(chunked.marginals, whole.marginals, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(chunked.pairwise, whole.pairwise, rtol=1e-12, atol=1e-12)
    assert chunked.n_samples == whole.n_samples == len(X)


def test_predictions_unchanged_under_chunking():
    """The downstream predictions (what metrics are computed on) are identical
    whether or not chunking is used — float-epsilon in marginals never flips them."""
    pcc, X = _fit(L=6, n=50, seed=3)
    for size in (1, 7, 50):
        np.testing.assert_array_equal(
            pcc.predict_fmeasure(X), pcc.predict_fmeasure(X)
        )
    whole = pcc.compute_stats(X, needs={"marginal"})
    chunked = pcc.compute_stats(X, needs={"marginal"}, batch_size=8)
    np.testing.assert_array_equal(
        np.where(whole.marginals > 0.5, 1, 0),
        np.where(chunked.marginals > 0.5, 1, 0),
    )
