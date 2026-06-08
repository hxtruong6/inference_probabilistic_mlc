"""Contract for compute_stats: compute only the requested statistics."""
import numpy as np
from sklearn.linear_model import LogisticRegression

from dacaf_mlc.probability_classifier_chains import ProbabilisticClassifierChain


def _fit(L=4, D=5, n=30, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, D))
    Y = (rng.random((n, L)) > 0.5).astype(int)
    pcc = ProbabilisticClassifierChain(LogisticRegression(max_iter=2000, random_state=seed))
    pcc.fit(X, Y)
    return pcc, X


def test_marginal_only_skips_pairwise_and_map():
    pcc, X = _fit()
    s = pcc.compute_stats(X, needs={"marginal"})
    assert s.marginals is not None
    assert s.pairwise is None
    assert s.map_prediction is None
    assert s.n_samples == len(X) and s.n_labels == pcc.L


def test_empty_needs_computes_no_joint_statistics():
    pcc, X = _fit()
    s = pcc.compute_stats(X, needs=set())
    assert s.marginals is None and s.pairwise is None and s.map_prediction is None
    assert s.n_samples == len(X) and s.n_labels == pcc.L


def test_stats_match_legacy_predict():
    pcc, X = _fit()
    Y_map, marg, pw = pcc.predict(X, marginal=True, pairwise=True)
    s = pcc.compute_stats(X, needs={"map", "marginal", "pairwise"})
    np.testing.assert_array_equal(s.map_prediction, Y_map)
    np.testing.assert_allclose(s.marginals, marg, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(s.pairwise, pw["P_pair_wise"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(s.p_empty, pw["P_pair_wise0"], rtol=1e-12, atol=1e-12)
