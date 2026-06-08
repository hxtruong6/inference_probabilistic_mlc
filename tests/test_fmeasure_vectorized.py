"""The vectorized predict_fmeasure must match the reference loop exactly."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from dacaf_mlc.probability_classifier_chains import ProbabilisticClassifierChain


def _fit(L, D=5, n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, D))
    Y = (rng.random((n, L)) > 0.5).astype(int)
    pcc = ProbabilisticClassifierChain(LogisticRegression(max_iter=2000, random_state=seed))
    pcc.fit(X, Y)
    return pcc, X


@pytest.mark.parametrize("L", [2, 3, 4, 6])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_vectorized_matches_loop(L, beta):
    pcc, X = _fit(L, seed=L)
    fast = pcc.predict_fmeasure(X, beta=beta)
    ref = pcc._predict_fmeasure_loop(X, beta=beta)
    np.testing.assert_array_equal(fast, ref)
