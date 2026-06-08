"""The vectorized bop_markedness must match the reference loop exactly."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from dacaf_mlc.probability_classifier_chains import (
    ProbabilisticClassifierChain,
    bop_markedness,
    _bop_markedness_loop,
)


def _fit(L, D=6, n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, D))
    Y = (rng.random((n, L)) > 0.5).astype(int)
    pcc = ProbabilisticClassifierChain(LogisticRegression(max_iter=2000, random_state=seed))
    pcc.fit(X, Y)
    return pcc, X


@pytest.mark.parametrize("L", [1, 2, 3, 4, 6, 8])
def test_vectorized_matches_loop(L):
    pcc, X = _fit(L, seed=L)
    stats = pcc.compute_stats(X, needs={"marginal"})
    np.testing.assert_array_equal(bop_markedness(stats), _bop_markedness_loop(stats))
