"""The registry's (bop, needs) entries must match the public predict_* methods."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from dacaf_mlc.probability_classifier_chains import ProbabilisticClassifierChain
from dacaf_mlc.metrics_registry import PREDICT_FUNCTIONS

# Map each registry rule to the equivalent public method for cross-checking.
RULE_TO_METHOD = {
    "Predict Hamming": "predict_hamming",
    "Predict Subset": "predict_subset",
    "Predict Precision": "predict_precision",
    "Predict NPV": "predict_npv",
    "Predict Recall": "predict_recall",
    "Predict Markedness": "predict_markedness",
    "Predict Fmeasure": "predict_fmeasure",
}


def _fit(L=5, D=6, n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, D))
    Y = (rng.random((n, L)) > 0.5).astype(int)
    pcc = ProbabilisticClassifierChain(LogisticRegression(max_iter=2000, random_state=seed))
    pcc.fit(X, Y)
    return pcc, X


@pytest.mark.parametrize("pf", PREDICT_FUNCTIONS, ids=lambda pf: pf["name"])
def test_registry_bop_matches_method(pf):
    pcc, X = _fit()
    stats = pcc.compute_stats(X, needs=pf["needs"])
    via_registry = pf["bop"](stats)
    via_method = getattr(pcc, RULE_TO_METHOD[pf["name"]])(X)
    np.testing.assert_array_equal(via_registry, via_method)
