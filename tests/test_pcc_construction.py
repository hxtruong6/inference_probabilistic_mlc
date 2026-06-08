"""Construction contracts for ProbabilisticClassifierChain."""
from sklearn.linear_model import LogisticRegression

from dacaf_mlc.probability_classifier_chains import ProbabilisticClassifierChain


def test_default_base_estimator_not_shared_between_instances():
    """Each instance must get its own base estimator, not a shared module-level
    default object (the classic mutable-default-argument trap)."""
    a = ProbabilisticClassifierChain()
    b = ProbabilisticClassifierChain()
    assert a.base_estimator is not b.base_estimator


def test_default_base_estimator_is_logistic_regression():
    assert isinstance(ProbabilisticClassifierChain().base_estimator, LogisticRegression)
