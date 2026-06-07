"""DaCaF: Bayes-optimal inference for probabilistic multi-label classification.

Official implementation of the divide-and-conquer and fusion (DaCaF) approach
described in Nguyen et al., *Information Fusion* (2026),
https://doi.org/10.1016/j.inffus.2026.104517
"""

from dacaf_mlc.evaluation_metrics import EvaluationMetrics
from dacaf_mlc.probability_classifier_chains import (
    ProbabilisticClassifierChain,
    ProbabilisticClassifierChainCustom,  # deprecated alias, removed in v2.0
)

__version__ = "1.2.0"

__all__ = [
    "ProbabilisticClassifierChain",
    "ProbabilisticClassifierChainCustom",  # deprecated alias
    "EvaluationMetrics",
    "__version__",
]
