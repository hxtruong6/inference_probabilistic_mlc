"""DaCaF: Bayes-optimal inference for probabilistic multi-label classification.

Official implementation of the divide-and-conquer and fusion (DaCaF) approach
described in Nguyen et al., *Information Fusion* (2026),
https://doi.org/10.1016/j.inffus.2026.104517
"""

from dacaf_mlc.evaluation_metrics import EvaluationMetrics
from dacaf_mlc.probability_classifier_chains import ProbabilisticClassifierChainCustom

__version__ = "1.1.0"

__all__ = [
    "ProbabilisticClassifierChainCustom",
    "EvaluationMetrics",
    "__version__",
]
