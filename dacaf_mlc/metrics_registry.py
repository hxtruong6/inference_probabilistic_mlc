"""Registry mapping inference rules to the metrics evaluated on their output.

- BINARY_METRICS    : the seven paper metrics, applied to each rule's output.
- PREDICT_FUNCTIONS : the inference rules to run. Each entry bundles the pure
                      Bayes-optimal predictor (``bop``), the minimal set of
                      statistics it needs from ``compute_stats`` (``needs``),
                      and the metric list to score it with.

Together these produce the paper's 7×7 target-metric × evaluation-metric table.
To add a metric: append an EvaluationMetrics method to BINARY_METRICS (to score
it everywhere) and/or a new bop_* rule entry here (to add a target column).
"""
from dacaf_mlc.evaluation_metrics import EvaluationMetrics
from dacaf_mlc.probability_classifier_chains import (
    bop_fmeasure,
    bop_hamming,
    bop_markedness,
    bop_npv,
    bop_precision,
    bop_recall,
    bop_subset,
)

BINARY_METRICS = [
    {"name": "Hamming Accuracy",          "func": EvaluationMetrics.hamming_accuracy},
    {"name": "Subset Accuracy",            "func": EvaluationMetrics.subset_accuracy},
    {"name": "Precision Score",            "func": EvaluationMetrics.precision_score},
    {"name": "Negative Predictive Value",  "func": EvaluationMetrics.negative_predictive_value},
    {"name": "Recall Score",               "func": EvaluationMetrics.recall_score},
    {"name": "Markedness",                 "func": EvaluationMetrics.markedness},
    {"name": "Fmeasure Score",             "func": EvaluationMetrics.f_beta},
]

PREDICT_FUNCTIONS = [
    {"name": "Predict Hamming",    "bop": bop_hamming,    "needs": {"marginal"}, "metrics": BINARY_METRICS},
    {"name": "Predict Subset",     "bop": bop_subset,     "needs": {"map"},      "metrics": BINARY_METRICS},
    {"name": "Predict Precision",  "bop": bop_precision,  "needs": {"marginal"}, "metrics": BINARY_METRICS},
    {"name": "Predict NPV",        "bop": bop_npv,        "needs": set(),        "metrics": BINARY_METRICS},
    {"name": "Predict Recall",     "bop": bop_recall,     "needs": set(),        "metrics": BINARY_METRICS},
    {"name": "Predict Markedness", "bop": bop_markedness, "needs": {"marginal"}, "metrics": BINARY_METRICS},
    {"name": "Predict Fmeasure",   "bop": bop_fmeasure,   "needs": {"pairwise"}, "metrics": BINARY_METRICS},
]
