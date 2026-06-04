"""Registry mapping inference rules to the metrics evaluated on their output.

- BINARY_METRICS    : the seven paper metrics, applied to each predict_* output.
- PREDICT_FUNCTIONS : the inference rules to run, each with its metric list.

Together these produce the paper's 7×7 target-metric × evaluation-metric table.
"""
from dacaf_mlc.evaluation_metrics import EvaluationMetrics

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
    {"name": "Predict Hamming",      "func": "predict_hamming",         "metrics": BINARY_METRICS},
    {"name": "Predict Subset",       "func": "predict_subset",          "metrics": BINARY_METRICS},
    {"name": "Predict Precision",    "func": "predict_precision",       "metrics": BINARY_METRICS},
    {"name": "Predict NPV",          "func": "predict_npv",             "metrics": BINARY_METRICS},
    {"name": "Predict Recall",       "func": "predict_recall",          "metrics": BINARY_METRICS},
    {"name": "Predict Markedness",   "func": "predict_markedness",      "metrics": BINARY_METRICS},
    {"name": "Predict Fmeasure",     "func": "predict_fmeasure",        "metrics": BINARY_METRICS},
]
