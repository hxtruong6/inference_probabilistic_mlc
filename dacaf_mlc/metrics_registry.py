"""Registry mapping inference rules to the metrics evaluated on their output.

- BINARY_METRICS  : applied to each binary predict_* output.
- RANKING_METRICS : applied to continuous marginal scores (model-level).
- PREDICT_FUNCTIONS : the inference rules to run, each with its metric list.
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
    {"name": "Informedness",               "func": EvaluationMetrics.informedness},
    {"name": "Macro F1",                   "func": EvaluationMetrics.macro_f1},
    {"name": "Micro F1",                   "func": EvaluationMetrics.micro_f1},
]

RANKING_METRICS = [
    {"name": "One-Error Score",       "func": EvaluationMetrics.one_error_score},
    {"name": "Coverage Score",         "func": EvaluationMetrics.coverage_score},
    {"name": "Ranking Loss Score",     "func": EvaluationMetrics.ranking_loss_score},
    {"name": "Average Precision",      "func": EvaluationMetrics.average_precision_score},
]

PREDICT_FUNCTIONS = [
    {"name": "Predict Hamming",      "func": "predict_hamming",         "metrics": BINARY_METRICS},
    {"name": "Predict Subset",       "func": "predict_subset",          "metrics": BINARY_METRICS},
    {"name": "Predict Precision",    "func": "predict_precision",       "metrics": BINARY_METRICS},
    {"name": "Predict NPV",          "func": "predict_npv",             "metrics": BINARY_METRICS},
    {"name": "Predict Recall",       "func": "predict_recall",          "metrics": BINARY_METRICS},
    {"name": "Predict Markedness",   "func": "predict_markedness",      "metrics": BINARY_METRICS},
    {"name": "Predict Fmeasure",     "func": "predict_fmeasure",        "metrics": BINARY_METRICS},
    {"name": "Predict Informedness", "func": "predict_informedness",    "metrics": BINARY_METRICS},
    {"name": "Marginal Scores",      "func": "predict_marginal_scores", "metrics": RANKING_METRICS},
]
