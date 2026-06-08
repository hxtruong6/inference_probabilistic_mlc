"""Registry of inference rules and the metrics evaluated on their output.

- ``EvalMetric``      : a named evaluation metric (``func(Y_true, Y_pred)``).
- ``InferenceRule``   : a Bayes-optimal rule bundling its pure predictor
                        (``bop``), the minimal statistics it needs from
                        ``compute_stats`` (``needs``), and the metrics to score
                        it with.
- ``BINARY_METRICS``    : the seven paper metrics, applied to every rule's output.
- ``PREDICT_FUNCTIONS`` : the seven inference rules.

Together these produce the paper's 7×7 target-metric × evaluation-metric table.
To add a metric: append an ``EvalMetric`` to ``BINARY_METRICS`` (to score it
everywhere) and/or an ``InferenceRule`` here (to add a target column).
"""
from collections.abc import Callable
from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class EvalMetric:
    """An evaluation metric scored as ``func(Y_true, Y_pred, **options)``."""

    name: str
    func: Callable
    options: dict = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceRule:
    """A Bayes-optimal rule: pure ``bop`` predictor, its ``needs``, and its metrics."""

    name: str
    bop: Callable
    needs: frozenset
    metrics: tuple


BINARY_METRICS = (
    EvalMetric("Hamming Accuracy",          EvaluationMetrics.hamming_accuracy),
    EvalMetric("Subset Accuracy",           EvaluationMetrics.subset_accuracy),
    EvalMetric("Precision Score",           EvaluationMetrics.precision_score),
    EvalMetric("Negative Predictive Value", EvaluationMetrics.negative_predictive_value),
    EvalMetric("Recall Score",              EvaluationMetrics.recall_score),
    EvalMetric("Markedness",                EvaluationMetrics.markedness),
    EvalMetric("Fmeasure Score",            EvaluationMetrics.f_beta),
)

PREDICT_FUNCTIONS = (
    InferenceRule("Predict Hamming",    bop_hamming,    frozenset({"marginal"}), BINARY_METRICS),
    InferenceRule("Predict Subset",     bop_subset,     frozenset({"map"}),      BINARY_METRICS),
    InferenceRule("Predict Precision",  bop_precision,  frozenset({"marginal"}), BINARY_METRICS),
    InferenceRule("Predict NPV",        bop_npv,        frozenset(),             BINARY_METRICS),
    InferenceRule("Predict Recall",     bop_recall,     frozenset(),             BINARY_METRICS),
    InferenceRule("Predict Markedness", bop_markedness, frozenset({"marginal"}), BINARY_METRICS),
    InferenceRule("Predict Fmeasure",   bop_fmeasure,   frozenset({"pairwise"}), BINARY_METRICS),
)
