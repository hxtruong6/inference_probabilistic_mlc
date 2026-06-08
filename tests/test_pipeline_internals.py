"""Unit tests for pipeline helper behavior (logging / error handling contracts)."""
import numpy as np
import pytest

from dacaf_mlc.pipeline import calculate_metrics


def test_calculate_metrics_propagates_metric_errors():
    """A metric that raises must NOT be silently swallowed; the error propagates
    so a broken metric can never produce a silently-incomplete result table."""
    Y_true = np.array([[1, 0], [0, 1]])
    Y_pred = np.array([[1, 0], [0, 1]])

    def boom(y_true, y_pred):
        raise RuntimeError("metric is broken")

    bad_metric = [{"name": "Boom", "func": boom}]
    with pytest.raises(RuntimeError, match="metric is broken"):
        calculate_metrics(Y_true, Y_pred, bad_metric)
