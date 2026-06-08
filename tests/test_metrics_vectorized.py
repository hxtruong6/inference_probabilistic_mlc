"""Vectorized example-based metrics must equal the per-sample loop, bit-for-bit.

Inputs are binary, so every reduction is an exact integer sum — vectorization
must produce identical floats, not merely close ones.
"""
import numpy as np
import pytest

from dacaf_mlc.evaluation_metrics import EvaluationMetrics as EM


# --- independent loop references (the pre-vectorization semantics) -----------

def _precision_loop(yt, yp):
    out = np.zeros(len(yt))
    for i in range(len(yt)):
        st, sp = yt[i].sum(), yp[i].sum()
        out[i] = 1 if (st == 0 and sp == 0) else (0 if sp == 0 else np.dot(yt[i], yp[i]) / sp)
    return out.mean()


def _recall_loop(yt, yp):
    out = np.zeros(len(yt))
    for i in range(len(yt)):
        st = yt[i].sum()
        out[i] = (1 if yp[i].sum() == 0 else 0) if st == 0 else np.dot(yt[i], yp[i]) / st
    return out.mean()


def _npv_loop(yt, yp):
    L = yt.shape[1]
    out = np.zeros(len(yt))
    for i in range(len(yt)):
        sp = yp[i].sum()
        out[i] = 1 if sp == L else np.dot(1 - yt[i], 1 - yp[i]) / np.sum(1 - yp[i])
    return out.mean()


def _fbeta_loop(yt, yp, beta=1):
    out = np.zeros(len(yt))
    for i in range(len(yt)):
        denom = (beta**2) * yt[i].sum() + yp[i].sum()
        out[i] = 1 if denom == 0 else (1 + beta**2) * np.dot(yt[i], yp[i]) / denom
    return out.mean()


def _markedness_loop(yt, yp):
    L = yt.shape[1]
    out = np.zeros(len(yt))
    for i in range(len(yt)):
        sp = yp[i].sum()
        npv = 1 if sp == L else np.dot(1 - yt[i], 1 - yp[i]) / np.sum(1 - yp[i])
        prec = 1 if sp == 0 else np.dot(yt[i], yp[i]) / sp
        out[i] = 0.5 * (npv + prec)
    return out.mean()


CASES = [
    ("random", lambda rng, L, n: ((rng.random((n, L)) > 0.5).astype(int),
                                  (rng.random((n, L)) > 0.5).astype(int))),
    ("all_zero_pred", lambda rng, L, n: ((rng.random((n, L)) > 0.5).astype(int),
                                         np.zeros((n, L), dtype=int))),
    ("all_one_pred", lambda rng, L, n: ((rng.random((n, L)) > 0.5).astype(int),
                                        np.ones((n, L), dtype=int))),
    ("all_zero_true", lambda rng, L, n: (np.zeros((n, L), dtype=int),
                                         (rng.random((n, L)) > 0.5).astype(int))),
    ("float_pred", lambda rng, L, n: ((rng.random((n, L)) > 0.5).astype(int),
                                      (rng.random((n, L)) > 0.5).astype(float))),
]


@pytest.mark.parametrize("name,gen", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("metric,ref", [
    (EM.precision_score, _precision_loop),
    (EM.recall_score, _recall_loop),
    (EM.negative_predictive_value, _npv_loop),
    (EM.f_beta, _fbeta_loop),
    (EM.markedness, _markedness_loop),
], ids=["precision", "recall", "npv", "fbeta", "markedness"])
def test_vectorized_metric_matches_loop(name, gen, metric, ref):
    rng = np.random.default_rng(abs(hash(name)) % 1000)
    for L in (3, 6, 14):
        yt, yp = gen(rng, L, 40)
        assert metric(yt, yp) == ref(yt, yp), f"{name} L={L}"
