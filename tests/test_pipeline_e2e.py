"""End-to-end pipeline test (safety net for the driver refactor).

Runs the real k-fold evaluation pipeline on a small synthetic dataset and checks
both that it produces the expected result structure and that the paper signature
holds end-to-end (optimising NPV and Recall yield identical scores, both with
NPV=Recall=1.0 on every fold).
"""
import numpy as np
import pandas as pd
import pytest

from dacaf_mlc.pipeline import (
    evaluate_kfold,
    prepare_model_to_evaluate,
    model_display_key,
)
from dacaf_mlc.metrics_registry import PREDICT_FUNCTIONS
from dacaf_mlc.arff_dataset import MultiLabelArffDataset


def _synth_handler(L=4, N=60, D=5, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(N, D)
    Y = (rng.rand(N, L) < 0.5).astype(int)
    # Ensure every sample has >=1 positive (so Recall on all-ones = 1) WITHOUT
    # making any label constant (LR needs both classes per label).
    for i in range(N):
        if Y[i].sum() == 0:
            Y[i, rng.randint(L)] = 1
    Xdf = pd.DataFrame(X, columns=[f"x{i}" for i in range(D)])
    Ydf = pd.DataFrame(Y, columns=[f"y{i}" for i in range(L)])
    return MultiLabelArffDataset("synth", X=Xdf, Y=Ydf)


def test_pipeline_kfold_smoke_and_paper_signature():
    h = _synth_handler()
    models = prepare_model_to_evaluate(estimator_names=["lr"], seed=1)
    idx = np.arange(len(h.X))
    res = evaluate_kfold(h, models, PREDICT_FUNCTIONS, idx[:48], idx[48:], 0)

    assert "synth" in res
    pcc_key = next(model_display_key(m) for m in models if "PCC" in model_display_key(m))
    npv_row = res["synth"][pcc_key]["Predict NPV"]
    rec_row = res["synth"][pcc_key]["Predict Recall"]
    # Optimising NPV and Recall both give the all-ones prediction → identical scores.
    assert npv_row == rec_row
    assert float(npv_row["Negative Predictive Value"]) == pytest.approx(1.0)
    assert float(npv_row["Recall Score"]) == pytest.approx(1.0)
