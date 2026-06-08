"""Verify optimized predict() matches brute-force reference numerically.

The optimized prefix-tree batched predict() must produce IDENTICAL outputs
(up to float64 precision) to the reference per-sample/per-combo implementation.
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from dacaf_mlc.probability_classifier_chains import ProbabilisticClassifierChain


def _fit_pcc(L, D=4, n_samples=50, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, D))
    Y = (rng.random((n_samples, L)) > 0.5).astype(int)
    pcc = ProbabilisticClassifierChain(LogisticRegression(max_iter=2000, random_state=seed))
    pcc.fit(X, Y)
    return pcc, X


@pytest.mark.parametrize("L", [2, 3, 5, 7])
def test_predict_matches_reference(L):
    pcc, X_train = _fit_pcc(L=L, n_samples=30)
    X_test = X_train[:10]

    Y_opt, M_opt, pw_opt = pcc.predict(X_test, marginal=True, pairwise=True)
    Y_ref, M_ref, pw_ref = pcc._predict_reference(X_test, marginal=True, pairwise=True)

    np.testing.assert_allclose(Y_opt, Y_ref, rtol=0, atol=0,
                               err_msg=f"L={L}: Y_pred mismatch")
    np.testing.assert_allclose(M_opt, M_ref, rtol=1e-10, atol=1e-12,
                               err_msg=f"L={L}: marginal mismatch")
    np.testing.assert_allclose(pw_opt["P_pair_wise"], pw_ref["P_pair_wise"],
                               rtol=1e-10, atol=1e-12,
                               err_msg=f"L={L}: pairwise mismatch")
    np.testing.assert_allclose(pw_opt["P_pair_wise0"], pw_ref["P_pair_wise0"],
                               rtol=1e-10, atol=1e-12,
                               err_msg=f"L={L}: P_pw0 mismatch")
    np.testing.assert_allclose(pw_opt["P_pair_wise1"], pw_ref["P_pair_wise1"],
                               rtol=1e-10, atol=1e-12,
                               err_msg=f"L={L}: P_pw1 mismatch")


def test_joint_p_sums_to_one():
    """Sanity: marginal sum + complement should give total probability ≈ 1."""
    pcc, X_train = _fit_pcc(L=4, n_samples=20)
    _, _, pw = pcc.predict(X_train[:5], pairwise=True)
    # Σ_s P(|y|=s | x) over s = 0..L should be 1.
    # P(|y|=0) = P_pair_wise0; P(|y|=s>0) = Σ_j P_pair[j, s] / s
    L = pcc.L
    for n in range(5):
        total = float(pw["P_pair_wise0"][n, 0])
        for s in range(1, L + 1):
            total += pw["P_pair_wise"][n, :, s].sum() / s
        assert abs(total - 1.0) < 1e-9, f"sample {n}: total = {total}"
