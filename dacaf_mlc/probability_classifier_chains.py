import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression

from dacaf_mlc.skmultiflow.meta.classifier_chains import ClassifierChain


def joint_probability(y, x, cc, payoff=np.prod):
    """Compute the joint probability P(Y=y | X=x) under the classifier chain cc.

    Parameters
    ----------
    y : array-like of shape (L,)
        Candidate label vector.
    x : array-like of shape (D,)
        Input feature vector.
    cc : ClassifierChain
        Trained classifier chain with an `ensemble` attribute.
    payoff : callable
        Aggregation function over per-label probabilities (default: np.prod).

    Returns
    -------
    float
        Payoff (joint probability) of predicting y given x.
    """
    D = len(x)
    L = len(y)
    p = np.zeros(L)
    xy = np.zeros(D + L)
    xy[:D] = x.copy()

    for j in range(L):
        P_j = cc.ensemble[j].predict_proba(xy[: D + j].reshape(1, -1))[0]
        xy[D + j] = y[j]
        p[j] = P_j[y[j]]

    return payoff(p)


class ProbabilisticClassifierChain(ClassifierChain):
    """Probabilistic Classifier Chains (PCC) for multi-label learning.

    Extends ClassifierChain with Bayes-optimal inference rules for
    various loss functions (Hamming, Subset, Precision, Recall, NPV,
    Markedness, F-measure, Informedness).

    Parameters
    ----------
    base_estimator : sklearn estimator (default=LogisticRegression)
        Base classifier used for each label in the chain.
    order : str or None
        Label order. None = default; 'random' = random order.
    random_state : int or None
        Random seed.
    """

    def __init__(self, base_estimator=None, order=None, random_state=None):
        if base_estimator is None:
            base_estimator = LogisticRegression()
        super().__init__(base_estimator=base_estimator, order=order, random_state=random_state)
        self.cache_key: str | None = None
        self.prediction_cache: dict = {}

    def set_cache_key(self, key: str) -> None:
        self.cache_key = key
        self.prediction_cache[key] = None

    @staticmethod
    def _binary_matrix(K, L):
        """Return (K, L) int8 matrix where row k is binary repr of k with L bits, MSB-first."""
        if L == 0:
            return np.zeros((1, 0), dtype=np.int8)
        powers = (1 << np.arange(L - 1, -1, -1)).astype(np.int64)
        return ((np.arange(K, dtype=np.int64)[:, None] & powers[None, :]) > 0).astype(np.int8)

    def predict(self, X, marginal=False, pairwise=False) -> tuple[np.ndarray, np.ndarray, dict]:
        """Enumerate all 2^L label combinations and compute joint probabilities.

        Uses prefix-tree batched inference: for each chain level j, all
        N × 2^j partial label sequences are evaluated with one batched
        `predict_proba` call (instead of N × 2^L × L per-element calls).
        Numerically equivalent to brute-force enumeration but ~L × 2^L
        fewer estimator calls. Verified by `_predict_reference()` test.

        Layout convention: at depth L, the k-th (0 ≤ k < 2^L) probability for
        sample n corresponds to the label vector = binary repr of k with L
        bits, MSB-first (i.e. y[j] = (k >> (L-1-j)) & 1).

        Results are cached by cache_key so multiple predict_* calls on the
        same fold reuse the computed probabilities.

        Returns
        -------
        (Y_pred, P_margin_yi_1, pairwise_dict)
            Y_pred        : (N, L) — MAP prediction (argmax over joint).
            P_margin_yi_1 : (N, L) — marginal P(y_j=1 | x).
            pairwise_dict : 'P_pair_wise' (N, L, L+1), 'P_pair_wise0' (N, 1),
                            'P_pair_wise1' (N, 1).
        """
        if (
            self.cache_key is not None
            and self.prediction_cache.get(self.cache_key) is not None
        ):
            return self.prediction_cache[self.cache_key]

        N, D = X.shape
        L = self.L
        K = 1 << L  # 2^L

        # Sample-major prefix-tree expansion.
        # At depth j: joint_p has shape (N * 2^j,), sample-major layout.
        # Sample n's entries occupy positions [n * 2^j : (n+1) * 2^j),
        # and within that block the entry at index k_partial corresponds to
        # prefix = binary repr of k_partial with j bits (MSB-first).
        joint_p = np.ones(N, dtype=np.float64)

        for j in range(L):
            K_j = 1 << j  # 2^j
            # Build inputs for predict_proba: (N * K_j, D + j).
            # Each sample contributes K_j rows (one per partial prefix).
            x_rep = np.repeat(X, K_j, axis=0)  # (N * K_j, D), sample-major
            if j == 0:
                inputs = x_rep
            else:
                prefixes = self._binary_matrix(K_j, j).astype(np.float64)  # (K_j, j)
                prefix_rep = np.tile(prefixes, (N, 1))                     # (N * K_j, j)
                inputs = np.concatenate([x_rep, prefix_rep], axis=1)

            proba = self.ensemble[j].predict_proba(inputs)  # (N * K_j, 2)

            # Interleaved expansion: each entry → 2 entries (||0 at even, ||1 at odd).
            # Preserves sample-major layout: sample n's new block at [n*2^(j+1), (n+1)*2^(j+1)).
            new_joint = np.empty(N * K_j * 2, dtype=np.float64)
            new_joint[0::2] = joint_p * proba[:, 0]
            new_joint[1::2] = joint_p * proba[:, 1]
            joint_p = new_joint

        # joint_p shape: (N * 2^L,) sample-major → reshape (N, 2^L).
        joint_p = joint_p.reshape(N, K)

        # Build all label vectors once: (K, L) MSB-first binary repr.
        all_vecs = self._binary_matrix(K, L).astype(np.float64)
        s_vals = all_vecs.sum(axis=1).astype(np.int64)  # (K,) cardinality of each vec

        # MAP prediction
        idx_max = np.argmax(joint_p, axis=1)         # (N,)
        Y_pred = all_vecs[idx_max]                   # (N, L)

        # All-zero vec is at index 0 (binary repr 0...0), all-ones at K-1.
        P_pair_wise0 = joint_p[:, 0:1].copy()        # (N, 1)
        P_pair_wise1 = joint_p[:, K - 1:K].copy()    # (N, 1)

        # Always compute marginal AND pairwise (cheap once joint_p is computed)
        # so that the cached result satisfies any later marginal/pairwise request.
        # The `marginal` / `pairwise` flags exist for API compatibility only.
        P_margin_yi_1 = joint_p @ all_vecs           # (N, K) @ (K, L) → (N, L)
        P_pair_wise = np.zeros((N, L, L + 1))
        for s in range(L + 1):
            mask = s_vals == s
            if mask.any():
                P_pair_wise[:, :, s] = joint_p[:, mask] @ all_vecs[mask]

        result = (
            Y_pred,
            P_margin_yi_1,
            {
                "P_pair_wise": P_pair_wise,
                "P_pair_wise0": P_pair_wise0,
                "P_pair_wise1": P_pair_wise1,
            },
        )
        self.prediction_cache[self.cache_key] = result
        return result

    def _predict_reference(self, X, marginal=False, pairwise=False):
        """Brute-force reference implementation (slow). For testing only.

        Same return shape as predict() but computed via per-sample, per-combination,
        per-label predict_proba calls. Use to verify the optimized predict().
        """
        N, _ = X.shape
        L = self.L
        Y_pred = np.zeros((N, L))
        P_margin_yi_1 = np.zeros((N, L))
        P_pair_wise = np.zeros((N, L, L + 1))
        P_pair_wise0 = np.zeros((N, 1))
        P_pair_wise1 = np.zeros((N, 1))

        for n in range(N):
            w_max = -1.0
            for b in range(1 << L):
                y_ = np.array(list(map(int, np.binary_repr(b, width=L))))
                w_ = joint_probability(y_, X[n], self)
                s = int(y_.sum())
                if s == 0:
                    P_pair_wise0[n] = w_
                if s == L:
                    P_pair_wise1[n] = w_
                if marginal or pairwise:
                    for j in range(L):
                        if y_[j] == 1:
                            P_margin_yi_1[n, j] += w_
                            P_pair_wise[n, j, s] += w_
                if w_ > w_max:
                    Y_pred[n, :] = y_
                    w_max = w_
        return (
            Y_pred,
            P_margin_yi_1,
            {"P_pair_wise": P_pair_wise, "P_pair_wise0": P_pair_wise0, "P_pair_wise1": P_pair_wise1},
        )

    def predict_hamming(self, X):
        """Bayes-optimal predictor for Hamming Accuracy.

        Thresholds each marginal P(y_j=1|x) at 0.5.
        """
        _, P_margin_yi_1, _ = self.predict(X, marginal=True)
        return np.where(P_margin_yi_1 > 0.5, 1, 0)

    def predict_subset(self, X):
        """Bayes-optimal predictor for Subset Accuracy.

        Returns the joint MAP label vector (argmax over all 2^L combinations).
        """
        predictions, _, _ = self.predict(X)
        return predictions

    def predict_precision(self, X):
        """Bayes-optimal predictor for Precision.

        Predicts exactly one label: the one with the highest marginal probability.
        """
        N = X.shape[0]
        Y_pred = np.zeros((N, self.L))
        _, P_margin_yi_1, _ = self.predict(X, marginal=True)
        for n in range(N):
            Y_pred[n, np.argmax(P_margin_yi_1[n])] = 1
        return Y_pred

    def predict_npv(self, X):
        """Bayes-optimal predictor for Negative Predictive Value (NPV).

        Paper Corollary 2 (eq. negative): under the convention NPV(y, 1_K) = 1
        (predicting all-positive leaves no predicted negatives → vacuously
        perfect), the all-ones vector 1_K attains the maximum expected NPV and
        is the BOP whenever it is a valid prediction — which it always is in
        this setting. Hence the NPV BOP is trivial and coincides with the
        Recall BOP (this is why the F_neg and F_rec rows are identical in the
        paper's result tables).
        """
        return np.ones((X.shape[0], self.L))

    def predict_recall(self, X):
        """Bayes-optimal predictor for Recall.

        Predicting all labels as positive trivially achieves perfect recall
        (TP / (TP + FN) = 1 since FN = 0). This is the theoretical optimum
        when optimising recall alone without a precision constraint.
        """
        return np.ones((X.shape[0], self.L))

    def predict_markedness(self, X):
        """Bayes-optimal predictor for Markedness = 0.5 * (NPV + Precision).

        Conventions (match `EvaluationMetrics`):
            - Precision = 1 if pred=0 and true=0 (vacuous), 0 if pred=0 and true>0.
            - NPV      = 1 if pred=all-ones and true=all-ones (vacuous), 0 if
                         pred=all-ones and true has any 0.

        Paper Proposition 6 / Algorithm 4 conventions: Fpre(y, 0_K) = 1 and
        Fneg(y, 1_K) = 1 (vacuous precision / NPV). For candidate top-l
        (l ∈ {0,..,L}), expected markedness is:
            l = 0  : 0.5 * (1 + 1 - sum_p/L)        # NPV = 1 - sum_p/L, Precision = 1 (vacuous)
            0<l<L  : 0.5 * (A_l/l + 1 - (sum_p - A_l)/(L-l))
            l = L  : 0.5 * (sum_p/L + 1)            # Precision = sum_p/L, NPV = 1 (vacuous)
        where sum_p = Σ_j P(y_j=1|x) and A_l = sum of top-l marginals.
        Only the marginals p_j are needed (cf. Algorithm 4), in O(L log L).
        """
        N, _ = X.shape
        _, P_margin_yi_1, _ = self.predict(X, marginal=True)
        L = self.L

        indices = np.argsort(P_margin_yi_1, axis=1)[:, ::-1]
        Y_pred = np.zeros((N, L))

        for i in range(N):
            sum_p = float(np.sum(P_margin_yi_1[i]))

            E = np.zeros(L + 1)
            # Boundary cases use the paper's vacuous conventions (Fpre(.,0_K)=1, Fneg(.,1_K)=1).
            E[0] = 0.5 * (2.0 - sum_p / L)
            E[L] = 0.5 * (sum_p / L + 1.0)

            A_l = 0.0
            for l in range(1, L):
                A_l += P_margin_yi_1[i, indices[i, l - 1]]
                E[l] = 0.5 * (A_l / l + 1.0 - (sum_p - A_l) / (L - l))

            l_opt = int(np.argmax(E))
            for k in range(l_opt):
                Y_pred[i, indices[i, k]] = 1
        return Y_pred

    def _predict_fmeasure_loop(self, X, beta=1):
        """Reference (pre-vectorization) F-beta BOP. Kept for regression tests."""
        N, _ = X.shape
        _, _, pw = self.predict(X, pairwise=True)
        P_pair_wise = pw["P_pair_wise"]
        P_pair_wise0 = pw["P_pair_wise0"]

        Y_pred = np.zeros((N, self.L))
        for i in range(N):
            q = np.zeros((self.L, self.L))
            indices_desc = []
            E = np.zeros(self.L + 1)
            E0 = P_pair_wise0[i][0]

            for k in range(self.L):
                # k is 0-indexed; the prediction size is l = k + 1.
                # Paper Algorithm 1: q^beta_{l,k} = (1+β²) Σ_{s=1}^L P_{k,s} / (β²·s + l),
                # where P_{k,s} = P(y_label=1, |y|=s | x) = P_pair_wise[i][label][s].
                for label in range(self.L):
                    for s in range(1, self.L + 1):
                        q[k][label] += (1 + beta**2) * (
                            P_pair_wise[i][label][s]
                            / (beta**2 * s + (k + 1))
                        )
                indices_desc.append(np.argsort(q[k])[::-1].tolist())
                for i_ in range(k + 1):
                    E[k] += q[k][int(indices_desc[k][i_])]

            if E0 > np.max(E):
                Y_pred[i] = np.zeros(self.L)
            else:
                k_opt = np.argmax(E)
                for _l in range(k_opt + 1):
                    Y_pred[i][int(indices_desc[k_opt][_l])] = 1

        return Y_pred

    def predict_fmeasure(self, X, beta=1):
        """Bayes-optimal predictor for the F-beta measure (vectorized).

        Numerically equivalent to ``_predict_fmeasure_loop`` but replaces the
        O(N·L³) Python triple loop with tensor contractions. Paper Algorithm 1:
        for prediction size l (= k+1) the per-label score is
            q_{l,label} = (1+β²) Σ_{s=1}^L P(y_label=1, |y|=s) / (β²·s + l),
        the best size-l prediction is the top-l labels by q_{l,·}, and the
        expected F-beta of that prediction is the sum of its top-l q values.
        The all-zero prediction (expected F-beta = P(|y|=0)) wins iff it beats
        every nonempty size.
        """
        N, _ = X.shape
        L = self.L
        _, _, pw = self.predict(X, pairwise=True)
        P_pair_wise = pw["P_pair_wise"]          # (N, L, L+1): P(y_label=1, |y|=s)
        E0 = pw["P_pair_wise0"][:, 0]            # (N,): P(|y|=0)

        # Weight matrix W[l-1, s-1] = (1+β²) / (β²·s + l), for l, s ∈ {1..L}.
        s = np.arange(1, L + 1)
        l = np.arange(1, L + 1)
        W = (1 + beta**2) / (beta**2 * s[None, :] + l[:, None])  # (L, L)

        # q[n, label, l-1] = Σ_s P(y_label=1,|y|=s) · W[l-1, s-1]   (s from 1..L)
        Pe = P_pair_wise[:, :, 1:]               # (N, L, L) over s = 1..L
        q = np.einsum("nls,ks->nlk", Pe, W)      # (N, L_label, L_size)

        # E[n, k] = sum of the top-(k+1) label scores for size l=k+1.
        q_sorted_desc = np.sort(q, axis=1)[:, ::-1, :]     # sort labels descending
        cum = np.cumsum(q_sorted_desc, axis=1)             # (N, L, L_size)
        rank = np.arange(L)
        E_main = cum[:, rank, rank]                        # (N, L): E_main[:,k]=top-(k+1) sum
        E = np.concatenate([E_main, np.zeros((N, 1))], axis=1)  # (N, L+1), pad to match loop

        Y_pred = np.zeros((N, L))
        choose_empty = E0 > E.max(axis=1)
        k_opt = E.argmax(axis=1)                           # (N,)
        for i in range(N):
            if choose_empty[i]:
                continue
            order = np.argsort(q[i, :, k_opt[i]])[::-1]     # same tie order as the loop
            Y_pred[i, order[: k_opt[i] + 1]] = 1
        return Y_pred


class ProbabilisticClassifierChainCustom(ProbabilisticClassifierChain):
    """Deprecated alias for :class:`ProbabilisticClassifierChain`.

    Retained for backward compatibility with v1.1.x. It will be removed in
    v2.0; use :class:`ProbabilisticClassifierChain` instead.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ProbabilisticClassifierChainCustom is deprecated and will be removed "
            "in v2.0; use ProbabilisticClassifierChain instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
