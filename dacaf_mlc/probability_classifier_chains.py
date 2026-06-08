"""Probabilistic Classifier Chains (PCC) and the DaCaF Bayes-optimal inference rules.

Implements Nguyen et al., *Information Fusion* (2026). A PCC estimates the joint
``P(y | x)`` over ``L`` binary labels; each ``bop_*`` rule then returns the
prediction ``ŷ`` that maximises the *expected* value of one target metric — its
Bayes-optimal prediction (BOP).

The paper's recipe is "divide-and-conquer and fusion" (DaCaF):

* **Divide & Conquer** — the ``2^L`` candidate predictions are partitioned into
  ``L+1`` groups by cardinality ``|ŷ|``; within a group the optimum is found by
  sorting labels by a score, and the global optimum is the best across groups.
* **Fusion** — those scores need marginal / pairwise probabilities, obtained by
  fusing the chain's per-label classifiers (ancestral sampling) into the joint
  ``P(y | x)`` — see :meth:`ProbabilisticClassifierChain._joint`.

Per-rule references: Hamming/Subset are standard; F-beta — Algorithm 1;
Markedness — Proposition 6 / Algorithm 4; NPV/Recall — Corollary 2; Precision —
Corollary 1. Metric conventions are documented in ``docs/CONVENTIONS.md``.
"""
import warnings
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from dacaf_mlc.skmultiflow.meta.classifier_chains import ClassifierChain


@dataclass
class InferenceStats:
    """Probabilistic statistics derived from the joint P(y | x) for a batch.

    Only the fields requested via ``compute_stats(..., needs=...)`` are filled;
    the rest stay ``None``. ``n_samples`` / ``n_labels`` are always set so that
    trivial rules (Recall, NPV) need no joint computation at all.
    """

    n_samples: int
    n_labels: int
    map_prediction: "np.ndarray | None" = None   # (N, L) argmax over the joint
    marginals: "np.ndarray | None" = None          # (N, L) P(y_j = 1 | x)
    pairwise: "np.ndarray | None" = None           # (N, L, L+1) P(y_j=1, |y|=s | x)
    p_empty: "np.ndarray | None" = None            # (N, 1) P(|y| = 0 | x)
    p_full: "np.ndarray | None" = None             # (N, 1) P(|y| = L | x)


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


# ---------------------------------------------------------------------------
# Bayes-optimal prediction (BOP) rules.
#
# Each rule is a PURE function of an InferenceStats object. To support a new
# metric, add one bop_* function here and one entry in metrics_registry — no
# changes to the model are required. The `needs` comment on each rule is the
# minimal statistic set its caller must request from compute_stats().
# ---------------------------------------------------------------------------

def bop_hamming(stats: InferenceStats) -> np.ndarray:  # needs: {"marginal"}
    """Hamming BOP: threshold each marginal P(y_j=1|x) at 0.5."""
    return np.where(stats.marginals > 0.5, 1, 0)


def bop_subset(stats: InferenceStats) -> np.ndarray:  # needs: {"map"}
    """Subset-0/1 BOP: the joint MAP label vector (argmax over all 2^L)."""
    return stats.map_prediction


def bop_precision(stats: InferenceStats) -> np.ndarray:  # needs: {"marginal"}
    """Precision BOP: predict exactly the single highest-marginal label."""
    N, L = stats.n_samples, stats.n_labels
    Y = np.zeros((N, L))
    Y[np.arange(N), stats.marginals.argmax(axis=1)] = 1
    return Y


def bop_npv(stats: InferenceStats) -> np.ndarray:  # needs: set()
    """NPV BOP: the all-ones vector (trivial; coincides with Recall, paper Cor. 2)."""
    return np.ones((stats.n_samples, stats.n_labels))


def bop_recall(stats: InferenceStats) -> np.ndarray:  # needs: set()
    """Recall BOP: the all-ones vector (FN = 0 → perfect recall)."""
    return np.ones((stats.n_samples, stats.n_labels))


def _bop_markedness_loop(stats: InferenceStats) -> np.ndarray:
    """Reference (pre-vectorization) markedness BOP. Kept for regression tests."""
    P = stats.marginals
    N, L = stats.n_samples, stats.n_labels
    indices = np.argsort(P, axis=1)[:, ::-1]
    Y_pred = np.zeros((N, L))
    for i in range(N):
        sum_p = float(np.sum(P[i]))
        E = np.zeros(L + 1)
        E[0] = 0.5 * (2.0 - sum_p / L)
        E[L] = 0.5 * (sum_p / L + 1.0)
        A_l = 0.0
        for l in range(1, L):
            A_l += P[i, indices[i, l - 1]]
            E[l] = 0.5 * (A_l / l + 1.0 - (sum_p - A_l) / (L - l))
        l_opt = int(np.argmax(E))
        for k in range(l_opt):
            Y_pred[i, indices[i, k]] = 1
    return Y_pred


def bop_markedness(stats: InferenceStats) -> np.ndarray:  # needs: {"marginal"}
    """Markedness BOP = argmax expected 0.5*(NPV + Precision) over top-l predictions.

    Paper Proposition 6 / Algorithm 4 (vacuous conventions Fpre(.,0_K)=Fneg(.,1_K)=1):
        l = 0  : 0.5 * (2 - sum_p/L)
        0<l<L  : 0.5 * (A_l/l + 1 - (sum_p - A_l)/(L-l))
        l = L  : 0.5 * (sum_p/L + 1)
    where sum_p = Σ_j p_j and A_l = sum of the top-l marginals. O(L log L), marginals only.
    Vectorized over samples; equivalent to :func:`_bop_markedness_loop`.
    """
    P = stats.marginals
    N, L = stats.n_samples, stats.n_labels
    order = np.argsort(P, axis=1)[:, ::-1]                 # (N, L) descending label indices
    P_desc = np.take_along_axis(P, order, axis=1)          # (N, L) marginals sorted desc
    sum_p = P.sum(axis=1)                                  # (N,)
    A = np.cumsum(P_desc, axis=1)                          # (N, L): A[:, l-1] = sum of top-l

    E = np.zeros((N, L + 1))
    E[:, 0] = 0.5 * (2.0 - sum_p / L)
    E[:, L] = 0.5 * (sum_p / L + 1.0)
    if L > 1:
        ls = np.arange(1, L)                               # l = 1..L-1
        A_mid = A[:, : L - 1]                              # (N, L-1)
        E[:, 1:L] = 0.5 * (A_mid / ls + 1.0 - (sum_p[:, None] - A_mid) / (L - ls))

    l_opt = E.argmax(axis=1)                               # (N,)
    keep = np.arange(L)[None, :] < l_opt[:, None]          # (N, L) top-l_opt in sorted order
    Y_pred = np.zeros((N, L))
    np.put_along_axis(Y_pred, order, keep.astype(float), axis=1)
    return Y_pred


def bop_fmeasure(stats: InferenceStats, beta: float = 1) -> np.ndarray:  # needs: {"pairwise"}
    """F-beta BOP (vectorized); paper Algorithm 1.

    For prediction size l (= k+1) the per-label score is
        q_{l,label} = (1+β²) Σ_{s=1}^L P(y_label=1, |y|=s) / (β²·s + l);
    the best size-l prediction is the top-l labels by q_{l,·}, and its expected
    F-beta is the sum of those top-l q values. The all-zero prediction (expected
    F-beta = P(|y|=0)) wins iff it beats every nonempty size.
    """
    N, L = stats.n_samples, stats.n_labels
    P_pair_wise = stats.pairwise          # (N, L, L+1): P(y_label=1, |y|=s)
    E0 = stats.p_empty[:, 0]              # (N,): P(|y|=0)

    s = np.arange(1, L + 1)
    l = np.arange(1, L + 1)
    W = (1 + beta**2) / (beta**2 * s[None, :] + l[:, None])  # (L, L): W[l-1, s-1]

    Pe = P_pair_wise[:, :, 1:]                          # (N, L, L) over s = 1..L
    q = np.einsum("nls,ks->nlk", Pe, W)                # (N, L_label, L_size)
    q_sorted_desc = np.sort(q, axis=1)[:, ::-1, :]     # labels descending
    cum = np.cumsum(q_sorted_desc, axis=1)             # (N, L, L_size)
    rank = np.arange(L)
    E_main = cum[:, rank, rank]                        # (N, L): top-(k+1) sums
    E = np.concatenate([E_main, np.zeros((N, 1))], axis=1)  # (N, L+1)

    Y_pred = np.zeros((N, L))
    choose_empty = E0 > E.max(axis=1)
    k_opt = E.argmax(axis=1)                                            # (N,)
    qk = np.take_along_axis(q, k_opt[:, None, None], axis=2)[:, :, 0]   # (N, L) scores at chosen size
    order = np.argsort(qk, axis=1)[:, ::-1]                             # (N, L) desc, same tie order
    keep = (np.arange(L)[None, :] < (k_opt + 1)[:, None]) & (~choose_empty[:, None])
    np.put_along_axis(Y_pred, order, keep.astype(float), axis=1)
    return Y_pred


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

    @staticmethod
    def _binary_matrix(K, L):
        """Return (K, L) int8 matrix where row k is binary repr of k with L bits, MSB-first."""
        if L == 0:
            return np.zeros((1, 0), dtype=np.int8)
        powers = (1 << np.arange(L - 1, -1, -1)).astype(np.int64)
        return ((np.arange(K, dtype=np.int64)[:, None] & powers[None, :]) > 0).astype(np.int8)

    def _joint(self, X):
        """Compute the full joint P(y | x) for a batch via prefix-tree batching.

        A classifier chain factorises the joint by the probability chain rule

            P(y | x) = ∏_{j=0}^{L-1} P(y_j | x, y_0, …, y_{j-1}),

        the j-th factor being the chain's j-th binary classifier. Enumerating all
        2^L vectors is therefore a depth-L prefix tree: extending each partial
        prefix y_0..y_{j-1} by y_j multiplies its probability by that factor.

        For each chain level j, all N × 2^j partial label sequences are scored
        with one batched `predict_proba` call (instead of N × 2^L × L per-element
        calls). Numerically equivalent to brute-force enumeration but ~L × 2^L
        fewer estimator calls (verified by `_predict_reference()`).

        Layout convention: the k-th (0 ≤ k < 2^L) probability for sample n
        corresponds to the label vector = binary repr of k with L bits, MSB-first
        (i.e. y[j] = (k >> (L-1-j)) & 1).

        Returns
        -------
        (joint_p, all_vecs, s_vals)
            joint_p  : (N, 2^L) joint probability of every label vector.
            all_vecs : (2^L, L) the corresponding label vectors (float).
            s_vals   : (2^L,) cardinality |y| of each label vector.
        """
        N, _ = X.shape
        L = self.L
        K = 1 << L

        # Sample-major prefix-tree expansion. At depth j: joint_p has shape
        # (N * 2^j,); sample n's entries occupy [n*2^j : (n+1)*2^j), and within
        # that block index k_partial is the prefix = binary repr of k_partial
        # with j bits (MSB-first).
        joint_p = np.ones(N, dtype=np.float64)
        for j in range(L):
            K_j = 1 << j
            x_rep = np.repeat(X, K_j, axis=0)  # (N * K_j, D), sample-major
            if j == 0:
                inputs = x_rep
            else:
                prefixes = self._binary_matrix(K_j, j).astype(np.float64)  # (K_j, j)
                prefix_rep = np.tile(prefixes, (N, 1))                     # (N * K_j, j)
                inputs = np.concatenate([x_rep, prefix_rep], axis=1)

            proba = self.ensemble[j].predict_proba(inputs)  # (N * K_j, 2)

            # Interleaved expansion: each entry → 2 entries (||0 even, ||1 odd),
            # preserving the sample-major layout.
            new_joint = np.empty(N * K_j * 2, dtype=np.float64)
            new_joint[0::2] = joint_p * proba[:, 0]
            new_joint[1::2] = joint_p * proba[:, 1]
            joint_p = new_joint

        joint_p = joint_p.reshape(N, K)
        all_vecs = self._binary_matrix(K, L).astype(np.float64)  # (K, L) MSB-first
        s_vals = all_vecs.sum(axis=1).astype(np.int64)           # (K,) cardinality
        return joint_p, all_vecs, s_vals

    def compute_stats(
        self,
        X: np.ndarray,
        needs: "Iterable[str]" = ("map", "marginal", "pairwise"),
        batch_size: "int | None" = None,
    ) -> InferenceStats:
        """Compute only the probabilistic statistics named in ``needs``.

        ``needs`` is any subset of {"map", "marginal", "pairwise"}. The expensive
        2^L joint is computed once if any of those are requested, then the
        requested quantities are derived from it. Requesting nothing (e.g. for
        the trivial Recall / NPV rules) skips the joint entirely.

        ``batch_size`` caps the number of samples held in the 2^L joint at once.
        Samples are independent, so chunking is numerically identical to a single
        pass; it bounds peak memory (the joint is (batch_size, 2^L)) for datasets
        with many samples. ``None`` (default) processes all samples in one batch.
        """
        needs = set(needs)
        N, _ = X.shape
        L = self.L
        if not (needs & {"map", "marginal", "pairwise"}):
            return InferenceStats(n_samples=N, n_labels=L)

        if batch_size is None or batch_size >= N:
            return self._compute_stats_batch(X, needs)

        parts = [
            self._compute_stats_batch(X[i:i + batch_size], needs)
            for i in range(0, N, batch_size)
        ]
        merged = InferenceStats(n_samples=N, n_labels=L)
        for field in ("map_prediction", "marginals", "pairwise", "p_empty", "p_full"):
            if getattr(parts[0], field) is not None:
                setattr(merged, field, np.concatenate([getattr(p, field) for p in parts], axis=0))
        return merged

    def _compute_stats_batch(self, X, needs) -> InferenceStats:
        """Compute the requested statistics for one batch (no chunking)."""
        N, _ = X.shape
        L = self.L
        stats = InferenceStats(n_samples=N, n_labels=L)
        joint_p, all_vecs, s_vals = self._joint(X)
        K = all_vecs.shape[0]

        if "map" in needs:
            # Subset-0/1 BOP: the single most probable label vector.
            stats.map_prediction = all_vecs[np.argmax(joint_p, axis=1)]  # (N, L)
        if "marginal" in needs:
            # Marginal p_j = P(y_j=1 | x) = Σ_y P(y | x)·y_j  →  joint_p @ all_vecs.
            stats.marginals = joint_p @ all_vecs                         # (N, L)
        if "pairwise" in needs:
            # Pairwise P(y_j=1, |y|=s | x) = Σ_{y : |y|=s} P(y | x)·y_j, grouped by
            # cardinality s (each joint column belongs to exactly one s).
            P_pair = np.zeros((N, L, L + 1))
            for s in range(L + 1):
                mask = s_vals == s
                if mask.any():
                    P_pair[:, :, s] = joint_p[:, mask] @ all_vecs[mask]
            stats.pairwise = P_pair
            stats.p_empty = joint_p[:, 0:1].copy()       # P(|y|=0): all-zero vec at index 0
            stats.p_full = joint_p[:, K - 1:K].copy()    # P(|y|=L): all-ones vec at index K-1
        return stats

    def predict(self, X, marginal=False, pairwise=False) -> tuple[np.ndarray, np.ndarray, dict]:
        """Backward-compatible wrapper returning the legacy (MAP, marginal, pairwise-dict) tuple.

        Prefer :meth:`compute_stats`, which computes only what a given rule needs.
        The ``marginal`` / ``pairwise`` flags are retained for API compatibility;
        this method always returns all three statistics.
        """
        s = self.compute_stats(X, needs={"map", "marginal", "pairwise"})
        return (
            s.map_prediction,
            s.marginals,
            {"P_pair_wise": s.pairwise, "P_pair_wise0": s.p_empty, "P_pair_wise1": s.p_full},
        )

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

    def predict_hamming(self, X: np.ndarray) -> np.ndarray:
        """Bayes-optimal predictor for Hamming Accuracy (see :func:`bop_hamming`)."""
        return bop_hamming(self.compute_stats(X, needs={"marginal"}))

    def predict_subset(self, X: np.ndarray) -> np.ndarray:
        """Bayes-optimal predictor for Subset Accuracy (see :func:`bop_subset`)."""
        return bop_subset(self.compute_stats(X, needs={"map"}))

    def predict_precision(self, X: np.ndarray) -> np.ndarray:
        """Bayes-optimal predictor for Precision (see :func:`bop_precision`)."""
        return bop_precision(self.compute_stats(X, needs={"marginal"}))

    def predict_npv(self, X: np.ndarray) -> np.ndarray:
        """Bayes-optimal predictor for Negative Predictive Value (see :func:`bop_npv`)."""
        return bop_npv(self.compute_stats(X, needs=set()))

    def predict_recall(self, X: np.ndarray) -> np.ndarray:
        """Bayes-optimal predictor for Recall (see :func:`bop_recall`)."""
        return bop_recall(self.compute_stats(X, needs=set()))

    def predict_markedness(self, X: np.ndarray) -> np.ndarray:
        """Bayes-optimal predictor for Markedness (see :func:`bop_markedness`)."""
        return bop_markedness(self.compute_stats(X, needs={"marginal"}))

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

    def predict_fmeasure(self, X: np.ndarray, beta: float = 1) -> np.ndarray:
        """Bayes-optimal predictor for the F-beta measure (see :func:`bop_fmeasure`)."""
        return bop_fmeasure(self.compute_stats(X, needs={"pairwise"}), beta=beta)


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
