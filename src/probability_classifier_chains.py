from uuid import uuid4
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.skmultiflow.meta.classifier_chains import ClassifierChain


def P(y, x, cc, payoff=np.prod):
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


class ProbabilisticClassifierChainCustom(ClassifierChain):
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

    def __init__(self, base_estimator=LogisticRegression(), order=None, random_state=None):
        super().__init__(base_estimator=base_estimator, order=order, random_state=random_state)
        self.store_key: str | None = None
        self.predicted_store: dict = {}

    def set_store_key(self, key: str) -> None:
        self.store_key = key
        self.predicted_store[key] = None

    def predict(self, X, marginal=False, pairwise=False) -> tuple[np.ndarray, np.ndarray, dict]:
        """Enumerate all 2^L label combinations and compute joint probabilities.

        Results are cached by store_key so multiple predict_* calls on the
        same fold reuse the computed probabilities.

        Returns
        -------
        (Yp, P_margin_yi_1, pairwise_dict)
            Yp : (N, L) — MAP prediction (argmax over joint distribution).
            P_margin_yi_1 : (N, L) — marginal P(y_j=1 | x).
            pairwise_dict : dict with 'P_pair_wise', 'P_pair_wise0', 'P_pair_wise1'.
        """
        # Return cached result if available
        if (
            self.store_key is not None
            and self.predicted_store.get(self.store_key) is not None
        ):
            print(f"Cached [{self.store_key}]")
            return self.predicted_store[self.store_key]

        print(f"Predicting... [{self.store_key}]")
        N, D = X.shape
        Yp = np.zeros((N, self.L))
        P_margin_yi_1 = np.zeros((N, self.L))
        P_pair_wise = np.zeros((N, self.L, self.L + 1))
        P_pair_wise0 = np.zeros((N, 1))
        P_pair_wise1 = np.zeros((N, 1))

        for n in range(N):
            w_max = 0.0
            for b in range(2 ** self.L):
                y_ = np.array(list(map(int, np.binary_repr(b, width=self.L))))
                w_ = P(y_, X[n], self)

                s = int(np.sum(y_))
                if s == 0:
                    P_pair_wise0[n] = w_
                if s == self.L:
                    P_pair_wise1[n] = w_

                if marginal or pairwise:
                    for j in range(self.L):
                        if y_[j] == 1:
                            P_margin_yi_1[n, j] += w_
                            P_pair_wise[n, j, s] += w_

                if w_ > w_max:
                    Yp[n, :] = y_.copy()
                    w_max = w_

        result = (
            Yp,
            P_margin_yi_1,
            {
                "P_pair_wise": P_pair_wise,
                "P_pair_wise0": P_pair_wise0,
                "P_pair_wise1": P_pair_wise1,
            },
        )
        # Fixed: update the existing dict entry instead of replacing the whole dict,
        # which would wipe out cached results for other keys.
        self.predicted_store[self.store_key] = result
        return result

    def predict_Hamming(self, X):
        """Bayes-optimal predictor for Hamming Accuracy.

        Thresholds each marginal P(y_j=1|x) at 0.5.
        """
        _, P_margin_yi_1, _ = self.predict(X, marginal=True)
        return np.where(P_margin_yi_1 > 0.5, 1, 0)

    def predict_Subset(self, X):
        """Bayes-optimal predictor for Subset Accuracy.

        Returns the joint MAP label vector (argmax over all 2^L combinations).
        """
        predictions, _, _ = self.predict(X)
        return predictions

    def predict_Precision(self, X):
        """Bayes-optimal predictor for Precision.

        Predicts exactly one label: the one with the highest marginal probability.
        """
        N = X.shape[0]
        Yp = np.zeros((N, self.L))
        _, P_margin_yi_1, _ = self.predict(X, marginal=True)
        for n in range(N):
            Yp[n, np.argmax(P_margin_yi_1[n])] = 1
        return Yp

    def predict_Neg(self, X):
        """Bayes-optimal predictor for Negative Predictive Value (NPV).

        Predicts all labels as positive except the one with the lowest marginal
        probability, maximising the fraction of true negatives among predicted
        negatives.
        """
        N, _ = X.shape
        _, P_margin_yi_1, _ = self.predict(X, marginal=True)
        # Sort marginals ascending; set only the least-likely label to 0.
        indices = np.argsort(P_margin_yi_1, axis=1)
        Yp = np.ones((N, self.L))
        Yp[np.arange(N)[:, None], indices[:, :1]] = 0
        return Yp

    def predict_Recall(self, X):
        """Bayes-optimal predictor for Recall.

        Predicting all labels as positive trivially achieves perfect recall
        (TP / (TP + FN) = 1 since FN = 0). This is the theoretical optimum
        when optimising recall alone without a precision constraint.
        """
        return np.ones((X.shape[0], self.L))

    def predict_Mar(self, X):
        """Bayes-optimal predictor for Markedness."""
        N, _ = X.shape
        _, P_margin_yi_1, _ = self.predict(X, marginal=True)
        indices = np.argsort(P_margin_yi_1, axis=1)[:, ::-1]

        E = np.zeros((N, self.L + 1))
        for i in range(N):
            sum_p = np.sum(P_margin_yi_1[i])
            E[i][0] = 2 - (1 / self.L) * sum_p
            E[i][self.L] = 1 + (1 / self.L) * sum_p
            s2 = 0
            for _l in range(1, self.L):
                s2 += P_margin_yi_1[i, indices[i, _l - 1]]
                E[i][_l] = (
                    1 - (1 / (self.L - _l)) * sum_p
                    + (1 / ((self.L - _l) * _l)) * s2
                )

        l_optimal = np.argsort(E, axis=1)[:, ::-1]
        Yp = np.zeros((N, self.L))
        for i in range(N):
            for _l in range(l_optimal[i][0]):
                Yp[i][indices[i, _l]] = 1
        return Yp

    def predict_Fmeasure(self, X, beta=1):
        """Bayes-optimal predictor for F-beta measure."""
        N, _ = X.shape
        _, _, pw = self.predict(X, pairwise=True)
        P_pair_wise = pw["P_pair_wise"]
        P_pair_wise0 = pw["P_pair_wise0"]

        Yp = np.zeros((N, self.L))
        for i in range(N):
            q = np.zeros((self.L, self.L))
            indices_desc = []
            E = np.zeros(self.L + 1)
            E0 = P_pair_wise0[i][0]

            for k in range(self.L):
                for label in range(self.L):
                    for s in range(self.L):
                        q[k][label] += (1 + beta**2) * (
                            P_pair_wise[i][label][s]
                            / (beta**2 * (s + 1) + k + 1)
                        )
                indices_desc.append(np.argsort(q[k])[::-1].tolist())
                for i_ in range(k + 1):
                    E[k] += q[k][int(indices_desc[k][i_])]

            if E0 > np.max(E):
                Yp[i] = np.zeros(self.L)
            else:
                k_opt = np.argmax(E)
                for _l in range(k_opt + 1):
                    Yp[i][int(indices_desc[k_opt][_l])] = 1

        return Yp

    def predict_Inf(self, X):
        """Bayes-optimal predictor for Informedness (Sensitivity + Specificity - 1).

        Include label j iff q_sens[j] + q_spec_cost[j] > C, where:
            q_sens[j]      = Σ_{s=1}^{L}   P(y_j=1, |y|=s | x) / s
            q_spec_cost[j] = Σ_{s=1}^{L-1} P(y_j=1, |y|=s | x) / (L-s)
            C              = P(|y|=0 | x)/L + Σ_{s=1}^{L-1} P(|y|=s | x) / (s*(L-s))
        """
        N, _ = X.shape
        _, _, pw = self.predict(X, pairwise=True)
        P_pair_wise = pw["P_pair_wise"]    # (N, L, L+1)
        P_pair_wise0 = pw["P_pair_wise0"]  # (N, 1)
        L = self.L
        Yp = np.zeros((N, L))

        for i in range(N):
            q_sens = np.zeros(L)
            q_spec_cost = np.zeros(L)
            for j in range(L):
                for s in range(1, L + 1):
                    q_sens[j] += P_pair_wise[i, j, s] / s
                    if s < L:
                        q_spec_cost[j] += P_pair_wise[i, j, s] / (L - s)

            C = float(P_pair_wise0[i, 0]) / L
            for s in range(1, L):
                P_s = np.sum(P_pair_wise[i, :, s]) / s
                C += P_s / (L - s)

            Yp[i] = np.where(q_sens + q_spec_cost > C, 1.0, 0.0)

        return Yp
