import copy

import numpy as np
from sklearn.linear_model import LogisticRegression
from src.skmultiflow.core.base import (
    BaseSKMObject,
    ClassifierMixin,
    MetaEstimatorMixin,
    MultiOutputMixin,
)


from src.skmultiflow.utils.validation import check_random_state


class ClassifierChain(
    BaseSKMObject, ClassifierMixin, MetaEstimatorMixin, MultiOutputMixin
):
    """Classifier Chains for multi-label learning.

    Parameters
    ----------
    base_estimator: skmultiflow.core.BaseSKMObject or sklearn.BaseEstimator
        (default=LogisticRegression) Each member of the ensemble is
        an instance of the base estimator

    order : str (default=None)
        `None` to use default order, 'random' for random order.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.

    Examples
    --------
    >>> from skmultiflow.data import make_logical
    >>> from sklearn.linear_model import SGDClassifier
    >>>
    >>> X, Y = make_logical(random_state=1)
    >>>
    >>> print("TRUE: ")
    >>> print(Y)
    >>> print("vs")
    >>>
    >>> print("CC")
    >>> cc = ClassifierChain(SGDClassifier(max_iter=100, loss='log', random_state=1))
    >>> cc.fit(X, Y)
    >>> print(cc.predict(X))
    >>>
    >>> print("RCC")
    >>> cc = ClassifierChain(SGDClassifier(max_iter=100, loss='log', random_state=1),
    ...                                     order='random', random_state=1)
    >>> cc.fit(X, Y)
    >>> print(cc.predict(X))
    >>>
    TRUE:
    [[1. 0. 1.]
     [1. 1. 0.]
     [0. 0. 0.]
     [1. 1. 0.]]
    vs
    CC
    [[1. 0. 1.]
     [1. 1. 0.]
     [0. 0. 0.]
     [1. 1. 0.]]
    RCC
    [[1. 0. 1.]
     [1. 1. 0.]
     [0. 0. 0.]
     [1. 1. 0.]]



    Notes
    -----
    Classifier Chains [1]_ is a popular method for multi-label learning. It exploits correlation
    between labels by incrementally building binary classifiers for each label.

    scikit-learn also includes 'ClassifierChain'. A difference is probabilistic extensions
    are included here.


    References
    ----------
    .. [1] Read, Jesse, Bernhard Pfahringer, Geoff Holmes, and Eibe Frank. "Classifier chains
        for multi-label classification." In Joint European Conference on Machine Learning and
        Knowledge Discovery in Databases, pp. 254-269. Springer, Berlin, Heidelberg, 2009.

    """

    # TODO: much of this can be shared with Regressor Chains, probably should
    # use a base class to inherit here.

    def __init__(
        self, base_estimator=LogisticRegression(), order=None, random_state=None
    ):
        super().__init__()
        self.base_estimator = base_estimator
        self.order = order
        self.random_state = random_state
        self.chain = None
        self.ensemble = None
        self.L = None
        self._random_state = (
            None  # This is the actual random_state object used internally
        )
        self.__configure()

    def __configure(self):
        self.ensemble = None
        self.L = -1
        self._random_state = check_random_state(self.random_state)

    def fit(self, X, y, classes=None, sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples, n_targets)
            An array-like with the labels of all samples in X.

        classes: Not used (default=None)

        sample_weight: Not used (default=None)

        Returns
        -------
        self

        """
        N, self.L = y.shape
        L = self.L
        N, D = X.shape

        self.chain = np.arange(L)
        if self.order == "random":
            self._random_state.shuffle(self.chain)

        # Set the chain order
        y = y[:, self.chain]

        # Train
        self.ensemble = [copy.deepcopy(self.base_estimator) for _ in range(L)]
        XY = np.zeros((N, D + L - 1))
        XY[:, 0:D] = X
        XY[:, D:] = y[:, 0 : L - 1]
        for j in range(self.L):
            self.ensemble[j].fit(XY[:, 0 : D + j], y[:, j])
        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Partially (incrementally) fit the model.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples)
            An array-like with the labels of all samples in X.

        classes: Not used (default=None)

        sample_weight: NOT used (default=None)

        Returns
        -------
        self

        """
        if self.ensemble is None:
            # This is the first time that the model is fit
            self.fit(X, y)
            return self

        N, self.L = y.shape
        L = self.L
        N, D = X.shape

        # Set the chain order
        y = y[:, self.chain]

        XY = np.zeros((N, D + L - 1))
        XY[:, 0:D] = X
        XY[:, D:] = y[:, 0 : L - 1]
        for j in range(L):
            self.ensemble[j].partial_fit(XY[:, 0 : D + j], y[:, j])

        return self

    def predict(self, X):
        """Predict classes for the passed data.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of data samples to predict the labels for.

        Returns
        -------
        A numpy.ndarray with all the predictions for the samples in X.

        """
        N, D = X.shape
        Y = np.zeros((N, self.L))
        for j in range(self.L):
            if j > 0:
                X = np.column_stack([X, Y[:, j - 1]])
            Y[:, j] = self.ensemble[j].predict(X)

        # Unset the chain order (back to default)
        return Y[:, np.argsort(self.chain)]

    def predict_proba(self, X):
        """Estimates the probability of each sample in X belonging to each of the class-labels.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The matrix of samples one wants to predict the class probabilities for.

        Returns
        -------
        A numpy.ndarray of shape (n_samples, n_labels), in which each outer entry is associated
        with the X entry of the same index. And where the list in index [i] contains
        len(self.target_values) elements, each of which represents the probability that
        the i-th sample of X belongs to a certain class-label.

        Notes
        -----
        Returns marginals [P(y_1=1|x),...,P(y_L=1|x,y_1,...,y_{L-1})]
        i.e., confidence predictions given inputs, for each instance.

        This function suitable for multi-label (binary) data
        only at the moment (may give index-out-of-bounds error if
        uni- or multi-target (of > 2 values) data is used in training).
        """
        N, D = X.shape
        Y = np.zeros((N, self.L))
        for j in range(self.L):
            if j > 0:
                X = np.column_stack([X, Y[:, j - 1]])
            Y[:, j] = self.ensemble[j].predict_proba(X)[:, 1]
        return Y

    def reset(self):
        self.__configure()
        return self

    def _more_tags(self):
        return {"multioutput": True, "multioutput_only": True}
