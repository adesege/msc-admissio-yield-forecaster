import numpy as np
import pandas as pd
from fairlearn.reductions import AbsoluteLoss, BoundedGroupLoss, ExponentiatedGradient
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted

CHANGE_LOSS_BOUNDS = (-1.0, 1.0)


def group_balanced_weights(groups: pd.Series) -> np.ndarray:
    frequency = groups.map(groups.value_counts(normalize=True)).to_numpy()
    weights = 1.0 / frequency
    return weights / weights.mean()


class BoundedGroupLossRegressor(RegressorMixin, BaseEstimator):
    """Exponentiated gradient under a bounded group loss constraint, served as a deterministic mixture.

    Fairlearn returns a randomised mixture of predictors. Absolute loss is convex, so averaging the
    predictors by their mixture weights keeps every group's expected loss within the same bound while
    giving partners the same forecast each time they ask.
    """

    def __init__(self, estimator, upper_bound: float = 0.08, max_iter: int = 30, eps: float = 0.01,
                 loss_bounds: tuple[float, float] = CHANGE_LOSS_BOUNDS):
        self.estimator = estimator
        self.upper_bound = upper_bound
        self.max_iter = max_iter
        self.eps = eps
        self.loss_bounds = loss_bounds

    def fit(self, X, y, sensitive_features):
        reduction = ExponentiatedGradient(
            clone(self.estimator),
            constraints=BoundedGroupLoss(AbsoluteLoss(*self.loss_bounds), upper_bound=self.upper_bound),
            max_iter=self.max_iter,
            eps=self.eps,
        )
        reduction.fit(X, np.asarray(y), sensitive_features=np.asarray(sensitive_features))
        weights = np.asarray(reduction.weights_, dtype=float)
        active = weights > 0
        self.predictors_ = [p for p, keep in zip(reduction.predictors_, active) if keep]
        self.weights_ = weights[active] / weights[active].sum()
        return self

    def predict(self, X):
        check_is_fitted(self, "predictors_")
        return np.sum([w * p.predict(X) for w, p in zip(self.weights_, self.predictors_)], axis=0)
