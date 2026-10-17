import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted

from src.models.features import LAG_COLUMN, prepare_features

INTERVAL_COVERAGE = 0.80
YIELD_DROP_ALERT_THRESHOLD = -0.03


class YieldForecaster(RegressorMixin, BaseEstimator):
    """Forecasts next-cycle yield as last cycle's yield plus a learned change."""

    def __init__(self, change_model, interval_coverage: float = INTERVAL_COVERAGE,
                 alert_threshold: float = YIELD_DROP_ALERT_THRESHOLD):
        self.change_model = change_model
        self.interval_coverage = interval_coverage
        self.alert_threshold = alert_threshold

    def fit(self, X: pd.DataFrame, y, **fit_params):
        change = np.asarray(y, dtype=float) - X[LAG_COLUMN].to_numpy(dtype=float)
        self.change_model_ = clone(self.change_model).fit(prepare_features(X), change, **fit_params)
        return self

    def predict_change(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "change_model_")
        return np.asarray(self.change_model_.predict(prepare_features(X)), dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        lag = pd.to_numeric(X[LAG_COLUMN], errors="coerce").to_numpy(dtype=float)
        return np.clip(lag + self.predict_change(X), 0.0, 1.0)

    def calibrate_intervals(self, residuals, groups: pd.Series) -> "YieldForecaster":
        absolute = pd.Series(np.abs(np.asarray(residuals, dtype=float)), index=groups.index)
        self.interval_halfwidths_ = absolute.groupby(groups).quantile(self.interval_coverage).to_dict()
        self.overall_halfwidth_ = float(absolute.quantile(self.interval_coverage))
        return self

    def predict_interval(self, X: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
        check_is_fitted(self, "interval_halfwidths_")
        prediction = self.predict(X)
        halfwidth = np.array([self.interval_halfwidths_.get(g, self.overall_halfwidth_) for g in groups])
        return pd.DataFrame(
            {
                "yield": prediction,
                "lower": np.clip(prediction - halfwidth, 0.0, 1.0),
                "upper": np.clip(prediction + halfwidth, 0.0, 1.0),
            },
            index=X.index,
        )

    def yield_drop_alert(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_change(X) <= self.alert_threshold
