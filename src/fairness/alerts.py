import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)

FOUR_FIFTHS_RATIO = 0.8


def actual_yield_drop(frame: pd.DataFrame, threshold: float) -> np.ndarray:
    return ((frame["YIELD"] - frame["YIELD_LAG1"]) <= threshold).to_numpy()


def alert_metric_frame(y_true, y_pred, groups: pd.Series) -> MetricFrame:
    return MetricFrame(
        metrics={
            "institutions": count,
            "actual_drop_rate": lambda t, p: float(np.mean(t)),
            "alert_rate": selection_rate,
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
        },
        y_true=np.asarray(y_true, dtype=int),
        y_pred=np.asarray(y_pred, dtype=int),
        sensitive_features=groups,
    )


def alert_fairness_summary(y_true, y_pred, groups: pd.Series) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    disparate_impact = demographic_parity_ratio(y_true, y_pred, sensitive_features=groups)
    return {
        "demographic_parity_difference": round(float(demographic_parity_difference(y_true, y_pred, sensitive_features=groups)), 3),
        "equalized_odds_difference": round(float(equalized_odds_difference(y_true, y_pred, sensitive_features=groups)), 3),
        "disparate_impact_ratio": round(float(disparate_impact), 3),
        "meets_four_fifths_rule": bool(disparate_impact >= FOUR_FIFTHS_RATIO),
    }
