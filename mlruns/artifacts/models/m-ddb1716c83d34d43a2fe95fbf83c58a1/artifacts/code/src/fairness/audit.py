import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import mean_absolute_error, median_absolute_error

MAX_ERROR_RATIO = 1.25


def group_error_table(y_true, y_pred, groups: pd.Series) -> pd.DataFrame:
    frame = MetricFrame(
        metrics={"median_abs_error": median_absolute_error, "mean_abs_error": mean_absolute_error},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=groups,
    )
    table = frame.by_group.copy()
    table["count"] = groups.value_counts()
    return table


def error_ratio_vs_rest(y_true, y_pred, groups: pd.Series, group: str) -> float:
    errors = (pd.Series(y_true, index=groups.index) - pd.Series(y_pred, index=groups.index)).abs()
    in_group = groups == group
    return errors[in_group].median() / errors[~in_group].median()


def breaches(y_true, y_pred, groups: pd.Series, limit: float = MAX_ERROR_RATIO) -> dict:
    ratios = {g: error_ratio_vs_rest(y_true, y_pred, groups, g) for g in groups.dropna().unique()}
    return {g: round(float(r), 2) for g, r in ratios.items() if r > limit}
