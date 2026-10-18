import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, root_mean_squared_error

from src.fairness.audit import MAX_ERROR_RATIO, error_ratio_vs_rest
from src.models.features import FAIRNESS_GROUPS

MAX_YIELD_MAE_POINTS = 4.8
INTAKE_MISS_TOLERANCE = 0.10


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "yield_mae_points": round(100 * mean_absolute_error(y_true, y_pred), 2),
        "yield_rmse_points": round(100 * root_mean_squared_error(y_true, y_pred), 2),
        "yield_median_abs_error_points": round(100 * median_absolute_error(y_true, y_pred), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 3),
    }


def intake_metrics(frame: pd.DataFrame, y_pred) -> dict[str, float]:
    predicted_enrolled = frame["ADMSSN"].to_numpy() * np.asarray(y_pred)
    actual = frame["ENRLT"].where(frame["ENRLT"] > 0)
    error = pd.Series((predicted_enrolled - actual) / actual).dropna()
    return {
        "share_intake_missed_by_over_10pct": round(float((error.abs() > INTAKE_MISS_TOLERANCE).mean()), 3),
        "share_intake_over_by_10pct": round(float((error > INTAKE_MISS_TOLERANCE).mean()), 3),
        "share_intake_under_by_10pct": round(float((error < -INTAKE_MISS_TOLERANCE).mean()), 3),
    }


def group_error_ratios(frame: pd.DataFrame, y_true, y_pred) -> dict[str, float]:
    return {
        name: round(float(error_ratio_vs_rest(np.asarray(y_true), np.asarray(y_pred), frame[column], label)), 2)
        for name, (column, label) in FAIRNESS_GROUPS.items()
    }


def group_error_table(frame: pd.DataFrame, y_true, y_pred) -> pd.DataFrame:
    errors = pd.Series(np.abs(np.asarray(y_true) - np.asarray(y_pred)), index=frame.index)
    rows = []
    for name, (column, label) in FAIRNESS_GROUPS.items():
        in_group = frame[column] == label
        rows.append(
            {
                "group": name,
                "institutions": int(in_group.sum()),
                "group_mae_points": round(100 * errors[in_group].mean(), 2),
                "rest_mae_points": round(100 * errors[~in_group].mean(), 2),
                "group_median_error_points": round(100 * errors[in_group].median(), 2),
                "rest_median_error_points": round(100 * errors[~in_group].median(), 2),
                "median_error_ratio": round(float(errors[in_group].median() / errors[~in_group].median()), 2),
            }
        )
    return pd.DataFrame(rows)


def sampling_noise_points(frame: pd.DataFrame, groups: pd.Series) -> pd.Series:
    binomial_sd = np.sqrt(frame["YIELD"] * (1 - frame["YIELD"]) / frame["ADMSSN"])
    return (100 * binomial_sd).groupby(groups).median().round(2)


def interval_coverage(y_true, intervals: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    covered = (np.asarray(y_true) >= intervals["lower"].to_numpy()) & (np.asarray(y_true) <= intervals["upper"].to_numpy())
    width = 100 * (intervals["upper"] - intervals["lower"])
    table = pd.DataFrame({"group": groups.to_numpy(), "covered": covered, "width_points": width.to_numpy()})
    summary = table.groupby("group").agg(
        institutions=("covered", "size"), coverage=("covered", "mean"), median_width_points=("width_points", "median")
    )
    overall = pd.DataFrame(
        {"institutions": [len(table)], "coverage": [covered.mean()], "median_width_points": [width.median()]},
        index=["All institutions"],
    )
    return pd.concat([summary, overall]).round(3)


def acceptance_check(metrics: dict, ratios: dict, baseline_intake_miss_share: float) -> dict:
    checks = {
        "yield_mae_at_or_below_4_8_points": metrics["yield_mae_points"] <= MAX_YIELD_MAE_POINTS,
        "intake_miss_share_below_baseline": metrics["share_intake_missed_by_over_10pct"] < baseline_intake_miss_share,
        **{f"error_ratio_{name}_at_or_below_1_25": ratio <= MAX_ERROR_RATIO for name, ratio in ratios.items()},
    }
    return {"checks": checks, "approved_for_release": all(checks.values())}
