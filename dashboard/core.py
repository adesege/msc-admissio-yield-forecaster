import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, count
from sklearn.metrics import mean_absolute_error, median_absolute_error

from api.main import LoadedModel, to_frame
from api.schemas import Driver, Forecast, InstitutionCycle
from src.fairness.alerts import alert_fairness_summary, alert_metric_frame
from src.fairness.audit import MAX_ERROR_RATIO
from src.models.evaluation import group_error_table, interval_coverage
from src.models.features import FEATURE_LABELS, FEATURES, fairness_group

DATA_DIR = Path(__file__).resolve().parent / "data"

GROUP_ORDER = ["All other institutions", "HBCU", "Under 1,000 students", "Private for-profit"]
MONITORED_GROUPS = GROUP_ORDER[1:]
GROUP_COLORS = {
    "All other institutions": "#a3a29c",
    "HBCU": "#2a78d6",
    "Under 1,000 students": "#eb6834",
    "Private for-profit": "#1baf7a",
}
BASELINE_COLOR = "#a3a29c"
MODEL_COLOR = "#2a78d6"
INK, MUTED = "#0b0b0b", "#52514e"

EXAMPLE_LABELS = {
    "HBCU": "Mid-size public HBCU",
    "Small for-profit": "Small private for-profit college",
    "Large public": "Large public university",
}

TEST_POLICY_NAMES = {1: "Test scores required", 3: "Test scores neither required nor recommended",
                     5: "Test scores considered but not required"}
CONTROL_NAMES = {1: "public", 2: "private nonprofit", 3: "private for-profit"}
SIZE_NAMES = {1: "under 1,000", 2: "1,000 to 4,999", 3: "5,000 to 9,999", 4: "10,000 to 19,999", 5: "20,000 or more"}
REGION_NAMES = {0: "US service schools", 1: "New England", 2: "Mid East", 3: "Great Lakes", 4: "Plains", 5: "Southeast",
                6: "Southwest", 7: "Rocky Mountains", 8: "Far West", 9: "Outlying areas"}
LOCALE_NAMES = {11: "large city", 12: "midsize city", 13: "small city", 21: "large suburb", 22: "midsize suburb",
                23: "small suburb", 31: "fringe town", 32: "distant town", 33: "remote town", 41: "fringe rural area",
                42: "distant rural area", 43: "remote rural area"}

PLAIN_LABELS = {
    "YIELD_LAG1": "Last cycle's yield",
    "ADMIT_RATE": "Share of applicants admitted",
    "ADMIT_RATE_LAG1": "Share admitted last cycle",
    "APPLCN_CHANGE": "Change in applications",
    "LOG_APPLCN": "Number of applications",
    "TEST_POLICY_CHANGED": "Change in test-score policy",
    "ADMCON7": "Test-score policy",
    "CONTROL": "Type of institution",
    "INSTSIZE": "Enrollment size",
    "LOCALE": "Setting",
    "OBEREG": "Region",
    "C21BASIC": "Carnegie classification",
}

CYCLE_FIELDS = list(InstitutionCycle.model_fields)
SHAP_COLUMNS = [f"shap_{feature}" for feature in FEATURES]
LABEL_TO_FEATURE = {label: feature for feature, label in FEATURE_LABELS.items()}


@dataclass(frozen=True)
class DashboardData:
    holdout: pd.DataFrame
    summary: dict

    def institution(self, unitid: str) -> pd.Series:
        return self.holdout.loc[self.holdout["unitid"] == unitid].iloc[0]

    def example_unitids(self) -> dict[str, str]:
        return {EXAMPLE_LABELS[key]: unitid for key, unitid in self.summary["examples"].items()}


def load_dashboard_data(directory: Path = DATA_DIR) -> DashboardData:
    return DashboardData(
        holdout=pd.read_parquet(directory / "holdout.parquet"),
        summary=json.loads((directory / "summary.json").read_text()),
    )


def _optional_int(value) -> int | None:
    return None if pd.isna(value) else int(value)


def _optional_float(value) -> float | None:
    return None if pd.isna(value) else float(value)


def cycle_from_row(row: pd.Series) -> InstitutionCycle:
    values = {field: row[field] for field in CYCLE_FIELDS if field != "institution_id"}
    for field in ("last_cycle_test_policy", "locale", "region", "carnegie_class"):
        values[field] = _optional_int(values[field])
    return InstitutionCycle(institution_id=row["unitid"], **values)


def scenario(cycle: InstitutionCycle, offers: int | None = None, applicants: int | None = None,
             test_policy: int | None = None) -> InstitutionCycle:
    changes = {"offers": offers, "applicants": applicants, "test_policy": test_policy}
    updated = cycle.model_copy(update={k: v for k, v in changes.items() if v is not None})
    return InstitutionCycle.model_validate(updated.model_dump())


def enrollment_curve(model: LoadedModel, cycle: InstitutionCycle, offer_counts) -> pd.DataFrame:
    scenarios = [scenario(cycle, offers=int(n)) for n in offer_counts]
    frame = pd.concat([to_frame(s) for s in scenarios], ignore_index=True)
    interval = model.forecaster.predict_interval(frame, fairness_group(frame))
    offers = np.array([s.offers for s in scenarios])
    return pd.DataFrame(
        {
            "offers": offers,
            "forecast_yield": interval["yield"].to_numpy(),
            "expected_enrolled": offers * interval["yield"].to_numpy(),
            "enrolled_lower": offers * interval["lower"].to_numpy(),
            "enrolled_upper": offers * interval["upper"].to_numpy(),
        }
    )


def describe_value(feature: str, value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "not reported"
    if feature in ("YIELD_LAG1", "ADMIT_RATE", "ADMIT_RATE_LAG1"):
        return f"{value:.0%}"
    if feature == "APPLCN_CHANGE":
        return f"{value:+.0%} on last cycle"
    if feature == "LOG_APPLCN":
        return f"{math.expm1(value):,.0f} applications"
    if feature == "TEST_POLICY_CHANGED":
        return "yes" if value else "no"
    if feature == "ADMCON7":
        return TEST_POLICY_NAMES.get(int(value), f"code {value:g}").lower()
    if feature == "CONTROL":
        return CONTROL_NAMES.get(int(value), f"code {value:g}")
    if feature == "INSTSIZE":
        return f"{SIZE_NAMES.get(int(value), f'code {value:g}')} students"
    if feature == "OBEREG":
        return REGION_NAMES.get(int(value), f"code {value:g}")
    if feature == "LOCALE":
        return LOCALE_NAMES.get(int(value), f"code {value:g}")
    return f"code {value:g}"


def driver_sentence(driver: Driver) -> str:
    feature = LABEL_TO_FEATURE[driver.feature]
    direction = "raised" if driver.contribution_points > 0 else "lowered"
    return (f"{PLAIN_LABELS[feature]} ({describe_value(feature, driver.value)}) {direction} the forecast by "
            f"{abs(driver.contribution_points):.1f} percentage points.")


def forecast_summary(forecast: Forecast, last_cycle_yield: float) -> str:
    interval = forecast.yield_interval_80
    enrolled = forecast.expected_enrolled_interval_80
    return (
        f"Last cycle, {last_cycle_yield:.0%} of admitted students enrolled. The model expects about "
        f"{forecast.forecast_yield:.0%} in the coming cycle, and in four cases out of five the true figure should fall between "
        f"{interval.lower:.0%} and {interval.upper:.0%}. With the offers entered, the forecast means about "
        f"{forecast.expected_enrolled:,} new students, within a range of {enrolled.lower:,.0f} to {enrolled.upper:,.0f}."
    )


def contributions(row: pd.Series) -> pd.DataFrame:
    inputs = to_frame(cycle_from_row(row)).iloc[0]
    values = row[SHAP_COLUMNS].astype(float).to_numpy()
    table = pd.DataFrame({"feature": [PLAIN_LABELS[f] for f in FEATURES], "points": values,
                          "value": [describe_value(f, _optional_float(inputs[f])) for f in FEATURES]})
    return table.reindex(table["points"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def global_importance(holdout: pd.DataFrame) -> pd.DataFrame:
    mean_impact = holdout[SHAP_COLUMNS].abs().mean()
    table = pd.DataFrame({"feature": [PLAIN_LABELS[f] for f in FEATURES], "points": mean_impact.to_numpy()})
    return table.sort_values("points", ascending=False).reset_index(drop=True)


def error_by_group(holdout: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    frame = MetricFrame(
        metrics={"institutions": count, "mean_error": mean_absolute_error, "median_error": median_absolute_error},
        y_true=holdout["actual_yield"],
        y_pred=holdout[prediction_column],
        sensitive_features=holdout["fairness_group"],
    )
    table = frame.by_group.reindex(GROUP_ORDER)
    table[["mean_error", "median_error"]] *= 100
    return table


def error_ratio_table(holdout: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    table = group_error_table(holdout, holdout["actual_yield"], holdout[prediction_column]).set_index("group")
    table["within_limit"] = table["median_error_ratio"] <= MAX_ERROR_RATIO
    return table


def coverage_by_group(holdout: pd.DataFrame) -> pd.DataFrame:
    table = interval_coverage(holdout["actual_yield"], holdout[["lower", "upper"]], holdout["fairness_group"])
    return table.reindex(GROUP_ORDER + ["All institutions"])


def single_threshold_alerts(holdout: pd.DataFrame, threshold_points: float) -> np.ndarray:
    return (holdout["predicted_change"] <= -threshold_points / 100).to_numpy()


def alert_fairness(holdout: pd.DataFrame, alerts: np.ndarray) -> tuple[pd.DataFrame, dict, dict]:
    frame = alert_metric_frame(holdout["actual_drop"], alerts, holdout["fairness_group"])
    overall = frame.overall.to_dict()
    overall["balanced_accuracy"] = (overall["true_positive_rate"] + 1 - overall["false_positive_rate"]) / 2
    summary = alert_fairness_summary(holdout["actual_drop"], alerts, holdout["fairness_group"])
    return frame.by_group.reindex(GROUP_ORDER), overall, summary
