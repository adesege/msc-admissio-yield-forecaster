from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.fairness.audit import MAX_ERROR_RATIO
from src.fairness.representation import REPRESENTATION_FLOOR

PSI_AMBER, PSI_RED = 0.1, 0.2
MAE_AMBER, MAE_RED = 5.5, 6.0
COVERAGE_AMBER, COVERAGE_RED = (0.75, 0.85), (0.70, 0.90)
FPR_AMBER, FPR_RED = 1.5, 2.0
MISSING_AMBER, MISSING_RED = 0.025, 0.05
RATIO_RISE_AMBER = 0.25
REPRESENTATION_RED = 0.80
MIN_GROUP_SIZE = 30
OVERRUN_AMBER, OVERRUN_RED = 0.10, 0.15

RETRAIN_SIGNALS = {"feature_psi", "yield_mae"}


@dataclass(frozen=True)
class Check:
    signal: str
    group: str | None
    value: float | None
    status: str
    rule: str


def _status(red: bool, amber: bool) -> str:
    return "red" if red else "amber" if amber else "green"


def psi_checks(psi: pd.Series) -> list[Check]:
    return [
        Check("feature_psi", feature, float(value), _status(value > PSI_RED, value > PSI_AMBER),
              f"PSI above {PSI_RED} is red, above {PSI_AMBER} amber")
        for feature, value in psi.items()
    ]


def mae_check(mae_points: float) -> Check:
    return Check("yield_mae", None, mae_points, _status(mae_points > MAE_RED, mae_points > MAE_AMBER),
                 f"Yield MAE above {MAE_RED} points is red, above {MAE_AMBER} amber")


def error_ratio_checks(current: dict[str, float], previous: dict[str, float]) -> list[Check]:
    checks = []
    for group, ratio in current.items():
        before = previous.get(group)
        breached_twice = ratio > MAX_ERROR_RATIO and before is not None and before > MAX_ERROR_RATIO
        checks.append(Check("error_ratio", group, ratio, _status(breached_twice, ratio > MAX_ERROR_RATIO),
                            f"Ratio above {MAX_ERROR_RATIO} for two cycles is red, for one cycle amber"))
        if before is not None:
            rise = ratio - before
            checks.append(Check("error_ratio_change", group, round(rise, 3), _status(False, rise > RATIO_RISE_AMBER),
                                f"Bias drift: a rise of more than {RATIO_RISE_AMBER} since the previous cycle is amber"))
    return checks


def coverage_checks(coverage: pd.DataFrame) -> list[Check]:
    checks = []
    for group, row in coverage.iterrows():
        value = float(row["coverage"])
        red = not COVERAGE_RED[0] <= value <= COVERAGE_RED[1]
        amber = not COVERAGE_AMBER[0] <= value <= COVERAGE_AMBER[1]
        checks.append(Check("interval_coverage", group, value, _status(red, amber),
                            "80% interval coverage outside 70% to 90% is red, outside 75% to 85% amber"))
    return checks


def false_alarm_checks(by_group: pd.DataFrame, overall_rate: float) -> list[Check]:
    checks = []
    for group, row in by_group.iterrows():
        rate = float(row["false_positive_rate"])
        multiple = rate / overall_rate if overall_rate > 0 else np.nan
        checks.append(Check("alert_false_positive_multiple", group, round(multiple, 3),
                            _status(multiple > FPR_RED, multiple > FPR_AMBER),
                            f"Group false-alarm rate above {FPR_RED} times the overall rate is red, "
                            f"above {FPR_AMBER} amber"))
    return checks


def missing_check(share: float, reference_share: float) -> Check:
    rise = share - reference_share
    return Check("missing_input_share", None, round(share, 4), _status(rise > MISSING_RED, rise > MISSING_AMBER),
                 f"Share of rows with a missing input more than {100 * MISSING_RED:.0f} points above the training "
                 f"cycle ({reference_share:.1%}) is red, more than {100 * MISSING_AMBER:.1f} points amber")


def representation_checks(report: pd.DataFrame) -> list[Check]:
    checks = []
    for row in report.itertuples():
        if row.source_rows < MIN_GROUP_SIZE:
            continue
        ratio = float(row.representation_ratio)
        checks.append(Check("representation_ratio", row.group, round(ratio, 3),
                            _status(ratio < REPRESENTATION_RED, ratio < REPRESENTATION_FLOOR),
                            f"Representation below {REPRESENTATION_RED} is red, below {REPRESENTATION_FLOOR} amber"))
    return checks


def overrun_checks(evaluation: pd.DataFrame) -> list[Check]:
    checks = []
    for group, row in evaluation.iterrows():
        if row["institutions"] < MIN_GROUP_SIZE:
            continue
        share = float(row["planner_overrun_share"])
        checks.append(Check("capacity_overrun_share", group, round(share, 3),
                            _status(share > OVERRUN_RED, share > OVERRUN_AMBER),
                            f"Planned intake exceeding capacity for more than {OVERRUN_RED:.0%} of institutions is red, "
                            f"more than {OVERRUN_AMBER:.0%} amber"))
    return checks


def decision(checks: list[Check]) -> dict:
    red = [c for c in checks if c.status == "red"]
    return {
        "overall_status": "red" if red else "amber" if any(c.status == "amber" for c in checks) else "green",
        "incident_required": bool(red),
        "retrain_required": any(c.signal in RETRAIN_SIGNALS for c in red),
        "panel_review_required": any(c.signal not in RETRAIN_SIGNALS for c in red),
        "red_signals": [asdict(c) for c in red],
    }
