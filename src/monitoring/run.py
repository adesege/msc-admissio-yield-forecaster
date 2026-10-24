import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.config import default_paths
from src.fairness.alerts import actual_yield_drop, alert_fairness_summary, alert_metric_frame
from src.fairness.audit import MAX_ERROR_RATIO
from src.models.dataset import load_modelling_data
from src.models.evaluation import group_error_ratios, interval_coverage, regression_metrics
from src.models.features import FEATURES, TARGET, TEST_CYCLE, TRAIN_CYCLE
from src.models.validation import cross_validated_yield
from src.monitoring import checks, figures
from src.monitoring.drift import feature_psi, missing_input_share
from src.monitoring.evidently_reports import PREDICTION, data_drift_report, regression_report
from src.planning.offers import goal_two_met, plan_evaluation, plan_offers


def latest_representation_report(processed_dir: Path) -> pd.DataFrame:
    latest = sorted((processed_dir / "representation_report").glob("*.parquet"))[-1]
    report = pd.read_parquet(latest)
    return report[report["dimension"].isin(["HBCU_GROUP", "SIZE_GROUP", "CONTROL_GROUP"])]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def main(output_dir: Path | None = None) -> dict:
    paths = default_paths()
    output_dir = output_dir or paths.reports_dir / "monitoring"
    model_dir = paths.root / "models"
    forecaster = joblib.load(model_dir / "admissio_yield.joblib")
    metadata = json.loads((model_dir / "metadata.json").read_text())
    data = load_modelling_data()
    reference, current = data.train, data.test

    reference_prediction = cross_validated_yield(forecaster, reference, metadata["change_model"])
    current_prediction = forecaster.predict(current)

    psi = feature_psi(reference, current)
    previous_ratios = group_error_ratios(reference, reference[TARGET], reference_prediction)
    current_ratios = group_error_ratios(current, current[TARGET], current_prediction)
    performance = regression_metrics(current[TARGET], current_prediction)
    coverage = interval_coverage(current[TARGET], forecaster.predict_interval(current, data.test_groups), data.test_groups)

    y_true = actual_yield_drop(current, forecaster.alert_threshold)
    y_pred = forecaster.yield_drop_alert(current)
    alert_frame = alert_metric_frame(y_true, y_pred, data.test_groups)
    overall_fpr = float(alert_frame.overall["false_positive_rate"])
    false_alarm_multiple = (alert_frame.by_group["false_positive_rate"] / overall_fpr).round(3)

    capacity = current["ENRLT"].astype(float)
    planner = plan_evaluation(forecaster, current, plan_offers(forecaster, current, capacity), capacity)
    representation = latest_representation_report(paths.processed_dir)

    results = (
        checks.psi_checks(psi)
        + [checks.mae_check(performance["yield_mae_points"])]
        + checks.error_ratio_checks(current_ratios, previous_ratios)
        + checks.coverage_checks(coverage)
        + checks.false_alarm_checks(alert_frame.by_group, overall_fpr)
        + [checks.missing_check(missing_input_share(current, FEATURES), missing_input_share(reference, FEATURES))]
        + checks.representation_checks(representation)
        + checks.overrun_checks(planner)
    )

    evidently_dir = output_dir / "evidently"
    drift = data_drift_report(reference, current, evidently_dir / "data_drift.html")
    regression = regression_report(
        reference.assign(**{PREDICTION: reference_prediction}),
        current.assign(**{PREDICTION: current_prediction}),
        evidently_dir / "regression.html",
    )

    ratio_table = pd.DataFrame({"previous": previous_ratios, "current": current_ratios})
    figure_dir = output_dir / "figures"
    figures.psi_chart(psi, figure_dir / "psi.png")
    figures.error_ratio_chart(ratio_table, MAX_ERROR_RATIO, figure_dir / "error_ratios.png")
    figures.coverage_and_alert_chart(coverage, false_alarm_multiple, figure_dir / "coverage_alerts.png")
    figures.planner_chart(planner, figure_dir / "planner.png")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": metadata["model_version"],
        "approved_for_release": metadata["approved_for_release"],
        "reference_cycle": TRAIN_CYCLE,
        "current_cycle": TEST_CYCLE,
        "institutions": {"reference": len(reference), "current": len(current)},
        "performance": {
            "model": performance,
            "baseline": regression_metrics(current[TARGET], current["YIELD_LAG1"]),
            "reference_cross_validated": regression_metrics(reference[TARGET], reference_prediction),
        },
        "psi": psi.to_dict(),
        "evidently": {"data_drift": drift, "regression": regression},
        "error_ratios": ratio_table.to_dict(orient="index"),
        "interval_coverage": coverage.reset_index(names="group").to_dict(orient="records"),
        "alerts": {
            "overall_false_positive_rate": round(overall_fpr, 3),
            "by_group": alert_frame.by_group.round(3).reset_index(names="group").to_dict(orient="records"),
            "summary": alert_fairness_summary(y_true, y_pred, data.test_groups),
        },
        "offer_planner": {
            "goal_two_met": goal_two_met(planner),
            "by_group": planner.reset_index(names="group").to_dict(orient="records"),
        },
        "representation": representation.round(3).to_dict(orient="records"),
        "checks": [asdict(c) for c in results],
        "decision": checks.decision(results),
    }
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "history.jsonl").open("a") as history:
        history.write(json.dumps({
            "generated_at": summary["generated_at"],
            "current_cycle": TEST_CYCLE,
            "yield_mae_points": performance["yield_mae_points"],
            "error_ratios": current_ratios,
            "max_psi": float(psi.max()),
            "overall_status": summary["decision"]["overall_status"],
        }) + "\n")
    print(json.dumps({k: summary["decision"][k] for k in ("overall_status", "incident_required", "retrain_required",
                                                          "panel_review_required")}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check drift, performance and fairness against the model card triggers.")
    parser.add_argument("--output-dir", type=Path, default=None)
    main(parser.parse_args().output_dir)
