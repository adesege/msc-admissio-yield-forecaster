import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.base import clone

from src.config import default_paths
from src.explain.shap_values import background_sample
from src.models.dataset import load_model_ready_panel
from src.models.evaluation import acceptance_check, group_error_ratios, intake_metrics, regression_metrics
from src.models.features import FEATURES, LAG_COLUMN, TARGET, add_model_features, fairness_group, modelling_rows
from src.models.forecaster import YieldForecaster
from src.models.mitigation import mitigation_fit_params
from src.models.tracking import REGISTERED_MODEL_NAME, configure_tracking, log_flat_metrics, log_forecaster
from src.models.validation import cross_validated_yield

CHALLENGER_ALIAS = "challenger"


def next_patch(version: str) -> str:
    major, minor, patch = (int(part) for part in version.split("."))
    return f"{major}.{minor}.{patch + 1}"


def pooled_rows(min_cycle: int | None = None) -> pd.DataFrame:
    rows = modelling_rows(add_model_features(load_model_ready_panel()))
    return rows if min_cycle is None else rows[rows["YEAR"] >= min_cycle].reset_index(drop=True)


def evaluate(rows: pd.DataFrame, prediction) -> dict:
    return {
        **regression_metrics(rows[TARGET], prediction),
        **intake_metrics(rows, prediction),
        "error_ratio": group_error_ratios(rows, rows[TARGET], prediction),
    }


def register_challenger(run_id: str, version_label: str, passed: bool) -> int:
    client = MlflowClient()
    version = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL_NAME)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHALLENGER_ALIAS, version.version)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "semantic_version", version_label)
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, version.version, "release_status",
        "challenger: passed automated gates, pending Ethics Review Panel" if passed
        else "challenger: failed automated gates, pending Ethics Review Panel",
    )
    return int(version.version)


def main(output_dir: Path | None = None, register: bool = True) -> dict:
    paths = default_paths()
    model_dir = paths.root / "models"
    output_dir = output_dir or model_dir / "challenger"
    champion = joblib.load(model_dir / "admissio_yield.joblib")
    metadata = json.loads((model_dir / "metadata.json").read_text())
    mitigation = metadata["change_model"]

    rows = pooled_rows()
    groups = fairness_group(rows)
    cycles = sorted(int(c) for c in rows["YEAR"].unique())
    oof = cross_validated_yield(champion, rows, mitigation)
    challenger = YieldForecaster(clone(champion.change_model)).fit(
        rows, rows[TARGET], **mitigation_fit_params(mitigation, groups)
    )
    challenger.calibrate_intervals(rows[TARGET].to_numpy() - oof, groups)

    metrics = evaluate(rows, oof)
    baseline = evaluate(rows, rows[LAG_COLUMN])
    acceptance = acceptance_check(metrics, metrics["error_ratio"], baseline["share_intake_missed_by_over_10pct"])
    version_label = next_patch(metadata["model_version"])
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    registry_version = None
    if register:
        configure_tracking()
        with mlflow.start_run(run_name=f"retraining-{trained_at}") as run:
            mlflow.set_tags({"run_type": "retraining", "model_version": version_label,
                             "cycles": ",".join(map(str, cycles)), "mitigation": mitigation,
                             "approved_for_release": "False"})
            mlflow.log_params({"rows": len(rows), "cv": "GroupKFold, 5 folds by UNITID across pooled cycles"})
            log_flat_metrics(metrics, "cv_")
            log_flat_metrics(metrics["error_ratio"], "cv_error_ratio_")
            log_flat_metrics(baseline, "baseline_")
            log_forecaster(challenger, rows[FEATURES].head(3))
            registry_version = register_challenger(run.info.run_id, version_label,
                                                   all(acceptance["checks"].values()))

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(challenger, output_dir / "admissio_yield.joblib")
    background_sample(rows).to_parquet(output_dir / "shap_background.parquet", index=False)
    challenger_metadata = {
        **metadata,
        "model_version": version_label,
        "trained_at": trained_at,
        "train_cycle": cycles,
        "registered_model_version": registry_version,
        "approved_for_release": False,
        "cv_yield_mae_points": metrics["yield_mae_points"],
        "status": "challenger",
    }
    (output_dir / "metadata.json").write_text(json.dumps(challenger_metadata, indent=2, default=str))
    summary = {
        "model_version": version_label,
        "trained_at": trained_at,
        "cycles": cycles,
        "rows": len(rows),
        "champion": {"model_version": metadata["model_version"],
                     "holdout_yield_mae_points": metadata["holdout_yield_mae_points"]},
        "challenger_cross_validated": metrics,
        "baseline_on_same_rows": baseline,
        "acceptance": acceptance,
        "registered_model_version": registry_version,
        "promotion": "pending Ethics Review Panel",
    }
    (output_dir / "retraining_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("model_version", "cycles", "acceptance")}, indent=2, default=str))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain on every available cycle and register a challenger.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-register", action="store_true", help="Skip the MLflow registry, for tests.")
    arguments = parser.parse_args()
    main(arguments.output_dir, register=not arguments.no_register)
