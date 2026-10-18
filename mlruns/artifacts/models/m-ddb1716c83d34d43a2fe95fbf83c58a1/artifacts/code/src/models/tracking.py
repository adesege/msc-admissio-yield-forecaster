import os
import re
from pathlib import Path

import mlflow
import pandas as pd
import skops.io
from mlflow import MlflowClient

from src.config import default_paths

EXPERIMENT_NAME = "admissio-yield-forecaster"
REGISTERED_MODEL_NAME = "admissio-yield"
CANDIDATE_ALIAS = "candidate"
CHAMPION_ALIAS = "champion"


def mlruns_dir() -> Path:
    return default_paths().root / "mlruns"


def configure_tracking() -> str:
    mlruns = mlruns_dir()
    mlruns.mkdir(parents=True, exist_ok=True)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{mlruns / 'mlflow.db'}")
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=(mlruns / "artifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    return tracking_uri


def log_search_candidates(results: pd.DataFrame, family: str, stage: str) -> None:
    for _, row in results.iterrows():
        with mlflow.start_run(run_name=f"{family}-{stage}-{int(row['rank_test_score'])}", nested=True):
            mlflow.set_tags({"family": family, "search_stage": stage, "run_type": "tuning_candidate"})
            mlflow.log_params({key.removeprefix("model__"): value for key, value in row["params"].items()})
            mlflow.log_metrics(
                {
                    "cv_mae_points": float(-100 * row["mean_test_score"]),
                    "cv_mae_std_points": float(100 * row["std_test_score"]),
                    "fit_seconds": float(row["mean_fit_time"]),
                }
            )


def metric_name(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", text.lower()).strip("_")


def log_flat_metrics(metrics: dict, prefix: str = "") -> None:
    numeric = {
        metric_name(f"{prefix}{key}"): float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    mlflow.log_metrics(numeric)


def log_forecaster(forecaster, input_example: pd.DataFrame) -> None:
    trusted = skops.io.get_untrusted_types(data=skops.io.dumps(forecaster))
    mlflow.sklearn.log_model(
        forecaster,
        name="model",
        code_paths=["src"],
        input_example=input_example,
        skops_trusted_types=trusted,
    )


def register_model(run_id: str, approved: bool, version_label: str) -> int:
    client = MlflowClient()
    version = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL_NAME)
    alias = CHAMPION_ALIAS if approved else CANDIDATE_ALIAS
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, alias, version.version)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "semantic_version", version_label)
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, version.version, "release_status",
        "approved" if approved else "not approved: pending Ethics Review Panel",
    )
    return int(version.version)
