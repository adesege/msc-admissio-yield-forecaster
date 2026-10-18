import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

from src.config import default_paths
from src.explain.shap_values import background_sample
from src.fairness.alerts import actual_yield_drop, alert_fairness_summary, alert_metric_frame
from src.models.dataset import ModellingData, load_modelling_data
from src.models.evaluation import (
    acceptance_check,
    group_error_ratios,
    group_error_table,
    intake_metrics,
    interval_coverage,
    regression_metrics,
    sampling_noise_points,
)
from src.models.features import FEATURES, LAG_COLUMN, NUMERIC_FEATURES, TARGET, prepare_features
from src.models.forecaster import YieldForecaster
from src.models.mitigation import BoundedGroupLossRegressor, group_balanced_weights
from src.models.tracking import (
    configure_tracking,
    log_flat_metrics,
    log_forecaster,
    log_search_candidates,
    register_model,
)
from src.models.tuning import MODEL_FAMILIES, ModelFamily, cross_validator, coarse_search, fine_search, tuned_pipeline

MODEL_VERSION = "0.3.0"
BOUNDED_LOSS_LIMITS = (0.06, 0.08, 0.10)
MITIGATION_MAE_TOLERANCE = 1.05


@dataclass
class Candidate:
    name: str
    family: str
    change_model: Pipeline
    fit_params: dict = field(default_factory=dict)
    oof_prediction: np.ndarray | None = None

    def forecaster(self) -> YieldForecaster:
        return YieldForecaster(self.change_model)


def change_target(frame: pd.DataFrame) -> np.ndarray:
    return (frame[TARGET] - frame[LAG_COLUMN]).to_numpy()


def evaluate_predictions(frame: pd.DataFrame, prediction) -> dict:
    return {
        **regression_metrics(frame[TARGET], prediction),
        **intake_metrics(frame, prediction),
        "error_ratio": group_error_ratios(frame, frame[TARGET], prediction),
    }


def out_of_fold_yield(candidate: Candidate, data: ModellingData) -> np.ndarray:
    change = cross_val_predict(
        candidate.change_model,
        prepare_features(data.train),
        change_target(data.train),
        groups=data.train["UNITID"],
        cv=cross_validator(),
        params=candidate.fit_params,
        n_jobs=-1,
    )
    return np.clip(data.train[LAG_COLUMN].to_numpy() + change, 0.0, 1.0)


def mitigation_candidates(family: ModelFamily, params: dict, data: ModellingData) -> list[Candidate]:
    base_params = {key.removeprefix("model__"): value for key, value in params.items()}
    candidates = [
        Candidate("unmitigated", family.name, tuned_pipeline(family, params)),
        Candidate(
            "group_reweighted",
            family.name,
            tuned_pipeline(family, params),
            {"model__sample_weight": group_balanced_weights(data.train_groups)},
        ),
    ]
    for limit in BOUNDED_LOSS_LIMITS:
        mitigated = Pipeline(
            [
                ("prepare", family.pipeline().named_steps["prepare"]),
                ("model", BoundedGroupLossRegressor(family.estimator().set_params(**base_params), upper_bound=limit)),
            ]
        )
        candidates.append(
            Candidate(
                f"bounded_group_loss_{limit:.2f}",
                family.name,
                mitigated,
                {"model__sensitive_features": data.train_groups.to_numpy()},
            )
        )
    return candidates


def select_mitigation(scores: pd.DataFrame) -> str:
    reference = scores.loc["unmitigated", "cv_yield_mae_points"]
    eligible = scores[scores["cv_yield_mae_points"] <= reference * MITIGATION_MAE_TOLERANCE]
    return eligible["cv_worst_error_ratio"].idxmin()


def naive_baseline(data: ModellingData) -> dict:
    with mlflow.start_run(run_name="naive-baseline", nested=True):
        mlflow.set_tags({"family": "naive", "run_type": "baseline"})
        metrics = evaluate_predictions(data.test, data.test[LAG_COLUMN])
        log_flat_metrics(metrics, "holdout_")
        log_flat_metrics(metrics["error_ratio"], "holdout_error_ratio_")
        return metrics


def tune_family(family: ModelFamily, data: ModellingData, quick: bool) -> dict:
    X, y, institutions = prepare_features(data.train), change_target(data.train), data.train["UNITID"]
    with mlflow.start_run(run_name=f"{family.name}-tuning", nested=True):
        mlflow.set_tags({"family": family.name, "run_type": "family_summary"})
        coarse = coarse_search(family, X, y, institutions, iterations=4 if quick else None)
        log_search_candidates(coarse.results, family.name, "coarse")
        fine = fine_search(family, coarse, X, y, institutions)
        log_search_candidates(fine.results, family.name, "fine")
        best = fine if fine.best_mae_points <= coarse.best_mae_points else coarse

        forecaster = YieldForecaster(tuned_pipeline(family, best.best_params)).fit(data.train, data.train[TARGET])
        holdout = evaluate_predictions(data.test, forecaster.predict(data.test))
        mlflow.log_params({key.removeprefix("model__"): value for key, value in best.best_params.items()})
        mlflow.log_metrics(
            {
                "cv_mae_points": best.best_mae_points,
                "coarse_cv_mae_points": coarse.best_mae_points,
                "fine_cv_mae_points": fine.best_mae_points,
            }
        )
        log_flat_metrics(holdout, "holdout_")
        log_flat_metrics(holdout["error_ratio"], "holdout_error_ratio_")
        for stage in (coarse, fine):
            path = default_paths().reports_dir / "tuning" / f"{family.name}_{stage.stage}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            stage.results.drop(columns="params").to_csv(path, index=False)
            mlflow.log_artifact(str(path), artifact_path="cv_results")
        return {
            "family": family,
            "params": best.best_params,
            "cv_mae_points": round(best.best_mae_points, 2),
            "coarse_cv_mae_points": round(coarse.best_mae_points, 2),
            "fine_cv_mae_points": round(fine.best_mae_points, 2),
            "coarse_candidates": len(coarse.results),
            "fine_candidates": len(fine.results),
            "holdout": holdout,
        }


def score_mitigations(candidates: list[Candidate], data: ModellingData) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        with mlflow.start_run(run_name=f"mitigation-{candidate.name}", nested=True):
            mlflow.set_tags({"family": candidate.family, "run_type": "mitigation", "mitigation": candidate.name})
            candidate.oof_prediction = out_of_fold_yield(candidate, data)
            cv = regression_metrics(data.train[TARGET], candidate.oof_prediction)
            ratios = group_error_ratios(data.train, data.train[TARGET], candidate.oof_prediction)
            forecaster = candidate.forecaster().fit(data.train, data.train[TARGET], **candidate.fit_params)
            holdout = evaluate_predictions(data.test, forecaster.predict(data.test))
            log_flat_metrics(cv, "cv_")
            log_flat_metrics(ratios, "cv_error_ratio_")
            log_flat_metrics(holdout, "holdout_")
            log_flat_metrics(holdout["error_ratio"], "holdout_error_ratio_")
            rows.append(
                {
                    "candidate": candidate.name,
                    "cv_yield_mae_points": cv["yield_mae_points"],
                    "cv_worst_error_ratio": max(ratios.values()),
                    **{f"cv_ratio_{k}": v for k, v in ratios.items()},
                    "holdout_yield_mae_points": holdout["yield_mae_points"],
                    **{f"holdout_ratio_{k}": v for k, v in holdout["error_ratio"].items()},
                }
            )
    return pd.DataFrame(rows).set_index("candidate")


def alert_report(forecaster: YieldForecaster, data: ModellingData) -> dict:
    y_true = actual_yield_drop(data.test, forecaster.alert_threshold)
    y_pred = forecaster.yield_drop_alert(data.test)
    frame = alert_metric_frame(y_true, y_pred, data.test_groups)
    return {
        "threshold_points": 100 * forecaster.alert_threshold,
        "overall": frame.overall.round(3).to_dict(),
        "by_group": frame.by_group.round(3).reset_index().to_dict(orient="records"),
        "summary": alert_fairness_summary(y_true, y_pred, data.test_groups),
    }


def write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def main(quick: bool = False) -> dict:
    paths = default_paths()
    tracking_uri = configure_tracking()
    data = load_modelling_data()
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with mlflow.start_run(run_name=f"training-{trained_at}") as parent:
        mlflow.set_tags({"run_type": "training", "model_version": MODEL_VERSION, "split": "train 2023, test 2024"})
        mlflow.log_params(
            {
                "train_cycle_rows": len(data.train),
                "test_cycle_rows": len(data.test),
                "features": ",".join(FEATURES),
                "target": "yield change from last cycle",
                "cv": "GroupKFold, 5 folds by UNITID",
            }
        )

        baseline = naive_baseline(data)
        families = [tune_family(family, data, quick) for family in MODEL_FAMILIES]
        champion = min(families, key=lambda f: f["cv_mae_points"])

        candidates = mitigation_candidates(champion["family"], champion["params"], data)
        mitigation_scores = score_mitigations(candidates, data)
        selected_name = select_mitigation(mitigation_scores)
        selected = next(c for c in candidates if c.name == selected_name)

        forecaster = selected.forecaster().fit(data.train, data.train[TARGET], **selected.fit_params)
        forecaster.calibrate_intervals(data.train[TARGET].to_numpy() - selected.oof_prediction, data.train_groups)

        prediction = forecaster.predict(data.test)
        holdout = evaluate_predictions(data.test, prediction)
        intervals = forecaster.predict_interval(data.test, data.test_groups)
        coverage = interval_coverage(data.test[TARGET], intervals, data.test_groups)
        acceptance = acceptance_check(holdout, holdout["error_ratio"], baseline["share_intake_missed_by_over_10pct"])
        alerts = alert_report(forecaster, data)
        baseline_alerts = alert_fairness_summary(
            actual_yield_drop(data.test, forecaster.alert_threshold),
            np.zeros(len(data.test), dtype=bool),
            data.test_groups,
        )

        mlflow.set_tags({"champion_family": champion["family"].name, "mitigation": selected_name,
                         "approved_for_release": str(acceptance["approved_for_release"])})
        log_flat_metrics(holdout, "holdout_")
        log_flat_metrics(holdout["error_ratio"], "holdout_error_ratio_")
        log_flat_metrics(alerts["summary"], "alert_")

        predictions = data.test[["UNITID", "INSTNM", "YEAR", TARGET, LAG_COLUMN, "ADMSSN", "ENRLT"]].assign(
            FAIRNESS_GROUP=data.test_groups,
            YIELD_PRED=prediction,
            YIELD_LOWER=intervals["lower"],
            YIELD_UPPER=intervals["upper"],
            YIELD_DROP_ALERT=forecaster.yield_drop_alert(data.test),
        )
        predictions_path = paths.reports_dir / "holdout_predictions.csv"
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(predictions_path, index=False)
        mitigation_scores.to_csv(paths.reports_dir / "mitigation_scores.csv")
        mlflow.log_artifact(str(predictions_path))
        mlflow.log_artifact(str(paths.reports_dir / "mitigation_scores.csv"))

        input_example = data.test[FEATURES].head(3)
        log_forecaster(forecaster, input_example)
        registry_version = register_model(parent.info.run_id, acceptance["approved_for_release"], MODEL_VERSION)

    model_dir = paths.root / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(forecaster, model_dir / "admissio_yield.joblib")
    background_sample(data.train).to_parquet(model_dir / "shap_background.parquet", index=False)

    summary = {
        "model_version": MODEL_VERSION,
        "trained_at": trained_at,
        "tracking_uri": tracking_uri,
        "parent_run_id": parent.info.run_id,
        "registered_model": {"name": "admissio-yield", "version": registry_version},
        "rows": {"train_2023": len(data.train), "test_2024": len(data.test)},
        "group_counts": {
            "train": data.train_groups.value_counts().to_dict(),
            "test": data.test_groups.value_counts().to_dict(),
        },
        "baseline": baseline,
        "families": [
            {key: value for key, value in f.items() if key != "family"} | {"family": f["family"].name}
            for f in families
        ],
        "champion_family": champion["family"].name,
        "mitigation_scores": mitigation_scores.reset_index().to_dict(orient="records"),
        "selected_mitigation": selected_name,
        "holdout": holdout,
        "group_error_table": group_error_table(data.test, data.test[TARGET], prediction).to_dict(orient="records"),
        "baseline_group_error_table": group_error_table(data.test, data.test[TARGET], data.test[LAG_COLUMN]).to_dict(orient="records"),
        "sampling_noise_points": sampling_noise_points(data.test, data.test_groups).to_dict(),
        "interval_halfwidths_points": {k: round(100 * v, 2) for k, v in forecaster.interval_halfwidths_.items()},
        "interval_coverage": coverage.reset_index(names="group").to_dict(orient="records"),
        "alerts": alerts,
        "baseline_alerts_never_fire": baseline_alerts,
        "acceptance": acceptance,
    }
    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": trained_at,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target": "YIELD",
        "train_cycle": 2023,
        "change_model": selected_name,
        "family": champion["family"].name,
        "registered_model_version": registry_version,
        "approved_for_release": acceptance["approved_for_release"],
        "holdout_yield_mae_points": holdout["yield_mae_points"],
        "interval_coverage": forecaster.interval_coverage,
        "alert_threshold": forecaster.alert_threshold,
    }
    write_json(paths.reports_dir / "training_summary.json", summary)
    write_json(model_dir / "metadata.json", metadata)
    print(json.dumps({k: summary[k] for k in ("champion_family", "selected_mitigation", "holdout", "acceptance")}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune, mitigate, evaluate and register the yield forecaster.")
    parser.add_argument("--quick", action="store_true", help="Run a short coarse search for smoke tests and CI.")
    main(quick=parser.parse_args().quick)
