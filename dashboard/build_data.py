import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from api.main import load_model
from src.config import default_paths
from src.explain.run import example_institutions
from src.explain.shap_values import shap_explanation
from src.fairness.alerts import actual_yield_drop
from src.fairness.report import equalized_odds_alerts
from src.models.dataset import load_model_ready_panel, prepare_modelling_data
from src.models.evaluation import intake_metrics, regression_metrics, sampling_noise_points
from src.models.features import FEATURES, fairness_group
from src.models.tracking import configure_tracking

DATA_DIR = Path(__file__).resolve().parent / "data"
REGISTERED_MODEL = "admissio-yield"


def with_previous_cycle(panel: pd.DataFrame) -> pd.DataFrame:
    ordered = panel.sort_values(["UNITID", "YEAR"]).copy()
    previous = ordered.groupby("UNITID")
    ordered["OFFERS_LAG1"] = previous["ADMSSN"].shift(1)
    ordered["ADMCON7_LAG1"] = previous["ADMCON7"].shift(1)
    return ordered


def holdout_table(model, test: pd.DataFrame, equalized_odds: np.ndarray) -> pd.DataFrame:
    forecaster = model.forecaster
    groups = fairness_group(test)
    interval = forecaster.predict_interval(test, groups)
    explanation = shap_explanation(forecaster, test, model.background)
    shap_points = pd.DataFrame(100 * explanation.values, columns=[f"shap_{f}" for f in FEATURES], index=test.index)
    table = pd.DataFrame(
        {
            "unitid": test["UNITID"].astype(int).astype(str),
            "institution": test["INSTNM"],
            "state": test["STABBR"],
            "fairness_group": groups,
            "HBCU_GROUP": test["HBCU_GROUP"],
            "SIZE_GROUP": test["SIZE_GROUP"],
            "CONTROL_GROUP": test["CONTROL_GROUP"],
            "applicants": test["APPLCN"].astype(int),
            "offers": test["ADMSSN"].astype(int),
            "last_cycle_applicants": test["APPLCN_LAG1"].astype(int),
            "last_cycle_offers": test["OFFERS_LAG1"].astype(int),
            "last_cycle_yield": test["YIELD_LAG1"],
            "test_policy": test["ADMCON7"].astype(int),
            "last_cycle_test_policy": test["ADMCON7_LAG1"].astype("Int64"),
            "control": test["CONTROL"].astype(int),
            "size_band": test["INSTSIZE"].astype(int),
            "locale": test["LOCALE"].astype("Int64"),
            "region": test["OBEREG"].astype("Int64"),
            "carnegie_class": test["C21BASIC"].astype("Int64"),
            "hbcu": test["HBCU_GROUP"].eq("HBCU"),
            "actual_yield": test["YIELD"],
            "actual_enrolled": test["ENRLT"].astype(int),
            "baseline_yield": test["YIELD_LAG1"],
            "forecast_yield": interval["yield"],
            "lower": interval["lower"],
            "upper": interval["upper"],
            "predicted_change": forecaster.predict_change(test),
            "actual_drop": actual_yield_drop(test, forecaster.alert_threshold),
            "alert_single_threshold": forecaster.yield_drop_alert(test),
            "alert_equalized_odds": np.asarray(equalized_odds, dtype=bool),
        }
    )
    return pd.concat([table, shap_points], axis=1).reset_index(drop=True)


def registered_mitigation_scores() -> list[dict]:
    configure_tracking()
    client = mlflow.MlflowClient()
    run_id = client.get_model_version_by_alias(REGISTERED_MODEL, "candidate").run_id
    path = client.download_artifacts(run_id, "mitigation_scores.csv")
    return pd.read_csv(path).round(3).to_dict(orient="records")


def summary(model, test: pd.DataFrame, table: pd.DataFrame, examples: dict, representation: pd.DataFrame) -> dict:
    base_value = shap_explanation(model.forecaster, test.head(1), model.background).base_values[0]
    return {
        "model": model.metadata,
        "shap_base_change_points": round(100 * float(base_value), 3),
        "holdout": {
            "baseline": {**regression_metrics(test["YIELD"], test["YIELD_LAG1"]), **intake_metrics(test, test["YIELD_LAG1"])},
            "model": {**regression_metrics(test["YIELD"], table["forecast_yield"]),
                      **intake_metrics(test, table["forecast_yield"].to_numpy())},
        },
        "sampling_noise_points": sampling_noise_points(test, fairness_group(test)).to_dict(),
        "examples": {label: str(int(row["UNITID"].iloc[0])) for label, row in examples.items()},
        "representation": representation.round(3).to_dict(orient="records"),
        "mitigation_scores": registered_mitigation_scores(),
    }


def latest_representation_report(processed_dir: Path) -> pd.DataFrame:
    latest = sorted((processed_dir / "representation_report").glob("*.parquet"))[-1]
    return pd.read_parquet(latest)


def main() -> None:
    paths = default_paths()
    model = load_model()
    data = prepare_modelling_data(with_previous_cycle(load_model_ready_panel()))
    mitigation = model.metadata["change_model"]
    table = holdout_table(model, data.test, equalized_odds_alerts(model.forecaster, data, mitigation))
    examples = example_institutions(data)
    representation = latest_representation_report(paths.processed_dir)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(DATA_DIR / "holdout.parquet", index=False)
    (DATA_DIR / "summary.json").write_text(
        json.dumps(summary(model, data.test, table, examples, representation), indent=2, default=str)
    )
    print(f"Wrote {len(table)} hold-out institutions to {DATA_DIR}")


if __name__ == "__main__":
    main()
