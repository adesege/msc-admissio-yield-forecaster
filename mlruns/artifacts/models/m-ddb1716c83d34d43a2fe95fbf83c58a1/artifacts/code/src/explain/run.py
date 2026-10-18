import json

import joblib
import mlflow
import numpy as np
import pandas as pd

from src.config import default_paths
from src.explain import figures
from src.explain.counterfactuals import counterfactual_offers
from src.explain.local import LocalExplainer
from src.explain.sensitivity import (
    missing_input_sensitivity,
    perturbation_sensitivity,
    policy_sensitivity,
    seed_stability,
)
from src.explain.shap_values import shap_explanation
from src.models.dataset import ModellingData, load_modelling_data
from src.models.mitigation import group_balanced_weights
from src.models.tracking import configure_tracking

FAMILY_LABELS = {"ridge": "Ridge regression", "random_forest": "Random forest", "lightgbm": "LightGBM"}


def example_institutions(data: ModellingData) -> dict[str, pd.DataFrame]:
    test = data.test
    hbcu = test[test["HBCU_GROUP"] == "HBCU"].sort_values("ADMSSN", ascending=False).head(1)
    small_for_profit = test[(test["CONTROL_GROUP"] == "Private for-profit") & (test["SIZE_GROUP"] == "Under 1,000")]
    small_for_profit = small_for_profit.iloc[[len(small_for_profit) // 2]] if len(small_for_profit) else small_for_profit
    large_public = test[(test["CONTROL"] == 1) & (test["INSTSIZE"] == 5)]
    large_public = large_public.iloc[[(large_public["YIELD_LAG1"] - large_public["YIELD_LAG1"].median()).abs().argmin()]]
    return {"HBCU": hbcu, "Small for-profit": small_for_profit, "Large public": large_public}


def change_model_fit_params(selected: str, data: ModellingData) -> dict:
    if selected == "group_reweighted":
        return {"model__sample_weight": group_balanced_weights(data.train_groups)}
    if selected.startswith("bounded_group_loss"):
        return {"model__sensitive_features": data.train_groups.to_numpy()}
    return {}


def shap_tables(forecaster, data: ModellingData) -> dict:
    explanation = shap_explanation(forecaster, data.test, data.train)
    values = pd.DataFrame(100 * np.abs(explanation.values), columns=explanation.feature_names)
    overall = values.mean().sort_values(ascending=False).round(3)
    by_group = values.groupby(data.test_groups.to_numpy()).mean().round(3).T
    return {"mean_abs_shap_points": overall.to_dict(), "mean_abs_shap_points_by_group": by_group.to_dict()}


def main() -> dict:
    paths = default_paths()
    figure_dir = paths.root / "docs" / "figures"
    data = load_modelling_data()
    forecaster = joblib.load(paths.root / "models" / "admissio_yield.joblib")
    summary = json.loads((paths.reports_dir / "training_summary.json").read_text())
    selected = summary["selected_mitigation"]

    comparison = pd.DataFrame(
        [{"model": "Naive baseline", "cv_mae": np.nan, "holdout_mae": summary["baseline"]["yield_mae_points"]}]
        + [
            {"model": FAMILY_LABELS[f["family"]], "cv_mae": f["cv_mae_points"], "holdout_mae": f["holdout"]["yield_mae_points"]}
            for f in summary["families"]
        ]
        + [
            {
                "model": f"Selected: {FAMILY_LABELS[summary['champion_family']]}, {selected.replace('_', ' ')}",
                "cv_mae": next(s["cv_yield_mae_points"] for s in summary["mitigation_scores"] if s["candidate"] == selected),
                "holdout_mae": summary["holdout"]["yield_mae_points"],
            }
        ]
    )
    holdout = pd.read_csv(paths.reports_dir / "holdout_predictions.csv")
    ratios = pd.DataFrame(
        {
            "group": [r["group"] for r in summary["group_error_table"]],
            "baseline": [r["median_error_ratio"] for r in summary["baseline_group_error_table"]],
            "model": [r["median_error_ratio"] for r in summary["group_error_table"]],
        }
    )
    mitigation = pd.DataFrame(summary["mitigation_scores"]).set_index("candidate")
    coverage = pd.DataFrame(summary["interval_coverage"])

    examples = example_institutions(data)
    local = LocalExplainer(forecaster, data.train)
    lime_results = {}
    for label, row in examples.items():
        lime_results[label] = {"institution": row["INSTNM"].iloc[0], "unitid": int(row["UNITID"].iloc[0]),
                               "actual_yield": float(row["YIELD"].iloc[0]), "last_cycle_yield": float(row["YIELD_LAG1"].iloc[0]),
                               **local.explain(row)}

    counterfactual_results = {}
    for label, row in examples.items():
        last_yield = float(row["YIELD_LAG1"].iloc[0])
        target = (last_yield, min(1.0, last_yield + 0.05))
        options = counterfactual_offers(forecaster, data.train, row, target)
        counterfactual_results[label] = {
            "institution": row["INSTNM"].iloc[0],
            "applicants": int(row["APPLCN"].iloc[0]),
            "current_offers": int(row["ADMSSN"].iloc[0]),
            "current_admit_rate": round(float(row["ADMIT_RATE"].iloc[0]), 3),
            "current_forecast_yield": round(float(forecaster.predict(row)[0]), 4),
            "current_implied_enrolled": int(round(row["ADMSSN"].iloc[0] * forecaster.predict(row)[0])),
            "target_yield_range": [round(v, 4) for v in target],
            "options": options.round(4).to_dict(orient="records"),
        }

    perturbation = perturbation_sensitivity(forecaster, data.test)
    policy = policy_sensitivity(forecaster, data.test)
    missing = missing_input_sensitivity(forecaster, data.test)
    seeds = seed_stability(forecaster, data.train, data.test, change_model_fit_params(selected, data))

    produced = [
        figures.model_comparison(comparison, figure_dir / "fig_model_comparison.png"),
        figures.predicted_vs_actual(holdout, figure_dir / "fig_predicted_vs_actual.png"),
        figures.fairness_ratios(ratios, figure_dir / "fig_fairness_ratios.png"),
        figures.mitigation_tradeoff(mitigation, selected, figure_dir / "fig_mitigation_tradeoff.png"),
        figures.lime_panels([(f"{k}: {v['institution']}", v) for k, v in lime_results.items()],
                            figure_dir / "fig_lime_examples.png"),
        figures.sensitivity_chart(perturbation, figure_dir / "fig_sensitivity.png"),
        figures.interval_coverage_chart(coverage, figure_dir / "fig_interval_coverage.png"),
    ]

    report = {
        "model_comparison": comparison.round(2).to_dict(orient="records"),
        "shap": shap_tables(forecaster, data),
        "lime": lime_results,
        "counterfactuals": counterfactual_results,
        "sensitivity": {
            "perturbation": perturbation.to_dict(orient="records"),
            "test_policy": policy.to_dict(orient="records"),
            "missing_input": missing.to_dict(orient="records"),
            "seed_stability": seeds.to_dict(orient="records"),
            "seed_mae_range_points": round(float(seeds["yield_mae_points"].max() - seeds["yield_mae_points"].min()), 2),
        },
    }
    report_path = paths.reports_dir / "explainability.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    configure_tracking()
    with mlflow.start_run(run_name="explainability-and-fairness"):
        mlflow.set_tags({"run_type": "explainability", "model_version": summary["model_version"],
                         "training_run_id": summary["parent_run_id"]})
        mlflow.log_metric("seed_mae_range_points", report["sensitivity"]["seed_mae_range_points"])
        mlflow.log_artifact(str(report_path))
        for path in produced:
            mlflow.log_artifact(str(path), artifact_path="figures")
    print(json.dumps({k: report[k] for k in ("model_comparison", "shap")}, indent=2))
    return report


if __name__ == "__main__":
    main()
