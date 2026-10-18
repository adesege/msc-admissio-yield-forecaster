import json

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict

from src.config import default_paths
from src.explain import figures
from src.fairness.alerts import actual_yield_drop, alert_fairness_summary, alert_metric_frame
from src.models.dataset import ModellingData, load_modelling_data
from src.models.features import LAG_COLUMN, TARGET, prepare_features
from src.models.forecaster import YieldForecaster
from src.models.mitigation import mitigation_fit_params
from src.models.tracking import configure_tracking
from src.models.tuning import cross_validator

RANDOM_STATE = 42


class DropScore(BaseEstimator):
    """Passes a precomputed drop score through so ThresholdOptimizer can choose group thresholds."""

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def predict(self, X):
        return np.asarray(X)[:, 0]


def out_of_fold_change(forecaster: YieldForecaster, data: ModellingData, mitigation: str) -> np.ndarray:
    return cross_val_predict(
        forecaster.change_model,
        prepare_features(data.train),
        (data.train[TARGET] - data.train[LAG_COLUMN]).to_numpy(),
        groups=data.train["UNITID"],
        cv=cross_validator(),
        params=mitigation_fit_params(mitigation, data.train_groups),
    )


def equalized_odds_alerts(forecaster: YieldForecaster, data: ModellingData, mitigation: str) -> np.ndarray:
    train_scores = pd.DataFrame({"drop_score": -out_of_fold_change(forecaster, data, mitigation)})
    test_scores = pd.DataFrame({"drop_score": -forecaster.predict_change(data.test)})
    optimizer = ThresholdOptimizer(
        estimator=DropScore().fit(train_scores),
        constraints="equalized_odds",
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method="predict",
    )
    optimizer.fit(train_scores, actual_yield_drop(data.train, forecaster.alert_threshold),
                  sensitive_features=data.train_groups)
    return optimizer.predict(test_scores, sensitive_features=data.test_groups, random_state=RANDOM_STATE)


def describe(y_true, y_pred, groups) -> dict:
    frame = alert_metric_frame(y_true, y_pred, groups)
    overall = frame.overall.to_dict()
    balanced = (overall["true_positive_rate"] + 1 - overall["false_positive_rate"]) / 2
    return {
        "overall": {**{k: round(float(v), 3) for k, v in overall.items()}, "balanced_accuracy": round(float(balanced), 3)},
        "by_group": frame.by_group.round(3).reset_index().to_dict(orient="records"),
        "summary": alert_fairness_summary(y_true, y_pred, groups),
    }


def alert_rates_chart(results: dict, path):
    order = list(figures.GROUP_COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(figures.FULL_WIDTH, 2.6), sharey=True)
    series = [("single_threshold", "Single 3-point threshold", figures.BLUE),
              ("equalized_odds_thresholds", "Equalized-odds thresholds", figures.ORANGE)]
    y = np.arange(len(order))
    for ax, metric, title in zip(axes, ("true_positive_rate", "false_positive_rate"),
                                 ("Real drops caught (TPR)", "False alarms (FPR)")):
        for offset, (key, label, color) in zip((0.2, -0.2), series):
            table = pd.DataFrame(results[key]["by_group"]).set_index("FAIRNESS_GROUP").loc[order]
            ax.barh(y + offset, 100 * table[metric], height=0.36, color=color, label=label)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Percent of institutions")
        ax.set_xlim(0, 100)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, order)
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return figures.save(fig, path)


def main() -> dict:
    paths = default_paths()
    data = load_modelling_data()
    forecaster = joblib.load(paths.root / "models" / "admissio_yield.joblib")
    mitigation = json.loads((paths.root / "models" / "metadata.json").read_text())["change_model"]
    y_true = actual_yield_drop(data.test, forecaster.alert_threshold)

    results = {
        "alert_threshold_points": 100 * forecaster.alert_threshold,
        "single_threshold": describe(y_true, forecaster.yield_drop_alert(data.test), data.test_groups),
        "equalized_odds_thresholds": describe(y_true, equalized_odds_alerts(forecaster, data, mitigation), data.test_groups),
    }
    report_path = paths.reports_dir / "alert_fairness.json"
    report_path.write_text(json.dumps(results, indent=2))
    figure = alert_rates_chart(results, paths.root / "docs" / "figures" / "fig_alert_rates.png")

    configure_tracking()
    with mlflow.start_run(run_name="alert-threshold-mitigation"):
        mlflow.set_tags({"run_type": "fairness_mitigation", "method": "ThresholdOptimizer equalized_odds"})
        for name in ("single_threshold", "equalized_odds_thresholds"):
            mlflow.log_metrics({f"{name}_{k}": float(v) for k, v in results[name]["summary"].items()})
            mlflow.log_metric(f"{name}_balanced_accuracy", results[name]["overall"]["balanced_accuracy"])
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(figure), artifact_path="figures")
    print(json.dumps({k: results[k]["summary"] for k in ("single_threshold", "equalized_odds_thresholds")}, indent=2))
    return results


if __name__ == "__main__":
    main()
