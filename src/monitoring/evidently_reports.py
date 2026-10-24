from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Regression, Report
from evidently.presets import DataDriftPreset, RegressionPreset

from src.models.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

PREDICTION = "PREDICTION"


def _dataset(frame: pd.DataFrame, with_target: bool) -> Dataset:
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ([TARGET, PREDICTION] if with_target else [])
    data = frame[columns].copy()
    data[CATEGORICAL_FEATURES] = data[CATEGORICAL_FEATURES].astype("string")
    definition = DataDefinition(
        numerical_columns=NUMERIC_FEATURES + ([TARGET, PREDICTION] if with_target else []),
        categorical_columns=CATEGORICAL_FEATURES,
        regression=[Regression(target=TARGET, prediction=PREDICTION)] if with_target else None,
    )
    return Dataset.from_pandas(data, data_definition=definition)


def _drift_results(snapshot) -> dict:
    share, drifted = None, []
    for metric in snapshot.dict()["metrics"]:
        name, config = metric["metric_name"], metric["config"]
        if name.startswith("DriftedColumnsCount"):
            share = float(metric["value"]["share"])
        elif name.startswith("ValueDrift"):
            value, threshold = float(metric["value"]), config["threshold"]
            is_p_value = "p_value" in config["method"]
            if (value < threshold) if is_p_value else (value >= threshold):
                drifted.append(config["column"])
    return {"drifted_column_share": share, "drifted_columns": sorted(drifted)}


def data_drift_report(reference: pd.DataFrame, current: pd.DataFrame, path: Path) -> dict:
    snapshot = Report([DataDriftPreset()]).run(current_data=_dataset(current, False),
                                               reference_data=_dataset(reference, False))
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(path))
    return {"report": path.name, **_drift_results(snapshot)}


def regression_report(reference: pd.DataFrame, current: pd.DataFrame, path: Path) -> dict:
    snapshot = Report([RegressionPreset()]).run(current_data=_dataset(current, True),
                                                reference_data=_dataset(reference, True))
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(path))
    return {"report": path.name}
