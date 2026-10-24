import pytest

pytest.importorskip("evidently")

from src.monitoring.evidently_reports import PREDICTION, data_drift_report, regression_report


def test_drift_report_flags_a_shifted_input(modelling_data, tmp_path):
    reference, current = modelling_data.train, modelling_data.test.copy()
    current["YIELD_LAG1"] = (current["YIELD_LAG1"] + 0.3).clip(upper=1)
    result = data_drift_report(reference, current, tmp_path / "drift.html")
    assert (tmp_path / "drift.html").stat().st_size > 0
    assert "YIELD_LAG1" in result["drifted_columns"]


def test_regression_report_is_written(modelling_data, tmp_path):
    reference = modelling_data.train.assign(**{PREDICTION: modelling_data.train["YIELD_LAG1"]})
    current = modelling_data.test.assign(**{PREDICTION: modelling_data.test["YIELD_LAG1"]})
    regression_report(reference, current, tmp_path / "regression.html")
    assert (tmp_path / "regression.html").stat().st_size > 0
