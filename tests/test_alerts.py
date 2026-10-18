import numpy as np
import pandas as pd
import pytest

from src.fairness.alerts import alert_fairness_summary, alert_metric_frame


def test_disparate_impact_ratio_and_four_fifths_rule():
    groups = pd.Series(["a"] * 4 + ["b"] * 4)
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    summary = alert_fairness_summary(y_true, y_pred, groups)
    assert summary["disparate_impact_ratio"] == pytest.approx(0.5)
    assert summary["demographic_parity_difference"] == pytest.approx(0.25)
    assert summary["equalized_odds_difference"] == pytest.approx(0.5)
    assert not summary["meets_four_fifths_rule"]


def test_metric_frame_reports_rates_by_group():
    groups = pd.Series(["a", "a", "b", "b"])
    frame = alert_metric_frame([1, 0, 1, 0], [1, 0, 1, 1], groups)
    assert frame.by_group.loc["b", "false_positive_rate"] == 1.0
    assert frame.by_group.loc["a", "true_positive_rate"] == 1.0
