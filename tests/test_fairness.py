import pandas as pd
import pytest

from src.fairness.audit import breaches, error_ratio_vs_rest


def test_ratio_flags_group_with_larger_errors():
    groups = pd.Series(["A", "A", "B", "B"])
    y_true = [0.2, 0.2, 0.2, 0.2]
    y_pred = [0.25, 0.15, 0.21, 0.19]
    assert error_ratio_vs_rest(y_true, y_pred, groups, "A") == pytest.approx(5)
    assert "A" in breaches(y_true, y_pred, groups)
