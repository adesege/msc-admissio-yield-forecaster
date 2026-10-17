import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.models.evaluation import acceptance_check, regression_metrics
from src.models.features import FEATURES, LeakageError, assert_no_leakage, prepare_features
from src.models.mitigation import BoundedGroupLossRegressor, group_balanced_weights


def test_leakage_guard_rejects_columns_measured_after_enrolment():
    with pytest.raises(LeakageError):
        assert_no_leakage(["ADMIT_RATE", "ENRLFT", "SATPCT"])


def test_feature_set_contains_no_leakage():
    assert_no_leakage(FEATURES)


def test_split_trains_on_2023_and_tests_on_2024(modelling_data):
    assert set(modelling_data.train["YEAR"]) == {2023}
    assert set(modelling_data.test["YEAR"]) == {2024}
    assert len(modelling_data.train) == len(modelling_data.test) == 1852


def test_features_exclude_target_and_fairness_labels(modelling_data):
    features = prepare_features(modelling_data.train)
    assert list(features.columns) == FEATURES
    assert not {"YIELD", "ENRLT", "HBCU", "HBCU_GROUP"} & set(features.columns)


def test_forecaster_returns_valid_yields_and_intervals(small_forecaster, modelling_data):
    test = modelling_data.test
    intervals = small_forecaster.predict_interval(test, modelling_data.test_groups)
    assert intervals["yield"].between(0, 1).all()
    assert (intervals["lower"] <= intervals["yield"]).all()
    assert (intervals["upper"] >= intervals["yield"]).all()


def test_forecaster_survives_serialisation(small_forecaster, modelling_data, tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(small_forecaster, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(restored.predict(modelling_data.test), small_forecaster.predict(modelling_data.test))


def test_bounded_group_loss_mixture_is_deterministic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = X[:, 0] * 0.1 + rng.normal(scale=0.05, size=300)
    groups = np.where(rng.random(300) < 0.2, "small", "large")
    model = BoundedGroupLossRegressor(Ridge(), upper_bound=0.1, max_iter=10).fit(X, y, sensitive_features=groups)
    assert np.isclose(model.weights_.sum(), 1.0)
    np.testing.assert_array_equal(model.predict(X), model.predict(X))


def test_group_balanced_weights_average_one():
    weights = group_balanced_weights(pd.Series(["a"] * 90 + ["b"] * 10))
    assert np.isclose(weights.mean(), 1.0)
    assert weights[-1] == pytest.approx(9 * weights[0])


def test_acceptance_requires_every_check():
    metrics = {"yield_mae_points": 4.5, "share_intake_missed_by_over_10pct": 0.4}
    assert acceptance_check(metrics, {"HBCU": 1.1}, 0.5)["approved_for_release"]
    assert not acceptance_check(metrics, {"HBCU": 1.3}, 0.5)["approved_for_release"]


def test_regression_metrics_in_percentage_points():
    metrics = regression_metrics([0.3, 0.5, 0.7], [0.4, 0.6, 0.8])
    assert metrics["yield_mae_points"] == pytest.approx(10.0)
