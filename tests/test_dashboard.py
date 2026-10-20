import numpy as np
import pytest
from pydantic import ValidationError

from api.main import forecast, load_model
from api.schemas import Driver
from dashboard.core import (
    GROUP_ORDER,
    alert_fairness,
    contributions,
    cycle_from_row,
    describe_value,
    driver_sentence,
    enrollment_curve,
    error_by_group,
    error_ratio_table,
    load_dashboard_data,
    scenario,
    single_threshold_alerts,
)


@pytest.fixture(scope="module")
def dashboard_data():
    return load_dashboard_data()


@pytest.fixture(scope="module")
def model():
    return load_model()


@pytest.fixture(scope="module")
def samples(dashboard_data):
    return [dashboard_data.institution(unitid) for unitid in dashboard_data.summary["examples"].values()]


def test_api_forecast_matches_exported_holdout(model, samples):
    for row in samples:
        result = forecast(model, cycle_from_row(row))
        assert result.forecast_yield == pytest.approx(row["forecast_yield"], abs=1e-4)
        assert result.fairness_group == row["fairness_group"]


def test_scenario_rejects_offers_above_applicants(samples):
    cycle = cycle_from_row(samples[0])
    with pytest.raises(ValidationError):
        scenario(cycle, offers=cycle.applicants + 1)


def test_enrollment_curve_rises_with_offers(model, samples):
    cycle = cycle_from_row(samples[0])
    curve = enrollment_curve(model, cycle, np.linspace(1000, cycle.applicants, 6).astype(int))
    assert (curve["enrolled_lower"] <= curve["expected_enrolled"]).all()
    assert (curve["expected_enrolled"] <= curve["enrolled_upper"]).all()
    assert curve["expected_enrolled"].iloc[-1] > curve["expected_enrolled"].iloc[0]


def test_contributions_cover_every_input(samples):
    table = contributions(samples[0])
    assert len(table) == 12
    assert table["points"].abs().is_monotonic_decreasing


def test_plain_language_values():
    assert describe_value("YIELD_LAG1", 0.234) == "23%"
    assert describe_value("APPLCN_CHANGE", -0.1) == "-10% on last cycle"
    assert describe_value("LOG_APPLCN", np.log1p(2500)) == "2,500 applications"
    assert describe_value("LOCALE", None) == "not reported"
    sentence = driver_sentence(Driver(feature="Last cycle's yield", value=0.12, contribution_points=-1.24))
    assert sentence == "Last cycle's yield (12%) lowered the forecast by 1.2 percentage points."


def test_fairness_tables_match_published_report(dashboard_data):
    ratios = error_ratio_table(dashboard_data.holdout, "forecast_yield")["median_error_ratio"]
    assert ratios.to_dict() == {"HBCU": 1.01, "Under 1,000 students": 3.76, "Private for-profit": 4.75}
    assert list(error_by_group(dashboard_data.holdout, "forecast_yield").index) == GROUP_ORDER


def test_single_threshold_reproduces_deployed_alert(dashboard_data):
    holdout = dashboard_data.holdout
    alerts = single_threshold_alerts(holdout, 3.0)
    assert np.array_equal(alerts, holdout["alert_single_threshold"].to_numpy())
    _, _, summary = alert_fairness(holdout, alerts)
    assert summary["disparate_impact_ratio"] == 0.29
    assert not summary["meets_four_fifths_rule"]
