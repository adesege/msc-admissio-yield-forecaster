import json

import numpy as np
import pandas as pd
import pytest

from src.monitoring import checks
from src.monitoring.drift import categorical_psi, feature_psi, numeric_psi
from src.monitoring.site import incident_body, render
from src.planning.offers import OVERRUN_RISK, plan_evaluation, plan_offers


def test_psi_is_zero_for_identical_samples_and_grows_with_shift():
    rng = np.random.default_rng(0)
    sample = pd.Series(rng.normal(size=2000))
    assert numeric_psi(sample, sample) == pytest.approx(0, abs=1e-9)
    assert numeric_psi(sample, sample + 0.5) > checks.PSI_RED
    assert categorical_psi(pd.Series([1, 1, 2, 2]), pd.Series([1, 1, 1, 2])) > 0


def test_feature_psi_covers_every_input(modelling_data):
    psi = feature_psi(modelling_data.train, modelling_data.test)
    assert len(psi) == 12
    assert psi.max() < checks.PSI_RED


def test_error_ratio_is_red_only_after_two_cycles_above_limit():
    results = {(c.signal, c.group): c for c in checks.error_ratio_checks({"A": 1.5, "B": 1.5}, {"A": 1.4, "B": 1.1})}
    assert results[("error_ratio", "A")].status == "red"
    assert results[("error_ratio", "B")].status == "amber"
    assert results[("error_ratio_change", "B")].status == "amber"


def test_missing_input_check_is_relative_to_training_cycle():
    assert checks.missing_check(0.083, 0.083).status == "green"
    assert checks.missing_check(0.14, 0.083).status == "red"


def test_fairness_reds_go_to_the_panel_and_error_reds_to_retraining():
    fairness = checks.decision([checks.Check("error_ratio", "A", 3.0, "red", "")])
    assert fairness["panel_review_required"] and not fairness["retrain_required"]
    error = checks.decision([checks.mae_check(6.4)])
    assert error["retrain_required"] and error["incident_required"]
    assert checks.decision([checks.mae_check(5.0)])["overall_status"] == "green"


def test_offer_plan_keeps_upper_bound_within_capacity(small_forecaster, modelling_data):
    test = modelling_data.test.head(200)
    capacity = test["ENRLT"].astype(float)
    plans = plan_offers(small_forecaster, test, capacity)
    feasible = plans[plans["feasible"]]
    assert (feasible["upper_intake"] <= feasible["capacity"] + 1e-9).all()
    assert (feasible["planned_offers"] <= test.loc[feasible.index, "APPLCN"]).all()


def test_planner_overfills_less_often_than_carrying_yield_forward(small_forecaster, modelling_data):
    test = modelling_data.test
    capacity = test["ENRLT"].astype(float)
    evaluation = plan_evaluation(small_forecaster, test, plan_offers(small_forecaster, test, capacity), capacity)
    overall = evaluation.loc["All institutions"]
    assert overall["planner_overrun_share"] < overall["naive_overrun_share"]
    assert overall["planner_overrun_share"] <= OVERRUN_RISK + 0.05


def _summary() -> dict:
    red = checks.Check("error_ratio", "Private for-profit", 4.75, "red", "Ratio above 1.25 for two cycles is red")
    green = checks.mae_check(5.33)
    return {
        "generated_at": "2026-10-25T00:00:00+00:00", "model_version": "0.3.0", "approved_for_release": False,
        "reference_cycle": 2023, "current_cycle": 2024, "institutions": {"current": 1852},
        "performance": {"model": {"yield_mae_points": 5.33}, "baseline": {"yield_mae_points": 5.99}},
        "psi": {"YIELD_LAG1": 0.01}, "alerts": {"summary": {"disparate_impact_ratio": 0.29}},
        "evidently": {"data_drift": {"drifted_columns": []}},
        "offer_planner": {"by_group": [{"group": "All institutions", "planner_overrun_share": 0.07,
                                        "naive_overrun_share": 0.33, "planner_median_fill": 0.83,
                                        "naive_median_fill": 0.98}]},
        "checks": [red.__dict__, green.__dict__],
        "decision": checks.decision([red, green]),
    }


def test_site_and_incident_report_the_red_signal():
    summary = json.loads(json.dumps(_summary()))
    page = render(summary)
    assert "Ethics Review Panel" in page and "Private for-profit" in page
    body = incident_body(summary)
    assert "| Median error ratio vs other institutions | Private for-profit | 4.75 |" in body
