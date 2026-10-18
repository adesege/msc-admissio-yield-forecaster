import numpy as np

from src.explain.counterfactuals import counterfactual_offers
from src.explain.local import LocalExplainer
from src.explain.sensitivity import missing_input_sensitivity, perturbation_sensitivity
from src.explain.shap_values import shap_explanation, top_drivers


def test_shap_values_add_up_to_forecast_change(small_forecaster, modelling_data):
    test = modelling_data.test.head(25)
    explanation = shap_explanation(small_forecaster, test, modelling_data.train)
    reconstructed = explanation.base_values + explanation.values.sum(axis=1)
    np.testing.assert_allclose(reconstructed, small_forecaster.predict_change(test), atol=1e-6)


def test_top_drivers_are_ordered_by_size(small_forecaster, modelling_data):
    explanation = shap_explanation(small_forecaster, modelling_data.test.head(5), modelling_data.train)
    drivers = top_drivers(explanation, 0)
    sizes = [abs(d["contribution_points"]) for d in drivers]
    assert len(drivers) == 3 and sizes == sorted(sizes, reverse=True)


def test_lime_explains_a_single_institution(small_forecaster, modelling_data):
    explainer = LocalExplainer(small_forecaster, modelling_data.train)
    result = explainer.explain(modelling_data.test.head(1), num_features=4)
    assert len(result["weights"]) == 4
    assert 0 <= result["forecast_yield"] <= 1


def test_counterfactuals_only_change_admit_rate(small_forecaster, modelling_data):
    row = modelling_data.test.iloc[[0]]
    last = float(row["YIELD_LAG1"].iloc[0])
    options = counterfactual_offers(small_forecaster, modelling_data.train, row, (last, min(1.0, last + 0.05)), total=2)
    if not options.empty:
        assert options["ADMIT_RATE"].between(0.05, 1.0).all()
        assert (options["offers"] <= row["APPLCN"].iloc[0]).all()


def test_sensitivity_tables_cover_every_input(small_forecaster, modelling_data):
    test = modelling_data.test.head(200)
    perturbation = perturbation_sensitivity(small_forecaster, test)
    assert (perturbation["mean_abs_shift_points"] >= 0).all()
    assert len(missing_input_sensitivity(small_forecaster, test)) == 11
