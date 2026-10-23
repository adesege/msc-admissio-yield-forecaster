import json

import joblib
import pytest
from fastapi.testclient import TestClient

from src.explain.shap_values import background_sample

VALID_REQUEST = {
    "institution_id": "100654",
    "applicants": 9000,
    "offers": 6000,
    "last_cycle_applicants": 8500,
    "last_cycle_offers": 5900,
    "last_cycle_yield": 0.22,
    "test_policy": 5,
    "last_cycle_test_policy": 5,
    "control": 1,
    "size_band": 3,
    "locale": 12,
    "region": 5,
    "carnegie_class": 18,
    "hbcu": True,
}


def _client(small_forecaster, modelling_data, tmp_path, monkeypatch, metadata: dict):
    joblib.dump(small_forecaster, tmp_path / "admissio_yield.joblib")
    background_sample(modelling_data.train).to_parquet(tmp_path / "shap_background.parquet", index=False)
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setenv("ADMISSIO_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("ADMISSIO_EXPLANATION_STORE", str(tmp_path / "explanations.sqlite"))
    monkeypatch.setenv("ADMISSIO_MONITORING_SUMMARY", str(tmp_path / "summary.json"))
    from api.main import app

    return TestClient(app)


@pytest.fixture()
def client(small_forecaster, modelling_data, tmp_path, monkeypatch):
    metadata = {"model_version": "test", "approved_for_release": False}
    with _client(small_forecaster, modelling_data, tmp_path, monkeypatch, metadata) as test_client:
        yield test_client


def test_root_redirects_to_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_health_reports_model_version(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "test", "approved_for_release": False}


def test_predict_returns_forecast_interval_and_drivers(client):
    body = client.post("/predict", json=VALID_REQUEST).json()
    assert 0 <= body["yield_interval_80"]["lower"] <= body["forecast_yield"] <= body["yield_interval_80"]["upper"] <= 1
    assert body["fairness_group"] == "HBCU"
    assert len(body["top_drivers"]) == 3
    assert body["expected_enrolled"] == round(VALID_REQUEST["offers"] * body["forecast_yield"])


def test_predict_rejects_offers_above_applicants(client):
    response = client.post("/predict", json={**VALID_REQUEST, "offers": 10000})
    assert response.status_code == 422


def test_predict_rejects_unknown_test_policy(client):
    assert client.post("/predict", json={**VALID_REQUEST, "test_policy": 2}).status_code == 422


def test_every_forecast_is_stored_with_its_explanation(client):
    body = client.post("/predict", json=VALID_REQUEST).json()
    stored = client.get(f"/explanations/{body['request_id']}").json()
    assert stored["inputs"]["applicants"] == VALID_REQUEST["applicants"]
    assert stored["forecast"]["top_drivers"] == body["top_drivers"]
    assert stored["fairness_group"] == "HBCU"


def test_unknown_explanation_returns_404(client):
    assert client.get("/explanations/not-a-request").status_code == 404


def test_metrics_count_forecasts_by_group(client):
    client.post("/predict", json=VALID_REQUEST)
    client.post("/predict", json={**VALID_REQUEST, "locale": None})
    text = client.get("/metrics/").text
    assert 'admissio_predictions_total{fairness_group="HBCU"}' in text
    assert 'admissio_missing_inputs_total{field="locale"}' in text


def test_metrics_expose_latest_monitoring_status(client, tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"checks": [
        {"signal": "error_ratio", "group": "Private for-profit", "value": 4.75, "status": "red"}]}))
    text = client.get("/metrics/").text
    assert 'admissio_monitoring_status{group="Private for-profit",signal="error_ratio"} 2.0' in text


def test_retired_model_stops_serving(small_forecaster, modelling_data, tmp_path, monkeypatch):
    metadata = {"model_version": "test", "approved_for_release": False, "status": "retired"}
    with _client(small_forecaster, modelling_data, tmp_path, monkeypatch, metadata) as retired:
        assert retired.get("/health").json()["status"] == "retired"
        assert retired.post("/predict", json=VALID_REQUEST).status_code == 410
