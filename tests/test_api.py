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


@pytest.fixture()
def client(small_forecaster, modelling_data, tmp_path, monkeypatch):
    joblib.dump(small_forecaster, tmp_path / "admissio_yield.joblib")
    background_sample(modelling_data.train).to_parquet(tmp_path / "shap_background.parquet", index=False)
    (tmp_path / "metadata.json").write_text(json.dumps({"model_version": "test", "approved_for_release": False}))
    monkeypatch.setenv("ADMISSIO_MODEL_DIR", str(tmp_path))
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


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
