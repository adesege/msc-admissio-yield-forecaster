import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request

from api.schemas import Forecast, Health, InstitutionCycle, Interval
from src.explain.shap_values import shap_explanation, top_drivers
from src.models.features import APPLICANT_CHANGE_BOUNDS, FEATURES, fairness_group
from src.models.forecaster import YieldForecaster

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
ADVISORY_NOTICE = "Advisory forecast for admissions planning. Offer decisions stay with named institution staff."
SIZE_LABELS = {1: "Under 1,000"}
CONTROL_LABELS = {3: "Private for-profit"}


@dataclass(frozen=True)
class LoadedModel:
    forecaster: YieldForecaster
    background: pd.DataFrame
    metadata: dict


def model_dir() -> Path:
    return Path(os.environ.get("ADMISSIO_MODEL_DIR", DEFAULT_MODEL_DIR))


def load_model(directory: Path | None = None) -> LoadedModel:
    directory = directory or model_dir()
    return LoadedModel(
        forecaster=joblib.load(directory / "admissio_yield.joblib"),
        background=pd.read_parquet(directory / "shap_background.parquet"),
        metadata=json.loads((directory / "metadata.json").read_text()),
    )


def to_frame(cycle: InstitutionCycle) -> pd.DataFrame:
    applicant_change = (cycle.applicants - cycle.last_cycle_applicants) / cycle.last_cycle_applicants
    policy_changed = cycle.last_cycle_test_policy is not None and cycle.last_cycle_test_policy != cycle.test_policy
    row = {
        "YIELD_LAG1": cycle.last_cycle_yield,
        "ADMIT_RATE": cycle.offers / cycle.applicants,
        "ADMIT_RATE_LAG1": cycle.last_cycle_offers / cycle.last_cycle_applicants,
        "APPLCN_CHANGE": float(np.clip(applicant_change, *APPLICANT_CHANGE_BOUNDS)),
        "LOG_APPLCN": float(np.log1p(cycle.applicants)),
        "TEST_POLICY_CHANGED": float(policy_changed),
        "ADMCON7": cycle.test_policy,
        "CONTROL": cycle.control,
        "INSTSIZE": cycle.size_band,
        "LOCALE": cycle.locale,
        "OBEREG": cycle.region,
        "C21BASIC": cycle.carnegie_class,
        "CONTROL_GROUP": CONTROL_LABELS.get(cycle.control, "Other control"),
        "SIZE_GROUP": SIZE_LABELS.get(cycle.size_band, "1,000 and above"),
        "HBCU_GROUP": "HBCU" if cycle.hbcu else "Not HBCU",
    }
    frame = pd.DataFrame([row])
    frame[FEATURES] = frame[FEATURES].astype(float)
    return frame


def forecast(model: LoadedModel, cycle: InstitutionCycle) -> Forecast:
    frame = to_frame(cycle)
    group = fairness_group(frame)
    interval = model.forecaster.predict_interval(frame, group).iloc[0]
    explanation = shap_explanation(model.forecaster, frame, model.background)
    return Forecast(
        institution_id=cycle.institution_id,
        forecast_yield=round(float(interval["yield"]), 4),
        yield_interval_80=Interval(lower=round(float(interval["lower"]), 4), upper=round(float(interval["upper"]), 4)),
        expected_enrolled=int(round(cycle.offers * interval["yield"])),
        expected_enrolled_interval_80=Interval(
            lower=float(round(cycle.offers * interval["lower"])), upper=float(round(cycle.offers * interval["upper"]))
        ),
        yield_drop_alert=bool(model.forecaster.yield_drop_alert(frame)[0]),
        fairness_group=str(group.iloc[0]),
        top_drivers=top_drivers(explanation, 0),
        model_version=model.metadata["model_version"],
        approved_for_release=model.metadata["approved_for_release"],
        notice=ADVISORY_NOTICE,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield


app = FastAPI(
    title="Admissio Yield Forecaster",
    version="0.3.0",
    description="Institution-level enrolment yield forecasts with intervals and drivers.",
    lifespan=lifespan,
)


@app.get("/health", response_model=Health)
def health(request: Request) -> Health:
    metadata = request.app.state.model.metadata
    return Health(status="ok", model_version=metadata["model_version"],
                  approved_for_release=metadata["approved_for_release"])


@app.post("/predict", response_model=Forecast)
def predict(cycle: InstitutionCycle, request: Request) -> Forecast:
    return forecast(request.app.state.model, cycle)
