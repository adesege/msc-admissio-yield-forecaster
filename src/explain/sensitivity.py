import numpy as np
import pandas as pd
from sklearn.base import clone

from src.models.evaluation import regression_metrics
from src.models.features import FEATURE_LABELS, FEATURES, TARGET
from src.models.forecaster import YieldForecaster

PERTURBATION_STEPS = (-0.10, -0.05, 0.05, 0.10)
CONTINUOUS_INPUTS = ["YIELD_LAG1", "ADMIT_RATE", "ADMIT_RATE_LAG1", "APPLCN_CHANGE", "LOG_APPLCN"]
RATE_INPUTS = {"YIELD_LAG1", "ADMIT_RATE", "ADMIT_RATE_LAG1"}
TEST_POLICY_CODES = {1.0: "required", 3.0: "neither required nor recommended", 5.0: "considered but not required"}


def _shift_points(forecaster: YieldForecaster, base: np.ndarray, changed: pd.DataFrame) -> pd.Series:
    return pd.Series(100 * np.abs(forecaster.predict(changed) - base))


def perturbation_sensitivity(forecaster: YieldForecaster, frame: pd.DataFrame) -> pd.DataFrame:
    base = forecaster.predict(frame)
    rows = []
    for feature in CONTINUOUS_INPUTS:
        for step in PERTURBATION_STEPS:
            changed = frame.copy()
            changed[feature] = changed[feature] * (1 + step)
            if feature in RATE_INPUTS:
                changed[feature] = changed[feature].clip(0, 1)
            shift = _shift_points(forecaster, base, changed)
            rows.append(
                {
                    "feature": FEATURE_LABELS[feature],
                    "relative_change": step,
                    "mean_abs_shift_points": round(shift.mean(), 3),
                    "p95_abs_shift_points": round(shift.quantile(0.95), 3),
                }
            )
    return pd.DataFrame(rows)


def policy_sensitivity(forecaster: YieldForecaster, frame: pd.DataFrame) -> pd.DataFrame:
    base = forecaster.predict(frame)
    rows = []
    for code, label in TEST_POLICY_CODES.items():
        changed = frame.assign(ADMCON7=code)
        signed = 100 * (forecaster.predict(changed) - base)
        rows.append({"policy": label, "mean_shift_points": round(signed.mean(), 3),
                     "mean_abs_shift_points": round(np.abs(signed).mean(), 3)})
    return pd.DataFrame(rows)


def missing_input_sensitivity(forecaster: YieldForecaster, frame: pd.DataFrame) -> pd.DataFrame:
    base = forecaster.predict(frame)
    rows = []
    for feature in FEATURES:
        if feature == "YIELD_LAG1":
            continue
        shift = _shift_points(forecaster, base, frame.assign(**{feature: np.nan}))
        rows.append({"feature": FEATURE_LABELS[feature], "mean_abs_shift_points": round(shift.mean(), 3)})
    return pd.DataFrame(rows).sort_values("mean_abs_shift_points", ascending=False)


def _with_seed(change_model, seed: int):
    model = clone(change_model)
    for name in ("model__random_state", "model__estimator__random_state"):
        if name in model.get_params():
            model.set_params(**{name: seed})
    return model


def seed_stability(forecaster: YieldForecaster, train: pd.DataFrame, test: pd.DataFrame, fit_params: dict,
                   seeds=(0, 1, 2, 3, 4)) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        refit = YieldForecaster(_with_seed(forecaster.change_model, seed)).fit(train, train[TARGET], **fit_params)
        rows.append({"seed": seed, **regression_metrics(test[TARGET], refit.predict(test))})
    return pd.DataFrame(rows)
