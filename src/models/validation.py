import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

from src.models.features import LAG_COLUMN, TARGET, fairness_group, prepare_features
from src.models.forecaster import YieldForecaster
from src.models.mitigation import mitigation_fit_params
from src.models.tuning import cross_validator


def cross_validated_yield(forecaster: YieldForecaster, frame: pd.DataFrame, mitigation: str) -> np.ndarray:
    change = cross_val_predict(
        clone(forecaster.change_model),
        prepare_features(frame),
        (frame[TARGET] - frame[LAG_COLUMN]).to_numpy(),
        groups=frame["UNITID"],
        cv=cross_validator(),
        params=mitigation_fit_params(mitigation, fairness_group(frame)),
        n_jobs=-1,
    )
    return np.clip(frame[LAG_COLUMN].to_numpy() + change, 0.0, 1.0)
