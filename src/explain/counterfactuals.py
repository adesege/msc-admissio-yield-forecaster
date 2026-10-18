import dice_ml
from raiutils.exceptions import UserConfigValidationException
import numpy as np
import pandas as pd

from src.explain.local import filled_features
from src.models.features import ACTIONABLE_FEATURES, FEATURES, NUMERIC_FEATURES, prepare_features
from src.models.forecaster import YieldForecaster

ADMIT_RATE_RANGE = [0.05, 1.0]


class _ForecasterAdapter:
    """DiCE passes categorical columns as strings; the forecaster expects numeric codes."""

    def __init__(self, forecaster: YieldForecaster):
        self.forecaster = forecaster

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.forecaster.predict(frame[FEATURES].apply(pd.to_numeric, errors="coerce"))


def _dice_frame(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    for column in FEATURES:
        if column not in NUMERIC_FEATURES:
            frame[column] = frame[column].map(lambda v: f"{v:.1f}")
    return frame


def counterfactual_offers(forecaster: YieldForecaster, train_frame: pd.DataFrame, row: pd.DataFrame,
                          target_yield: tuple[float, float], total: int = 3, random_seed: int = 42) -> pd.DataFrame:
    reference = prepare_features(train_frame)
    training = _dice_frame(filled_features(train_frame, reference)).assign(YIELD=train_frame["YIELD"].to_numpy())
    data = dice_ml.Data(dataframe=training, continuous_features=NUMERIC_FEATURES, outcome_name="YIELD")
    model = dice_ml.Model(model=_ForecasterAdapter(forecaster), backend="sklearn", model_type="regressor")
    explainer = dice_ml.Dice(data, model, method="random")
    query = _dice_frame(filled_features(row, reference))
    try:
        result = explainer.generate_counterfactuals(
            query,
            total_CFs=total,
            desired_range=list(target_yield),
            features_to_vary=ACTIONABLE_FEATURES,
            permitted_range={"ADMIT_RATE": ADMIT_RATE_RANGE},
            random_seed=random_seed,
            verbose=False,
        )
    except UserConfigValidationException:
        return pd.DataFrame()
    options = result.cf_examples_list[0].final_cfs_df
    if options is None or options.empty:
        return pd.DataFrame()
    applicants = float(row["APPLCN"].iloc[0])
    options = options[ACTIONABLE_FEATURES].apply(pd.to_numeric)
    candidates = row[FEATURES].loc[row.index.repeat(len(options))].reset_index(drop=True)
    candidates[ACTIONABLE_FEATURES] = options.to_numpy()
    forecast = forecaster.predict(candidates)
    return options.reset_index(drop=True).assign(
        offers=np.rint(applicants * options["ADMIT_RATE"].to_numpy()).astype(int),
        forecast_yield=forecast,
        implied_enrolled=np.rint(applicants * options["ADMIT_RATE"].to_numpy() * forecast).astype(int),
    ).sort_values("ADMIT_RATE", ascending=False).drop_duplicates(subset=ACTIONABLE_FEATURES)
