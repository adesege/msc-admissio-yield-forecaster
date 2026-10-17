import pandas as pd
import pytest

from src.cleaning.admissions import clean_admissions, clean_directory
from src.features.panel import build_panel, has_full_yield_history
from src.ingest.ipeds import load_admissions, load_directory


@pytest.fixture(scope="session")
def raw_admissions() -> pd.DataFrame:
    return load_admissions()


@pytest.fixture(scope="session")
def raw_directory() -> pd.DataFrame:
    return load_directory()


@pytest.fixture(scope="session")
def yield_panel(raw_admissions, raw_directory) -> pd.DataFrame:
    directory = clean_directory(raw_directory)
    admissions, _, _ = clean_admissions(raw_admissions, directory)
    panel = build_panel(admissions, directory)
    panel["FULL_YIELD_HISTORY"] = has_full_yield_history(panel)
    return panel


@pytest.fixture(scope="session")
def modelling_data(yield_panel):
    from src.models.dataset import prepare_modelling_data

    return prepare_modelling_data(yield_panel[yield_panel["FULL_YIELD_HISTORY"]])


@pytest.fixture(scope="session")
def small_forecaster(modelling_data):
    from lightgbm import LGBMRegressor
    from sklearn.pipeline import Pipeline

    from src.models.features import build_preprocessor
    from src.models.forecaster import YieldForecaster

    change_model = Pipeline(
        [
            ("prepare", build_preprocessor(scale_numeric=False)),
            ("model", LGBMRegressor(n_estimators=60, num_leaves=15, objective="l1", random_state=0, verbose=-1)),
        ]
    )
    train = modelling_data.train
    forecaster = YieldForecaster(change_model).fit(train, train["YIELD"])
    residuals = train["YIELD"] - forecaster.predict(train)
    return forecaster.calibrate_intervals(residuals.to_numpy(), modelling_data.train_groups)
