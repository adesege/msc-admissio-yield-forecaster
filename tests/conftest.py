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
