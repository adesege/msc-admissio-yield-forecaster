from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from src.config import default_paths
from src.models.features import add_model_features, fairness_group, modelling_rows, split_by_cycle

MODEL_READY_QUERY = "SELECT * FROM analytics.model_ready_panel"


@dataclass(frozen=True)
class ModellingData:
    train: pd.DataFrame
    test: pd.DataFrame

    @property
    def train_groups(self) -> pd.Series:
        return fairness_group(self.train)

    @property
    def test_groups(self) -> pd.Series:
        return fairness_group(self.test)


def load_model_ready_panel(warehouse_path: Path | None = None) -> pd.DataFrame:
    warehouse_path = warehouse_path or default_paths().warehouse_path
    if not warehouse_path.exists():
        raise FileNotFoundError(f"{warehouse_path} not found. Run `python -m src.pipeline.flow` first.")
    with duckdb.connect(str(warehouse_path), read_only=True) as connection:
        return connection.execute(MODEL_READY_QUERY).df()


def prepare_modelling_data(panel: pd.DataFrame) -> ModellingData:
    rows = modelling_rows(add_model_features(panel))
    train, test = split_by_cycle(rows)
    return ModellingData(train=train, test=test)


def load_modelling_data(warehouse_path: Path | None = None) -> ModellingData:
    return prepare_modelling_data(load_model_ready_panel(warehouse_path))
