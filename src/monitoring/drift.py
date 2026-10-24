import numpy as np
import pandas as pd

from src.models.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

PSI_BINS = 10
PSI_FLOOR = 1e-4


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(expected, PSI_FLOOR, None)
    actual = np.clip(actual, PSI_FLOOR, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def numeric_psi(reference: pd.Series, current: pd.Series, bins: int = PSI_BINS) -> float:
    reference, current = reference.dropna().to_numpy(float), current.dropna().to_numpy(float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return categorical_psi(pd.Series(reference), pd.Series(current))
    inner = edges[1:-1]
    expected = np.bincount(np.searchsorted(inner, reference, side="right"), minlength=len(inner) + 1) / len(reference)
    actual = np.bincount(np.searchsorted(inner, current, side="right"), minlength=len(inner) + 1) / len(current)
    return _psi(expected, actual)


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    categories = sorted(set(reference.dropna()) | set(current.dropna()))
    expected = reference.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    actual = current.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    return _psi(expected, actual)


def feature_psi(reference: pd.DataFrame, current: pd.DataFrame) -> pd.Series:
    values = {f: numeric_psi(reference[f], current[f]) for f in NUMERIC_FEATURES}
    values |= {f: categorical_psi(reference[f], current[f]) for f in CATEGORICAL_FEATURES}
    return pd.Series(values, name="psi").round(4)


def missing_input_share(frame: pd.DataFrame, columns: list[str]) -> float:
    return float(frame[columns].isna().any(axis=1).mean())
