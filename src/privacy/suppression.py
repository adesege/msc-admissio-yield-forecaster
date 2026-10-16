import numpy as np
import pandas as pd

MIN_PUBLISHED_CELL = 10


def suppression_mask(parts: pd.DataFrame, threshold: int = MIN_PUBLISHED_CELL) -> pd.DataFrame:
    values = parts.to_numpy(dtype=float)
    primary = (values > 0) & (values < threshold)
    mask = primary.copy()

    # A single hidden cell can be recovered by subtracting the published cells from the row total,
    # so the smallest remaining positive cell in that row is hidden as well.
    single_hidden = primary.sum(axis=1) == 1
    candidates = np.where(~primary & (values > 0), values, np.inf)
    complementary_column = candidates.argmin(axis=1)
    has_candidate = np.isfinite(candidates.min(axis=1))
    rows = np.flatnonzero(single_hidden & has_candidate)
    mask[rows, complementary_column[rows]] = True
    return pd.DataFrame(mask, index=parts.index, columns=parts.columns)


def suppress_small_cells(
    frame: pd.DataFrame,
    split_columns: dict[str, list[str]],
    threshold: int = MIN_PUBLISHED_CELL,
) -> tuple[pd.DataFrame, int]:
    published = frame.copy()
    suppressed_cells = 0
    for total_column, parts in split_columns.items():
        mask = suppression_mask(frame[parts], threshold)
        small_total = (frame[total_column] > 0) & (frame[total_column] < threshold)
        mask.loc[small_total, :] = True
        published[parts] = frame[parts].astype("Int64").mask(mask)
        published.loc[small_total, total_column] = np.nan
        suppressed_cells += int(mask.to_numpy().sum())
    return published, suppressed_cells
