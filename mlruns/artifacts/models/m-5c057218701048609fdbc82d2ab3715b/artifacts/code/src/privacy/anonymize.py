import hashlib
import hmac

import pandas as pd

from src.privacy.suppression import MIN_PUBLISHED_CELL

DIRECT_IDENTIFIERS = ["full_name", "email", "phone_number", "nin", "date_of_birth", "home_address"]
QUASI_IDENTIFIERS = ["state_of_origin"]
AGGREGATE_COLUMNS = ["tenant_ref", "cycle", "programme_code", "applications", "offers", "acceptances", "enrolled"]


class PersonalDataLeak(ValueError):
    pass


def pseudonymise(value: str, key: bytes) -> str:
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def strip_identifiers(records: pd.DataFrame) -> pd.DataFrame:
    return records.drop(columns=[c for c in DIRECT_IDENTIFIERS + QUASI_IDENTIFIERS if c in records.columns])


def aggregate_partner_extract(records: pd.DataFrame, key: bytes, min_cell: int = MIN_PUBLISHED_CELL) -> pd.DataFrame:
    stripped = strip_identifiers(records)
    aggregates = (
        stripped.groupby(["tenant_id", "cycle", "programme_code"], as_index=False)
        .agg(
            applications=("offered", "size"),
            offers=("offered", "sum"),
            acceptances=("accepted", "sum"),
            enrolled=("enrolled", "sum"),
        )
    )
    aggregates["tenant_ref"] = aggregates["tenant_id"].map(lambda tenant: pseudonymise(tenant, key))
    count_columns = ["applications", "offers", "acceptances", "enrolled"]
    aggregates[count_columns] = aggregates[count_columns].astype("Int64")
    small = (aggregates[count_columns] > 0) & (aggregates[count_columns] < min_cell)
    aggregates[count_columns] = aggregates[count_columns].mask(small)
    return aggregates[AGGREGATE_COLUMNS]


def assert_no_personal_data(frame: pd.DataFrame) -> None:
    leaked = sorted(set(frame.columns) & set(DIRECT_IDENTIFIERS + QUASI_IDENTIFIERS + ["tenant_id"]))
    if leaked:
        raise PersonalDataLeak(f"Columns that identify people or tenants remain: {leaked}")
