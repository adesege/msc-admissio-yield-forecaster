from dataclasses import asdict, dataclass

import pandas as pd

KEY_COLUMNS = ["UNITID", "YEAR"]
IPEDS_RESERVED_CODES = (-1, -2, -3)
CODED_DIRECTORY_COLUMNS = ["OBEREG", "SECTOR", "CONTROL", "HBCU", "TRIBAL", "LOCALE", "INSTSIZE", "C21BASIC"]


@dataclass(frozen=True)
class CleaningReport:
    input_rows: int
    duplicate_rows_removed: int
    rows_quarantined_without_directory_match: int
    rows_missing_admits: int
    rows_missing_enrolled: int
    rows_with_complete_funnel: int
    output_rows: int

    def as_dict(self) -> dict:
        return asdict(self)


def clean_directory(directory: pd.DataFrame) -> pd.DataFrame:
    cleaned = directory.drop_duplicates(subset="UNITID", keep="last").copy()
    for column in CODED_DIRECTORY_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned[column] = cleaned[column].where(~cleaned[column].isin(IPEDS_RESERVED_CODES)).astype("Int64")
    cleaned["INSTNM"] = cleaned["INSTNM"].str.strip()
    return cleaned


def clean_admissions(admissions: pd.DataFrame, directory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    deduplicated = admissions.drop_duplicates(subset=KEY_COLUMNS, keep="last")
    has_directory_record = deduplicated["UNITID"].isin(directory["UNITID"])
    quarantined = deduplicated[~has_directory_record].assign(QUARANTINE_REASON="no directory record")
    cleaned = deduplicated[has_directory_record].copy()
    cleaned["FUNNEL_COMPLETE"] = cleaned[["APPLCN", "ADMSSN", "ENRLT"]].notna().all(axis=1) & (cleaned["ADMSSN"] > 0)

    report = CleaningReport(
        input_rows=len(admissions),
        duplicate_rows_removed=len(admissions) - len(deduplicated),
        rows_quarantined_without_directory_match=len(quarantined),
        rows_missing_admits=int(cleaned["ADMSSN"].isna().sum()),
        rows_missing_enrolled=int(cleaned["ENRLT"].isna().sum()),
        rows_with_complete_funnel=int(cleaned["FUNNEL_COMPLETE"].sum()),
        output_rows=len(cleaned),
    )
    return cleaned.reset_index(drop=True), quarantined.reset_index(drop=True), report
