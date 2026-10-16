import pandas as pd

from src.cleaning.admissions import clean_admissions, clean_directory


def make_directory():
    return pd.DataFrame({
        "UNITID": [100001, 100002],
        "INSTNM": [" Alpha College ", "Beta University"],
        "STABBR": ["AL", "GA"],
        **{col: [1, -2] for col in ["OBEREG", "SECTOR", "CONTROL", "HBCU", "TRIBAL", "LOCALE", "INSTSIZE", "C21BASIC"]},
    })


def make_admissions():
    return pd.DataFrame({
        "UNITID": [100001, 100001, 100002, 999999],
        "YEAR": [2024, 2024, 2024, 2024],
        "APPLCN": [500.0, 520.0, 300.0, 80.0],
        "ADMSSN": [250.0, 260.0, None, 40.0],
        "ENRLT": [60.0, 65.0, None, 10.0],
        "ADMCON7": [5.0, 5.0, 1.0, 3.0],
    })


def test_reserved_ipeds_codes_become_missing():
    directory = clean_directory(make_directory())
    assert directory.loc[directory["UNITID"] == 100002, "CONTROL"].isna().all()
    assert directory["INSTNM"].iloc[0] == "Alpha College"


def test_duplicates_keep_latest_record_and_unmatched_rows_are_quarantined():
    cleaned, quarantined, report = clean_admissions(make_admissions(), clean_directory(make_directory()))
    assert report.duplicate_rows_removed == 1
    assert cleaned.loc[cleaned["UNITID"] == 100001, "APPLCN"].item() == 520.0
    assert quarantined["UNITID"].tolist() == [999999]
    assert report.output_rows == 2


def test_missing_counts_are_flagged_not_imputed():
    cleaned, _, report = clean_admissions(make_admissions(), clean_directory(make_directory()))
    incomplete = cleaned[cleaned["UNITID"] == 100002]
    assert incomplete["ADMSSN"].isna().all()
    assert not incomplete["FUNNEL_COMPLETE"].item()
    assert report.rows_missing_admits == 1
