from src.features.panel import build_panel
from src.ingest.ipeds import ADMISSIONS_FILES, GENDER_SPLIT_COLUMNS, file_sha256, land_raw_files, load_admissions


def test_landing_records_a_checksum_for_every_expected_file():
    landed = land_raw_files()
    assert len(landed) == len(ADMISSIONS_FILES) + 1
    assert all(file.sha256 == file_sha256(file.path) for file in landed)


def test_landing_fails_when_a_raw_file_is_missing(tmp_path):
    try:
        land_raw_files(tmp_path)
    except FileNotFoundError as error:
        assert "adm2022_rv.csv" in str(error)
    else:
        raise AssertionError("expected missing raw files to stop the pipeline")


def test_three_years_exceed_5000_records(raw_admissions):
    assert len(raw_admissions) >= 5000


def test_gender_splits_load_as_numbers():
    admissions = load_admissions(years=[2024], include_gender_splits=True)
    split_columns = [c for parts in GENDER_SPLIT_COLUMNS.values() for c in parts]
    assert admissions[split_columns].dtypes.eq(float).all()


def test_every_2023_admissions_row_joins_to_directory(raw_admissions, raw_directory):
    admissions_2023 = raw_admissions[raw_admissions["YEAR"] == 2023]
    assert admissions_2023["UNITID"].isin(raw_directory["UNITID"]).all()


def test_yield_is_a_share(raw_admissions, raw_directory):
    panel = build_panel(raw_admissions, raw_directory)
    assert panel["YIELD"].dropna().between(0, 1).all()
