import pytest

from src.ingest.ipeds import ADMISSIONS_COLUMNS
from src.validation.runner import ValidationFailed, build_context, validate_frame


@pytest.fixture(scope="module")
def gx_context(tmp_path_factory):
    return build_context(tmp_path_factory.mktemp("gx"))


def test_raw_admissions_pass_suite(gx_context, raw_admissions):
    outcome = validate_frame(gx_context, "raw_admissions", raw_admissions[ADMISSIONS_COLUMNS + ["YEAR"]])
    assert outcome.success and outcome.failed == []


def test_directory_passes_suite(gx_context, raw_directory):
    assert validate_frame(gx_context, "ipeds_directory", raw_directory).success


def test_yield_panel_passes_suite(gx_context, yield_panel):
    assert validate_frame(gx_context, "yield_panel", yield_panel).success


def test_enrolled_above_admits_fails_the_suite(gx_context, raw_admissions):
    corrupted = raw_admissions[ADMISSIONS_COLUMNS + ["YEAR"]].copy()
    corrupted.loc[corrupted.index[:50], "ENRLT"] = corrupted.loc[corrupted.index[:50], "ADMSSN"] + 1
    with pytest.raises(ValidationFailed, match="expect_column_pair_values_a_to_be_greater_than_b"):
        validate_frame(gx_context, "raw_admissions", corrupted)
