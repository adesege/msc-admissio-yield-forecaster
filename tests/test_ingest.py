from src.features.panel import build_panel
from src.ingest.ipeds import load_admissions, load_directory
from src.ingest.schema import admissions_schema


def test_admissions_pass_schema():
    admissions_schema.validate(load_admissions())


def test_three_years_exceed_5000_records():
    assert len(load_admissions()) >= 5000


def test_every_admissions_row_joins_to_directory():
    admissions = load_admissions(years=[2023])
    assert admissions["UNITID"].isin(load_directory()["UNITID"]).all()


def test_yield_is_a_share():
    panel = build_panel(load_admissions(), load_directory())
    assert panel["YIELD"].dropna().between(0, 1).all()
