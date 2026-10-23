import sqlite3

from api.explanations import ExplanationStore
from api.schemas import Driver, Forecast, InstitutionCycle, Interval

CYCLE = InstitutionCycle(
    institution_id="1", applicants=1000, offers=600, last_cycle_applicants=900, last_cycle_offers=550,
    last_cycle_yield=0.3, test_policy=5, control=1, size_band=2,
)
FORECAST = Forecast(
    institution_id="1", forecast_yield=0.31, yield_interval_80=Interval(lower=0.27, upper=0.35), expected_enrolled=186,
    expected_enrolled_interval_80=Interval(lower=162, upper=210), yield_drop_alert=False,
    fairness_group="All other institutions",
    top_drivers=[Driver(feature="Last cycle's yield", value=0.3, contribution_points=1.2)],
    model_version="test", approved_for_release=False, notice="advisory",
)


def test_records_are_retrievable_and_chained(tmp_path):
    store = ExplanationStore(tmp_path / "store.sqlite")
    first = store.record(CYCLE, FORECAST)
    store.record(CYCLE, FORECAST)
    assert store.get(first)["forecast"]["forecast_yield"] == 0.31
    assert store.count() == 2
    assert store.verify()


def test_tampering_breaks_the_chain(tmp_path):
    path = tmp_path / "store.sqlite"
    store = ExplanationStore(path)
    request_id = store.record(CYCLE, FORECAST)
    store.record(CYCLE, FORECAST)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE explanations SET forecast = REPLACE(forecast, '0.31', '0.41') WHERE request_id = ?",
                           (request_id,))
    assert not store.verify()


def test_purge_keeps_recent_records(tmp_path):
    store = ExplanationStore(tmp_path / "store.sqlite")
    store.record(CYCLE, FORECAST)
    assert store.purge() == 0
    assert store.purge(older_than_days=-1) == 1
    assert store.count() == 0
