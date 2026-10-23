import json
import os
from pathlib import Path

from prometheus_client import CollectorRegistry, Counter, Histogram, make_asgi_app
from prometheus_client.core import GaugeMetricFamily

from api.schemas import Forecast, InstitutionCycle

DEFAULT_MONITORING_SUMMARY = Path(__file__).resolve().parents[1] / "reports" / "monitoring" / "summary.json"
STATUS_CODES = {"green": 0, "amber": 1, "red": 2}
OPTIONAL_INPUTS = ("last_cycle_test_policy", "locale", "region", "carnegie_class")

registry = CollectorRegistry()

PREDICTIONS = Counter("admissio_predictions", "Forecasts served", ["fairness_group"], registry=registry)
ALERTS = Counter("admissio_yield_drop_alerts", "Yield-drop alerts raised", ["fairness_group"], registry=registry)
MISSING_INPUTS = Counter("admissio_missing_inputs", "Requests with an optional input left blank", ["field"],
                         registry=registry)
REQUESTS_WITH_MISSING = Counter("admissio_requests_with_missing_input", "Requests with any optional input blank",
                                registry=registry)
FORECAST_YIELD = Histogram("admissio_forecast_yield", "Forecast yield served", ["fairness_group"],
                           buckets=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0], registry=registry)
INTERVAL_WIDTH = Histogram("admissio_interval_width_points", "Width of the 80% interval in points", ["fairness_group"],
                           buckets=[5, 10, 20, 30, 40, 50, 60], registry=registry)
LATENCY = Histogram("admissio_predict_seconds", "Time to produce a forecast and its explanation", registry=registry)


def monitoring_summary_path() -> Path:
    return Path(os.environ.get("ADMISSIO_MONITORING_SUMMARY", DEFAULT_MONITORING_SUMMARY))


class MonitoringCollector:
    def collect(self):
        path = monitoring_summary_path()
        if not path.exists():
            return
        summary = json.loads(path.read_text())
        status = GaugeMetricFamily("admissio_monitoring_status", "0 green, 1 amber, 2 red", labels=["signal", "group"])
        value = GaugeMetricFamily("admissio_monitoring_value", "Latest value of each monitoring signal",
                                  labels=["signal", "group"])
        for check in summary.get("checks", []):
            labels = [check["signal"], check.get("group") or "all"]
            status.add_metric(labels, STATUS_CODES[check["status"]])
            if check.get("value") is not None:
                value.add_metric(labels, float(check["value"]))
        yield status
        yield value


registry.register(MonitoringCollector())
metrics_app = make_asgi_app(registry=registry)


def observe(cycle: InstitutionCycle, forecast: Forecast, seconds: float) -> None:
    group = forecast.fairness_group
    PREDICTIONS.labels(group).inc()
    if forecast.yield_drop_alert:
        ALERTS.labels(group).inc()
    missing = [field for field in OPTIONAL_INPUTS if getattr(cycle, field) is None]
    for field in missing:
        MISSING_INPUTS.labels(field).inc()
    if missing:
        REQUESTS_WITH_MISSING.inc()
    FORECAST_YIELD.labels(group).observe(forecast.forecast_yield)
    INTERVAL_WIDTH.labels(group).observe(100 * (forecast.yield_interval_80.upper - forecast.yield_interval_80.lower))
    LATENCY.observe(seconds)
