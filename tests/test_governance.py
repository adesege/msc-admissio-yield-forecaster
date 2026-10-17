import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.config import default_paths
from src.governance.policy import AccessDenied, GovernancePolicy
from src.governance.retention import RetentionTarget, expired_files
from src.privacy.audit import AuditTrail, configure_audit_logger

NOW = datetime(2026, 10, 16, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def policy():
    return GovernancePolicy.from_yaml(default_paths().governance_policy)


def test_analyst_cannot_write_the_panel(policy):
    with pytest.raises(AccessDenied):
        policy.authorize("analyst", "yield_panel", "write")


def test_only_pipeline_may_read_partner_extract(policy):
    assert policy.is_allowed("pipeline_service", "partner_extract", "read")
    assert not policy.is_allowed("analyst", "partner_extract", "read")


def test_audit_trail_records_allowed_and_denied_access(policy, tmp_path):
    logger = configure_audit_logger(tmp_path, now=NOW)
    analyst = AuditTrail("run-1", "ada", "analyst", policy, logger)
    analyst.access("yield_panel", "read", "train model", frame=pd.DataFrame({"x": [1, 2]}))
    with pytest.raises(AccessDenied):
        analyst.access("partner_extract", "read", "browse applicants")
    events = [json.loads(line) for line in (tmp_path / "audit-2026-10-16.jsonl").read_text().splitlines()]
    assert [e["outcome"] for e in events] == ["allowed", "denied"]
    assert events[0]["rows"] == 2 and len(events[0]["fingerprint"]) == 64


def test_keep_latest_retains_two_versions(policy, tmp_path):
    for stamp in ("20260101T000000Z", "20260201T000000Z", "20260301T000000Z"):
        (tmp_path / f"yield_panel_{stamp}.parquet").touch()
    expired = expired_files(policy, RetentionTarget("yield_panel", tmp_path, "*.parquet"), NOW)
    assert [p.name for p in expired] == ["yield_panel_20260101T000000Z.parquet"]


def test_audit_logs_expire_after_three_years(policy, tmp_path):
    old, recent = tmp_path / "audit-2023-01-01.jsonl", tmp_path / "audit-2026-01-01.jsonl"
    for path, age in ((old, 1200), (recent, 30)):
        path.touch()
        stamp = (NOW - timedelta(days=age)).timestamp()
        os.utime(path, (stamp, stamp))
    expired = expired_files(policy, RetentionTarget("audit_log", tmp_path, "audit-*.jsonl"), NOW)
    assert expired == [old]


def test_raw_ipeds_is_never_expired(policy, tmp_path):
    (tmp_path / "adm2024.csv").touch()
    assert expired_files(policy, RetentionTarget("ipeds_raw", tmp_path, "*.csv"), NOW) == []
