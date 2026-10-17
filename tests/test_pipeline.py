import json
import shutil

import pytest
from prefect.testing.utilities import prefect_test_harness

from src.config import PipelinePaths, default_paths
from src.pipeline.flow import yield_panel_pipeline


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    source = default_paths()
    paths = PipelinePaths(tmp_path_factory.mktemp("run"))
    shutil.copytree(source.raw_dir, paths.raw_dir)
    paths.governance_policy.parent.mkdir(parents=True)
    shutil.copy(source.governance_policy, paths.governance_policy)
    with prefect_test_harness():
        summary = yield_panel_pipeline(paths)
    return paths, summary


def test_pipeline_builds_the_model_ready_view(pipeline_run):
    _, summary = pipeline_run
    assert summary["warehouse_tables"]["analytics.yield_panel"] == 5908
    assert summary["warehouse_tables"]["analytics.model_ready_panel"] == summary["modelling_rows"]


def test_pipeline_deletes_the_partner_extract(pipeline_run):
    paths, summary = pipeline_run
    assert summary["files_removed_by_retention"]["partner_extract"] == 1
    assert not list(paths.interim_dir.glob("partner_extract_*"))


def test_pipeline_writes_audit_log_and_lineage(pipeline_run):
    paths, _ = pipeline_run
    events = [json.loads(line) for log in paths.audit_log_dir.glob("*.jsonl") for line in log.read_text().splitlines()]
    assert {e["action"] for e in events} >= {"read", "write", "anonymise", "delete"}
    lineage = json.loads((paths.reports_dir / "lineage.json").read_text())
    assert "analytics.model_ready_panel" in lineage["datasets"]
