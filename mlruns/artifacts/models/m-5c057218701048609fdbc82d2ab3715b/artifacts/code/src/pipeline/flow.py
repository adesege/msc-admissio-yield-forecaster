import json
import os
import secrets
from dataclasses import dataclass, replace
from datetime import datetime, timezone

import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE

from src.cleaning.admissions import clean_admissions, clean_directory
from src.config import PipelinePaths, default_paths
from src.fairness.representation import flagged_groups, representation_report
from src.features.panel import build_panel, has_full_yield_history
from src.governance.policy import GovernancePolicy
from src.governance.retention import RetentionTarget, expired_files
from src.ingest.ipeds import (
    ADMISSIONS_COLUMNS,
    GENDER_SPLIT_COLUMNS,
    LandedFile,
    land_raw_files,
    load_admissions,
    load_directory,
)
from src.pipeline.lineage import LineageGraph
from src.pipeline.warehouse import table_row_counts, write_tables
from src.privacy.anonymize import aggregate_partner_extract, assert_no_personal_data
from src.privacy.audit import AuditTrail, configure_audit_logger, frame_fingerprint
from src.privacy.partner_mock import generate_partner_extract
from src.privacy.suppression import suppress_small_cells
from src.validation.runner import ValidationOutcome, build_context, validate_frame

PIPELINE_ROLE = "pipeline_service"


@dataclass
class RunContext:
    run_id: str
    paths: PipelinePaths
    policy: GovernancePolicy
    audit: AuditTrail
    lineage: LineageGraph
    gx_context: object


def versioned_path(directory, dataset: str, run_id: str, suffix: str):
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{dataset}_{run_id}.{suffix}"


@task(name="land raw IPEDS files", retries=2, retry_delay_seconds=5, cache_policy=NO_CACHE)
def land_raw(ctx: RunContext) -> list[LandedFile]:
    landed = land_raw_files(ctx.paths.raw_dir)
    for file in landed:
        ctx.audit.access("ipeds_raw", "read", "land raw file and record checksum", fingerprint=file.sha256,
                         detail={"file": file.name, "bytes": file.size_bytes})
        ctx.lineage.dataset(f"file:{file.name}", "source", fingerprint=file.sha256)
    return landed


@task(name="load raw tables", cache_policy=NO_CACHE)
def load_raw(ctx: RunContext, landed: list[LandedFile]) -> tuple[pd.DataFrame, pd.DataFrame]:
    admissions = load_admissions(ctx.paths.raw_dir, include_gender_splits=True)
    directory = load_directory(ctx.paths.raw_dir)
    ctx.audit.access("ipeds_raw", "read", "load admissions for cleaning", frame=admissions)
    ctx.audit.access("ipeds_raw", "read", "load directory for integration", frame=directory)
    ctx.lineage.dataset("raw.admissions", "raw", len(admissions), frame_fingerprint(admissions))
    ctx.lineage.dataset("raw.directory", "raw", len(directory), frame_fingerprint(directory))
    ctx.lineage.step("ingest", [f"file:{f.name}" for f in landed], ["raw.admissions", "raw.directory"])
    return admissions, directory


@task(name="validate raw data", cache_policy=NO_CACHE)
def validate_raw(ctx: RunContext, admissions: pd.DataFrame, directory: pd.DataFrame) -> list[ValidationOutcome]:
    outcomes = [
        validate_frame(ctx.gx_context, "raw_admissions", admissions[ADMISSIONS_COLUMNS + ["YEAR"]]),
        validate_frame(ctx.gx_context, "ipeds_directory", directory),
    ]
    ctx.audit.access("ipeds_raw", "read", "validate raw data with Great Expectations",
                     detail={o.suite: f"{o.passed}/{o.evaluated} passed" for o in outcomes})
    return outcomes


@task(name="clean admissions and directory", cache_policy=NO_CACHE)
def clean(ctx: RunContext, admissions: pd.DataFrame, directory: pd.DataFrame):
    directory_clean = clean_directory(directory)
    admissions_clean, quarantined, report = clean_admissions(admissions, directory_clean)
    ctx.audit.access("quarantine", "write", "hold back rows without a directory match", frame=quarantined)
    ctx.lineage.dataset("staging.admissions", "staging", len(admissions_clean), frame_fingerprint(admissions_clean))
    ctx.lineage.dataset("staging.directory", "staging", len(directory_clean))
    ctx.lineage.dataset("quarantine.admissions", "quarantine", len(quarantined))
    ctx.lineage.step("clean", ["raw.admissions", "raw.directory"],
                     ["staging.admissions", "staging.directory", "quarantine.admissions"])
    return admissions_clean, directory_clean, quarantined, report


@task(name="transform and integrate panel", cache_policy=NO_CACHE)
def integrate(ctx: RunContext, admissions: pd.DataFrame, directory: pd.DataFrame) -> pd.DataFrame:
    panel = build_panel(admissions[ADMISSIONS_COLUMNS + ["YEAR", "FUNNEL_COMPLETE"]], directory)
    panel["FULL_YIELD_HISTORY"] = has_full_yield_history(panel)
    ctx.audit.access("yield_panel", "write", "derive yield features and join directory", frame=panel)
    ctx.lineage.dataset("analytics.yield_panel", "analytics", len(panel), frame_fingerprint(panel))
    ctx.lineage.step("transform_integrate", ["staging.admissions", "staging.directory"], ["analytics.yield_panel"])
    return panel


@task(name="validate yield panel", cache_policy=NO_CACHE)
def validate_panel(ctx: RunContext, panel: pd.DataFrame) -> ValidationOutcome:
    outcome = validate_frame(ctx.gx_context, "yield_panel", panel)
    ctx.audit.access("yield_panel", "read", "validate panel with Great Expectations",
                     detail={"passed": outcome.passed, "evaluated": outcome.evaluated})
    return outcome


@task(name="detect representation bias", cache_policy=NO_CACHE)
def detect_bias(ctx: RunContext, panel: pd.DataFrame) -> pd.DataFrame:
    report = representation_report(panel, panel["FULL_YIELD_HISTORY"])
    ctx.audit.access("representation_report", "write", "compare group shares before and after eligibility filter",
                     frame=report, detail={"flagged": flagged_groups(report)})
    ctx.lineage.dataset("reports.representation", "report", len(report))
    ctx.lineage.step("bias_detection", ["analytics.yield_panel"], ["reports.representation"])
    if flagged := flagged_groups(report):
        get_run_logger().warning("Under-represented groups in modelling set: %s", ", ".join(flagged))
    return report


@task(name="publish suppressed funnel", cache_policy=NO_CACHE)
def publish_funnel(ctx: RunContext, admissions: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    split_columns = [c for parts in GENDER_SPLIT_COLUMNS.values() for c in parts]
    funnel = admissions[["UNITID", "YEAR", *GENDER_SPLIT_COLUMNS, *split_columns]]
    published, suppressed = suppress_small_cells(funnel, GENDER_SPLIT_COLUMNS)
    ctx.audit.access("public_funnel", "write", "suppress gender cells below 10 before publication",
                     frame=published, detail={"cells_suppressed": suppressed})
    ctx.lineage.dataset("public.funnel_by_gender", "public", len(published))
    ctx.lineage.step("small_cell_suppression", ["staging.admissions"], ["public.funnel_by_gender"])
    return published, suppressed


@task(name="anonymise partner extract", cache_policy=NO_CACHE)
def anonymise_partner(ctx: RunContext) -> pd.DataFrame:
    extract = generate_partner_extract()
    extract_path = versioned_path(ctx.paths.interim_dir, "partner_extract", ctx.run_id, "csv")
    tenant_audit = replace(ctx.audit, actor="mock-tenant-export", role="partner_admin")
    tenant_audit.access("partner_extract", "write", "tenant exports applicant records for aggregation", frame=extract)
    extract.to_csv(extract_path, index=False)

    key_env = os.environ.get("ADMISSIO_PSEUDONYM_KEY")
    key = key_env.encode() if key_env else secrets.token_bytes(32)
    records = pd.read_csv(extract_path, dtype={"nin": str})
    ctx.audit.access("partner_extract", "read", "load applicant records for anonymisation", frame=records)
    aggregates = aggregate_partner_extract(records, key)
    assert_no_personal_data(aggregates)
    validate_frame(ctx.gx_context, "partner_aggregates", aggregates)
    ctx.audit.access("partner_extract", "anonymise", "strip identifiers, pseudonymise tenant, aggregate",
                     frame=aggregates, detail={"key_source": "environment" if key_env else "ephemeral"})
    ctx.lineage.dataset("interim.partner_extract", "confidential", len(records))
    ctx.lineage.dataset("analytics.partner_aggregates", "analytics", len(aggregates))
    ctx.lineage.step("anonymise", ["interim.partner_extract"], ["analytics.partner_aggregates"])
    return aggregates


@task(name="load warehouse", cache_policy=NO_CACHE)
def load_warehouse(ctx: RunContext, tables: dict[str, pd.DataFrame], dataset_names: dict[str, str]) -> dict[str, int]:
    write_tables(ctx.paths.warehouse_path, tables)
    for table, frame in tables.items():
        dataset = dataset_names[table]
        directory = ctx.paths.processed_dir / dataset
        frame.to_parquet(versioned_path(directory, dataset, ctx.run_id, "parquet"), index=False)
        ctx.audit.access(dataset, "write", f"load {table} into DuckDB and versioned Parquet", frame=frame)
    ctx.lineage.dataset("analytics.model_ready_panel", "view")
    ctx.lineage.step("create_view", ["analytics.yield_panel"], ["analytics.model_ready_panel"])
    return table_row_counts(ctx.paths.warehouse_path)


@task(name="apply retention policy", cache_policy=NO_CACHE)
def apply_retention(ctx: RunContext) -> dict[str, int]:
    paths = ctx.paths
    targets = [
        RetentionTarget("partner_extract", paths.interim_dir, "partner_extract_*.csv"),
        RetentionTarget("audit_log", paths.audit_log_dir, "audit-*.jsonl"),
        *[
            RetentionTarget(dataset, paths.processed_dir / dataset, "*.parquet")
            for dataset in ("yield_panel", "quarantine", "public_funnel", "partner_aggregates", "representation_report")
        ],
    ]
    removed = {}
    now = datetime.now(timezone.utc)
    for target in targets:
        expired = expired_files(ctx.policy, target, now)
        for path in expired:
            ctx.audit.access(target.dataset, "delete", "retention policy", detail={"file": path.name})
            path.unlink()
        removed[target.dataset] = len(expired)
    return removed


@flow(name="admissio-yield-panel", log_prints=True)
def yield_panel_pipeline(paths: PipelinePaths | None = None) -> dict:
    paths = paths or default_paths()
    paths.ensure_output_dirs()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    policy = GovernancePolicy.from_yaml(paths.governance_policy)
    audit = AuditTrail(run_id, os.environ.get("PIPELINE_ACTOR", "pipeline-service"), PIPELINE_ROLE, policy,
                       configure_audit_logger(paths.audit_log_dir))
    ctx = RunContext(run_id, paths, policy, audit, LineageGraph(run_id), build_context(paths.reports_dir))

    landed = land_raw(ctx)
    admissions, directory = load_raw(ctx, landed)
    raw_outcomes = validate_raw(ctx, admissions, directory)
    admissions_clean, directory_clean, quarantined, cleaning = clean(ctx, admissions, directory, wait_for=[raw_outcomes])
    panel = integrate(ctx, admissions_clean, directory_clean)
    panel_outcome = validate_panel(ctx, panel)
    representation = detect_bias(ctx, panel, wait_for=[panel_outcome])
    funnel, suppressed = publish_funnel(ctx, admissions_clean)
    partner_aggregates = anonymise_partner(ctx)
    row_counts = load_warehouse(
        ctx,
        {
            "analytics.yield_panel": panel,
            "quarantine.admissions": quarantined,
            "public.funnel_by_gender": funnel,
            "analytics.partner_aggregates": partner_aggregates,
            "reports.representation": representation,
        },
        {
            "analytics.yield_panel": "yield_panel",
            "quarantine.admissions": "quarantine",
            "public.funnel_by_gender": "public_funnel",
            "analytics.partner_aggregates": "partner_aggregates",
            "reports.representation": "representation_report",
        },
        wait_for=[panel_outcome],
    )
    removed = apply_retention(ctx, wait_for=[row_counts])
    lineage_path = ctx.lineage.write(paths.reports_dir / "lineage.json")

    summary = {
        "run_id": run_id,
        "cleaning": cleaning.as_dict(),
        "validation": {o.suite: f"{o.passed}/{o.evaluated}" for o in [*raw_outcomes, panel_outcome]},
        "modelling_rows": int(panel["FULL_YIELD_HISTORY"].sum()),
        "under_represented_groups": flagged_groups(representation),
        "gender_cells_suppressed": suppressed,
        "warehouse_tables": row_counts,
        "files_removed_by_retention": removed,
        "lineage": str(lineage_path.relative_to(paths.root)),
    }
    (paths.reports_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    yield_panel_pipeline()
