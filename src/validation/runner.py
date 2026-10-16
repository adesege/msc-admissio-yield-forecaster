from dataclasses import dataclass
from pathlib import Path

import great_expectations as gx
import pandas as pd
from great_expectations.checkpoint import UpdateDataDocsAction

from src.validation.suites import SUITES

DATASOURCE_NAME = "admissio_pipeline"


class ValidationFailed(RuntimeError):
    pass


@dataclass(frozen=True)
class ValidationOutcome:
    suite: str
    success: bool
    evaluated: int
    failed: list[str]

    @property
    def passed(self) -> int:
        return self.evaluated - len(self.failed)


def build_context(project_root: Path):
    project_root.mkdir(parents=True, exist_ok=True)
    return gx.get_context(mode="file", project_root_dir=project_root)


def _checkpoint_for(context, suite_name: str):
    datasource = context.data_sources.add_or_update_pandas(DATASOURCE_NAME)
    asset_names = {asset.name for asset in datasource.assets}
    asset = (
        datasource.get_asset(suite_name)
        if suite_name in asset_names
        else datasource.add_dataframe_asset(name=suite_name)
    )
    batch_names = {batch.name for batch in asset.batch_definitions}
    batch_definition = (
        asset.get_batch_definition("whole_frame")
        if "whole_frame" in batch_names
        else asset.add_batch_definition_whole_dataframe("whole_frame")
    )
    suite = context.suites.add_or_update(gx.ExpectationSuite(name=suite_name, expectations=SUITES[suite_name]()))
    definition = context.validation_definitions.add_or_update(
        gx.ValidationDefinition(name=f"{suite_name}_validation", data=batch_definition, suite=suite)
    )
    return context.checkpoints.add_or_update(
        gx.Checkpoint(
            name=f"{suite_name}_checkpoint",
            validation_definitions=[definition],
            actions=[UpdateDataDocsAction(name="update_data_docs")],
            result_format={"result_format": "SUMMARY"},
        )
    )


def validate_frame(context, suite_name: str, frame: pd.DataFrame, raise_on_failure: bool = True) -> ValidationOutcome:
    result = _checkpoint_for(context, suite_name).run(batch_parameters={"dataframe": frame})
    expectation_results = [r for run in result.run_results.values() for r in run.results]
    failed = [
        f"{r.expectation_config.type}({r.expectation_config.kwargs.get('column', r.expectation_config.kwargs.get('column_list', ''))})"
        for r in expectation_results
        if not r.success
    ]
    outcome = ValidationOutcome(suite_name, result.success, len(expectation_results), failed)
    if raise_on_failure and not outcome.success:
        raise ValidationFailed(f"Suite '{suite_name}' failed: {', '.join(failed)}")
    return outcome


def data_docs_index(project_root: Path) -> Path:
    return project_root / "gx" / "uncommitted" / "data_docs" / "local_site" / "index.html"
