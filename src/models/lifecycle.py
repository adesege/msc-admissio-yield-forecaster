import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from mlflow import MlflowClient

from src.config import default_paths
from src.models.retrain import CHALLENGER_ALIAS
from src.models.tracking import CANDIDATE_ALIAS, CHAMPION_ALIAS, REGISTERED_MODEL_NAME, configure_tracking

MODEL_FILES = ("admissio_yield.joblib", "shap_background.parquet", "metadata.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _archive(model_dir: Path, version: str) -> Path:
    archive = model_dir / "archive" / version
    archive.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        shutil.copy2(model_dir / name, archive / name)
    return archive


def promote(approval_reference: str, override_reason: str | None = None, model_dir: Path | None = None) -> dict:
    model_dir = model_dir or default_paths().root / "models"
    challenger_dir = model_dir / "challenger"
    summary = json.loads((challenger_dir / "retraining_summary.json").read_text())
    passed = all(summary["acceptance"]["checks"].values())
    if not passed and not override_reason:
        failed = [name for name, ok in summary["acceptance"]["checks"].items() if not ok]
        raise SystemExit(f"Challenger failed {', '.join(failed)}. Promotion needs a recorded panel override reason.")

    current = json.loads((model_dir / "metadata.json").read_text())
    _archive(model_dir, current["model_version"])
    metadata = json.loads((challenger_dir / "metadata.json").read_text())
    metadata |= {"status": "champion", "approved_for_release": True, "approval_reference": approval_reference,
                 "approved_at": _now(), "panel_override_reason": override_reason}
    for name in MODEL_FILES[:2]:
        shutil.copy2(challenger_dir / name, model_dir / name)
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    if metadata.get("registered_model_version"):
        configure_tracking()
        client = MlflowClient()
        version = str(metadata["registered_model_version"])
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version)
        client.delete_registered_model_alias(REGISTERED_MODEL_NAME, CHALLENGER_ALIAS)
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "release_status",
                                     f"approved: {approval_reference}")
    return metadata


def retire(reason: str, approval_reference: str, model_dir: Path | None = None) -> dict:
    model_dir = model_dir or default_paths().root / "models"
    metadata = json.loads((model_dir / "metadata.json").read_text())
    _archive(model_dir, metadata["model_version"])
    metadata |= {"status": "retired", "approved_for_release": False, "retired_at": _now(),
                 "retirement_reason": reason, "retirement_approval": approval_reference}
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    if metadata.get("registered_model_version"):
        configure_tracking()
        client = MlflowClient()
        aliases = client.get_registered_model(REGISTERED_MODEL_NAME).aliases
        for alias in (CHAMPION_ALIAS, CANDIDATE_ALIAS, CHALLENGER_ALIAS):
            if alias in aliases:
                client.delete_registered_model_alias(REGISTERED_MODEL_NAME, alias)
        client.set_model_version_tag(REGISTERED_MODEL_NAME, str(metadata["registered_model_version"]),
                                     "release_status", f"retired: {approval_reference}")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote a challenger or retire the served model.")
    commands = parser.add_subparsers(dest="command", required=True)
    promote_parser = commands.add_parser("promote")
    promote_parser.add_argument("--approval-reference", required=True, help="Ethics Review Panel minute reference.")
    promote_parser.add_argument("--override-reason", help="Recorded reason when the panel accepts a failed gate.")
    retire_parser = commands.add_parser("retire")
    retire_parser.add_argument("--reason", required=True)
    retire_parser.add_argument("--approval-reference", required=True)
    arguments = parser.parse_args()
    if arguments.command == "promote":
        print(json.dumps(promote(arguments.approval_reference, arguments.override_reason), indent=2, default=str))
    else:
        print(json.dumps(retire(arguments.reason, arguments.approval_reference), indent=2, default=str))
