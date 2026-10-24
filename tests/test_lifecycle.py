import json

import pytest

from src.models.lifecycle import promote, retire
from src.models.retrain import next_patch


def _model_dir(tmp_path, passed: bool):
    for directory in (tmp_path, tmp_path / "challenger"):
        directory.mkdir(exist_ok=True)
        (directory / "admissio_yield.joblib").write_bytes(directory.name.encode())
        (directory / "shap_background.parquet").write_bytes(b"background")
    (tmp_path / "metadata.json").write_text(json.dumps({"model_version": "0.3.0", "approved_for_release": False}))
    (tmp_path / "challenger" / "metadata.json").write_text(json.dumps({"model_version": "0.3.1", "status": "challenger"}))
    (tmp_path / "challenger" / "retraining_summary.json").write_text(json.dumps(
        {"acceptance": {"checks": {"yield_mae_at_or_below_4_8_points": passed}}}))
    return tmp_path


def test_next_patch_increments_semantic_version():
    assert next_patch("0.3.0") == "0.3.1"


def test_promotion_refuses_a_failed_challenger_without_panel_override(tmp_path):
    with pytest.raises(SystemExit, match="panel override"):
        promote("ERP-1", model_dir=_model_dir(tmp_path, passed=False))


def test_promotion_archives_the_previous_model(tmp_path):
    model_dir = _model_dir(tmp_path, passed=False)
    metadata = promote("ERP-1", override_reason="Noise-adjusted criterion approved", model_dir=model_dir)
    assert metadata["model_version"] == "0.3.1" and metadata["approved_for_release"]
    assert (model_dir / "archive" / "0.3.0" / "metadata.json").exists()
    assert (model_dir / "admissio_yield.joblib").read_bytes() == b"challenger"


def test_retirement_marks_the_model_retired_and_keeps_an_archive(tmp_path):
    model_dir = _model_dir(tmp_path, passed=True)
    metadata = retire("Replaced by partner-trained model", "ERP-2", model_dir=model_dir)
    assert metadata["status"] == "retired"
    assert json.loads((model_dir / "metadata.json").read_text())["retirement_approval"] == "ERP-2"
    assert (model_dir / "archive" / "0.3.0" / "admissio_yield.joblib").exists()
