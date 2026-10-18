import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PipelinePaths:
    root: Path

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw" / "ipeds"

    @property
    def interim_dir(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def warehouse_path(self) -> Path:
        return self.processed_dir / "warehouse.duckdb"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def audit_log_dir(self) -> Path:
        return self.root / "logs" / "audit"

    @property
    def governance_policy(self) -> Path:
        return self.root / "src" / "governance" / "policy.yaml"

    def ensure_output_dirs(self) -> None:
        for directory in (self.interim_dir, self.processed_dir, self.reports_dir, self.audit_log_dir):
            directory.mkdir(parents=True, exist_ok=True)


def default_paths() -> PipelinePaths:
    return PipelinePaths(Path(os.environ.get("ADMISSIO_ROOT", DEFAULT_ROOT)))
