from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.governance.policy import GovernancePolicy


@dataclass(frozen=True)
class RetentionTarget:
    dataset: str
    directory: Path
    pattern: str


def files_beyond_latest(directory: Path, pattern: str, keep: int) -> list[Path]:
    versions = sorted(directory.glob(pattern), key=lambda path: path.name, reverse=True)
    return versions[keep:]


def files_older_than(directory: Path, pattern: str, max_age_days: int, now: datetime) -> list[Path]:
    cutoff = now - timedelta(days=max_age_days)
    return [
        path for path in directory.glob(pattern)
        if datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) < cutoff
    ]


def expired_files(policy: GovernancePolicy, target: RetentionTarget, now: datetime) -> list[Path]:
    rule = policy.dataset(target.dataset).retention
    match rule["rule"]:
        case "keep_indefinitely":
            return []
        case "delete_after_run":
            return sorted(target.directory.glob(target.pattern))
        case "keep_latest":
            return files_beyond_latest(target.directory, target.pattern, rule["versions"])
        case "max_age_days":
            return files_older_than(target.directory, target.pattern, rule["days"], now)
        case unknown:
            raise ValueError(f"Unknown retention rule '{unknown}' for dataset '{target.dataset}'")
