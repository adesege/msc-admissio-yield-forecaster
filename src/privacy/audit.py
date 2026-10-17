import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.governance.policy import AccessDenied, GovernancePolicy

AUDIT_LOGGER_NAME = "admissio.audit"


class JsonLinesFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(record.audit_event, sort_keys=True, default=str)


def configure_audit_logger(log_dir: Path, now: datetime | None = None) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    day = (now or datetime.now(timezone.utc)).date().isoformat()
    logger = logging.getLogger(AUDIT_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    handler = logging.FileHandler(log_dir / f"audit-{day}.jsonl", encoding="utf-8")
    handler.setFormatter(JsonLinesFormatter())
    logger.addHandler(handler)
    return logger


def frame_fingerprint(frame: pd.DataFrame) -> str:
    row_hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


@dataclass
class AuditTrail:
    run_id: str
    actor: str
    role: str
    policy: GovernancePolicy
    logger: logging.Logger

    def access(
        self,
        dataset: str,
        action: str,
        purpose: str,
        frame: pd.DataFrame | None = None,
        fingerprint: str | None = None,
        detail: dict | None = None,
    ) -> None:
        try:
            self.policy.authorize(self.role, dataset, action)
        except AccessDenied:
            self._emit(dataset, action, purpose, outcome="denied")
            raise
        self._emit(
            dataset,
            action,
            purpose,
            outcome="allowed",
            rows=None if frame is None else len(frame),
            fingerprint=fingerprint or (None if frame is None else frame_fingerprint(frame)),
            detail=detail,
        )

    def _emit(self, dataset: str, action: str, purpose: str, outcome: str, **fields) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "actor": self.actor,
            "role": self.role,
            "dataset": dataset,
            "classification": self.policy.dataset(dataset).classification,
            "action": action,
            "purpose": purpose,
            "outcome": outcome,
            **{key: value for key, value in fields.items() if value is not None},
        }
        self.logger.info(action, extra={"audit_event": event})
