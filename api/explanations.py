import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from api.schemas import Forecast, InstitutionCycle

DEFAULT_STORE = Path(__file__).resolve().parents[1] / "logs" / "explanations" / "explanations.sqlite"
GENESIS_HASH = "0" * 64
RETENTION_DAYS = 1095

SCHEMA = """
CREATE TABLE IF NOT EXISTS explanations (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL,
    model_version TEXT NOT NULL,
    institution_id TEXT,
    fairness_group TEXT NOT NULL,
    inputs TEXT NOT NULL,
    forecast TEXT NOT NULL,
    previous_hash TEXT NOT NULL,
    record_hash TEXT NOT NULL
)
"""


def store_path() -> Path:
    return Path(os.environ.get("ADMISSIO_EXPLANATION_STORE", DEFAULT_STORE))


def _digest(previous_hash: str, payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{previous_hash}{canonical}".encode()).hexdigest()


class ExplanationStore:
    def __init__(self, path: Path | None = None):
        self.path = path or store_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def record(self, cycle: InstitutionCycle, forecast: Forecast) -> str:
        request_id = str(uuid.uuid4())
        payload = {
            "request_id": request_id,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_version": forecast.model_version,
            "institution_id": cycle.institution_id,
            "fairness_group": forecast.fairness_group,
            "inputs": cycle.model_dump(mode="json"),
            "forecast": forecast.model_dump(mode="json", exclude={"request_id"}),
        }
        with self._connect() as connection:
            last = connection.execute("SELECT record_hash FROM explanations ORDER BY sequence DESC LIMIT 1").fetchone()
            previous_hash = last["record_hash"] if last else GENESIS_HASH
            connection.execute(
                "INSERT INTO explanations (request_id, created_at, model_version, institution_id, fairness_group, "
                "inputs, forecast, previous_hash, record_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    request_id, payload["created_at"], payload["model_version"], payload["institution_id"],
                    payload["fairness_group"], json.dumps(payload["inputs"]), json.dumps(payload["forecast"]),
                    previous_hash, _digest(previous_hash, payload),
                ),
            )
        return request_id

    @staticmethod
    def _payload(row: sqlite3.Row) -> dict:
        return {
            "request_id": row["request_id"],
            "created_at": row["created_at"],
            "model_version": row["model_version"],
            "institution_id": row["institution_id"],
            "fairness_group": row["fairness_group"],
            "inputs": json.loads(row["inputs"]),
            "forecast": json.loads(row["forecast"]),
        }

    def get(self, request_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM explanations WHERE request_id = ?", (request_id,)).fetchone()
        return None if row is None else {**self._payload(row), "record_hash": row["record_hash"]}

    def count(self) -> int:
        with self._connect() as connection:
            return connection.execute("SELECT COUNT(*) FROM explanations").fetchone()[0]

    def verify(self) -> bool:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM explanations ORDER BY sequence").fetchall()
        previous_hash = rows[0]["previous_hash"] if rows else GENESIS_HASH
        for row in rows:
            if row["previous_hash"] != previous_hash or row["record_hash"] != _digest(previous_hash, self._payload(row)):
                return False
            previous_hash = row["record_hash"]
        return True

    def purge(self, older_than_days: int = RETENTION_DAYS) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat(timespec="seconds")
        with self._connect() as connection:
            return connection.execute("DELETE FROM explanations WHERE created_at < ?", (cutoff,)).rowcount
