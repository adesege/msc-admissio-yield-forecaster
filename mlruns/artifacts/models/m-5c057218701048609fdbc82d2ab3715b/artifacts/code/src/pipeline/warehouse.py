from pathlib import Path

import duckdb
import pandas as pd

MODEL_READY_VIEW = """
CREATE OR REPLACE VIEW analytics.model_ready_panel AS
SELECT *
FROM analytics.yield_panel
WHERE FULL_YIELD_HISTORY
"""


def write_tables(warehouse_path: Path, tables: dict[str, pd.DataFrame]) -> None:
    warehouse_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(warehouse_path)) as connection:
        for qualified_name, frame in tables.items():
            schema, _ = qualified_name.split(".")
            connection.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            connection.register("incoming", frame)
            connection.execute(f"CREATE OR REPLACE TABLE {qualified_name} AS SELECT * FROM incoming")
            connection.unregister("incoming")
        connection.execute(MODEL_READY_VIEW)


def table_row_counts(warehouse_path: Path) -> dict[str, int]:
    with duckdb.connect(str(warehouse_path), read_only=True) as connection:
        names = connection.execute(
            "SELECT table_schema || '.' || table_name FROM information_schema.tables ORDER BY 1"
        ).fetchall()
        return {name: connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] for (name,) in names}
