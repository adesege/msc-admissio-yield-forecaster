import pandas as pd
from fairlearn.metrics import MetricFrame, count, selection_rate

REPRESENTATION_FLOOR = 0.90
MIN_GROUP_SIZE = 30
GROUP_COLUMNS = ["HBCU_GROUP", "SIZE_GROUP", "CONTROL_GROUP", "TRIBAL_GROUP"]


def representation_table(panel: pd.DataFrame, eligible: pd.Series, group_column: str) -> pd.DataFrame:
    """Compare each group's share of the source panel with its share of the modelling set."""
    labelled = panel[group_column].notna()
    groups = panel.loc[labelled, group_column]
    kept = eligible[labelled].astype(int)
    frame = MetricFrame(
        metrics={"source_rows": count, "retention_rate": selection_rate},
        y_true=kept,
        y_pred=kept,
        sensitive_features=groups,
    )
    table = frame.by_group.rename_axis("group").reset_index()
    table["source_rows"] = table["source_rows"].astype(int)
    table["modelling_rows"] = (table["source_rows"] * table["retention_rate"]).round().astype(int)
    table["source_share"] = table["source_rows"] / table["source_rows"].sum()
    table["modelling_share"] = table["modelling_rows"] / table["modelling_rows"].sum()
    table["representation_ratio"] = table["modelling_share"] / table["source_share"]
    table["status"] = "ok"
    table.loc[table["representation_ratio"] < REPRESENTATION_FLOOR, "status"] = "under-represented"
    table.loc[table["source_rows"] < MIN_GROUP_SIZE, "status"] = "unassessed: fewer than 30 rows"
    table.insert(0, "dimension", group_column)
    return table


def representation_report(panel: pd.DataFrame, eligible: pd.Series) -> pd.DataFrame:
    return pd.concat(
        [representation_table(panel, eligible, column) for column in GROUP_COLUMNS],
        ignore_index=True,
    )


def flagged_groups(report: pd.DataFrame) -> list[str]:
    return report.loc[report["status"] == "under-represented", "group"].tolist()
