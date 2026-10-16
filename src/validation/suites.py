import great_expectations.expectations as gxe

from src.features.panel import CONTROL_LABELS, HBCU_LABELS, SIZE_LABELS
from src.privacy.anonymize import AGGREGATE_COLUMNS

UNITID_RANGE = {"min_value": 100000, "max_value": 999999}


def raw_admissions_expectations() -> list:
    return [
        gxe.ExpectTableRowCountToBeBetween(min_value=5000, max_value=7000),
        gxe.ExpectColumnValuesToNotBeNull(column="UNITID"),
        gxe.ExpectColumnValuesToBeBetween(column="UNITID", **UNITID_RANGE),
        gxe.ExpectColumnValuesToBeInSet(column="YEAR", value_set=[2022, 2023, 2024]),
        gxe.ExpectCompoundColumnsToBeUnique(column_list=["UNITID", "YEAR"]),
        gxe.ExpectColumnValuesToNotBeNull(column="APPLCN"),
        gxe.ExpectColumnValuesToNotBeNull(column="ADMSSN", mostly=0.95),
        gxe.ExpectColumnValuesToNotBeNull(column="ENRLT", mostly=0.95),
        *[gxe.ExpectColumnValuesToBeBetween(column=c, min_value=0) for c in ("APPLCN", "ADMSSN", "ENRLT")],
        gxe.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="APPLCN", column_B="ADMSSN", or_equal=True, ignore_row_if="either_value_is_missing"
        ),
        gxe.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="ADMSSN", column_B="ENRLT", or_equal=True, ignore_row_if="either_value_is_missing"
        ),
        gxe.ExpectColumnValuesToBeInSet(column="ADMCON7", value_set=[1, 3, 5]),
    ]


def directory_expectations() -> list:
    return [
        gxe.ExpectColumnValuesToBeUnique(column="UNITID"),
        gxe.ExpectColumnValuesToBeBetween(column="UNITID", **UNITID_RANGE),
        gxe.ExpectColumnValuesToNotBeNull(column="INSTNM"),
        gxe.ExpectColumnValuesToBeInSet(column="HBCU", value_set=[1, 2]),
        gxe.ExpectColumnValuesToBeInSet(column="CONTROL", value_set=[1, 2, 3, -3]),
        gxe.ExpectColumnValuesToBeInSet(column="INSTSIZE", value_set=[1, 2, 3, 4, 5, -1, -2]),
    ]


def yield_panel_expectations() -> list:
    return [
        gxe.ExpectTableRowCountToBeBetween(min_value=5000, max_value=7000),
        gxe.ExpectCompoundColumnsToBeUnique(column_list=["UNITID", "YEAR"]),
        gxe.ExpectColumnValuesToNotBeNull(column="INSTNM"),
        gxe.ExpectColumnValuesToBeBetween(column="YIELD", min_value=0, max_value=1),
        gxe.ExpectColumnValuesToNotBeNull(column="YIELD", mostly=0.95),
        gxe.ExpectColumnValuesToBeInSet(column="HBCU_GROUP", value_set=list(HBCU_LABELS.values())),
        gxe.ExpectColumnValuesToBeInSet(column="CONTROL_GROUP", value_set=list(CONTROL_LABELS.values())),
        gxe.ExpectColumnValuesToBeInSet(column="SIZE_GROUP", value_set=list(SIZE_LABELS.values())),
        gxe.ExpectColumnValuesToNotBeNull(column="SIZE_GROUP", mostly=0.98),
        gxe.ExpectColumnValuesToBeBetween(column="APPLCN_CHANGE", min_value=-1, max_value=10, mostly=0.99),
    ]


def partner_aggregate_expectations() -> list:
    return [
        gxe.ExpectTableColumnsToMatchSet(column_set=AGGREGATE_COLUMNS, exact_match=True),
        gxe.ExpectColumnValueLengthsToEqual(column="tenant_ref", value=16),
        gxe.ExpectColumnValuesToBeBetween(column="applications", min_value=10),
    ]


SUITES = {
    "raw_admissions": raw_admissions_expectations,
    "ipeds_directory": directory_expectations,
    "yield_panel": yield_panel_expectations,
    "partner_aggregates": partner_aggregate_expectations,
}
