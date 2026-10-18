import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "YIELD"
LAG_COLUMN = "YIELD_LAG1"
TRAIN_CYCLE = 2023
TEST_CYCLE = 2024

NUMERIC_FEATURES = [
    "YIELD_LAG1",
    "ADMIT_RATE",
    "ADMIT_RATE_LAG1",
    "APPLCN_CHANGE",
    "LOG_APPLCN",
    "TEST_POLICY_CHANGED",
]
CATEGORICAL_FEATURES = ["ADMCON7", "CONTROL", "INSTSIZE", "LOCALE", "OBEREG", "C21BASIC"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ACTIONABLE_FEATURES = ["ADMIT_RATE"]

FEATURE_LABELS = {
    "YIELD_LAG1": "Last cycle's yield",
    "ADMIT_RATE": "Admit rate this cycle",
    "ADMIT_RATE_LAG1": "Last cycle's admit rate",
    "APPLCN_CHANGE": "Change in applicants",
    "LOG_APPLCN": "Applicant volume (log)",
    "TEST_POLICY_CHANGED": "Test policy changed",
    "ADMCON7": "Test-score policy",
    "CONTROL": "Institution control",
    "INSTSIZE": "Enrolment size band",
    "LOCALE": "Locale",
    "OBEREG": "Region",
    "C21BASIC": "Carnegie class",
}

LEAKAGE_PREFIXES = ("ENRL", "SATNUM", "SATPCT", "ACTNUM", "ACTPCT", "SATVR", "SATMT", "ACTCM", "ACTEN", "ACTMT")
LEAKAGE_COLUMNS = {TARGET, "ENRLT"}

APPLICANT_CHANGE_BOUNDS = (-0.9, 3.0)
MIN_CATEGORY_FREQUENCY = 20

FAIRNESS_GROUPS = {
    "HBCU": ("HBCU_GROUP", "HBCU"),
    "Under 1,000 students": ("SIZE_GROUP", "Under 1,000"),
    "Private for-profit": ("CONTROL_GROUP", "Private for-profit"),
}


class LeakageError(ValueError):
    pass


def assert_no_leakage(columns) -> None:
    leaking = sorted(c for c in columns if c in LEAKAGE_COLUMNS or c.startswith(LEAKAGE_PREFIXES))
    if leaking:
        raise LeakageError(f"Columns measured after enrolment cannot be features: {', '.join(leaking)}")


def add_model_features(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.sort_values(["UNITID", "YEAR"]).copy()
    frame["ADMIT_RATE"] = (frame["ADMSSN"] / frame["APPLCN"]).where(frame["APPLCN"] > 0)
    frame["ADMIT_RATE_LAG1"] = frame.groupby("UNITID")["ADMIT_RATE"].shift(1)
    frame["LOG_APPLCN"] = np.log1p(frame["APPLCN"])
    frame["APPLCN_CHANGE"] = frame["APPLCN_CHANGE"].clip(*APPLICANT_CHANGE_BOUNDS)
    frame["TEST_POLICY_CHANGED"] = frame["TEST_POLICY_CHANGED"].astype(float)
    frame[CATEGORICAL_FEATURES] = frame[CATEGORICAL_FEATURES].astype(float)
    return frame


def modelling_rows(frame: pd.DataFrame) -> pd.DataFrame:
    usable = frame[TARGET].notna() & frame[LAG_COLUMN].notna() & frame["ADMIT_RATE_LAG1"].notna()
    return frame[usable].reset_index(drop=True)


def split_by_cycle(rows: pd.DataFrame, train_cycle: int = TRAIN_CYCLE, test_cycle: int = TEST_CYCLE):
    return rows[rows["YEAR"] == train_cycle].reset_index(drop=True), rows[rows["YEAR"] == test_cycle].reset_index(drop=True)


def fairness_group(frame: pd.DataFrame) -> pd.Series:
    labels = np.select(
        [
            frame["CONTROL_GROUP"].eq("Private for-profit"),
            frame["SIZE_GROUP"].eq("Under 1,000"),
            frame["HBCU_GROUP"].eq("HBCU"),
        ],
        ["Private for-profit", "Under 1,000 students", "HBCU"],
        default="All other institutions",
    )
    return pd.Series(labels, index=frame.index, name="FAIRNESS_GROUP")


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    assert_no_leakage(FEATURES)
    features = frame[FEATURES].copy()
    return features.apply(pd.to_numeric, errors="coerce").astype(float)


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=MIN_CATEGORY_FREQUENCY,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False,
    )


def source_feature(encoded_name: str) -> str:
    for feature in sorted(FEATURES, key=len, reverse=True):
        if encoded_name == feature or encoded_name.startswith(f"{feature}_"):
            return feature
    return encoded_name
