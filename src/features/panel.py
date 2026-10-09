import pandas as pd

SIZE_LABELS = {1: "Under 1,000", 2: "1,000-4,999", 3: "5,000-9,999", 4: "10,000-19,999", 5: "20,000 and above"}
CONTROL_LABELS = {1: "Public", 2: "Private nonprofit", 3: "Private for-profit"}


def build_panel(admissions: pd.DataFrame, directory: pd.DataFrame) -> pd.DataFrame:
    panel = admissions.sort_values(["UNITID", "YEAR"]).copy()
    panel["YIELD"] = (panel["ENRLT"] / panel["ADMSSN"]).where(panel["ADMSSN"] > 0)
    grouped = panel.groupby("UNITID")
    panel["YIELD_LAG1"] = grouped["YIELD"].shift(1)
    panel["APPLCN_LAG1"] = grouped["APPLCN"].shift(1)
    panel["APPLCN_CHANGE"] = (panel["APPLCN"] - panel["APPLCN_LAG1"]) / panel["APPLCN_LAG1"]
    panel["TEST_POLICY_CHANGED"] = grouped["ADMCON7"].shift(1).ne(panel["ADMCON7"]) & grouped["ADMCON7"].shift(1).notna()
    panel = panel.merge(directory, on="UNITID", how="left")
    panel["SIZE_GROUP"] = panel["INSTSIZE"].map(SIZE_LABELS)
    panel["CONTROL_GROUP"] = panel["CONTROL"].map(CONTROL_LABELS)
    panel["HBCU_GROUP"] = panel["HBCU"].map({1: "HBCU", 2: "Not HBCU"})
    return panel
