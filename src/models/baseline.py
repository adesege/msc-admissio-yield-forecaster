import pandas as pd

from src.fairness.audit import error_ratio_vs_rest
from src.features.panel import build_panel, has_full_yield_history
from src.ingest.ipeds import load_admissions, load_directory


def naive_yield_forecast(panel: pd.DataFrame, target_year: int) -> pd.DataFrame:
    rows = panel[(panel["YEAR"] == target_year) & panel["YIELD"].notna() & panel["YIELD_LAG1"].notna()].copy()
    rows["YIELD_PRED"] = rows["YIELD_LAG1"]
    rows["ENRLT_PRED"] = rows["ADMSSN"] * rows["YIELD_PRED"]
    rows["INTAKE_ERROR"] = (rows["ENRLT_PRED"] - rows["ENRLT"]) / rows["ENRLT"].where(rows["ENRLT"] > 0)
    return rows


def evaluate(forecast: pd.DataFrame) -> dict:
    abs_error = (forecast["YIELD"] - forecast["YIELD_PRED"]).abs()
    intake = forecast["INTAKE_ERROR"].dropna()
    groups = {
        "HBCU": forecast["HBCU_GROUP"],
        "Under 1,000": forecast["SIZE_GROUP"],
        "Private for-profit": forecast["CONTROL_GROUP"],
    }
    return {
        "institutions": len(forecast),
        "yield_mae_points": round(abs_error.mean() * 100, 1),
        "yield_median_abs_error_points": round(abs_error.median() * 100, 1),
        "share_yield_moved_over_5_points": round((abs_error > 0.05).mean(), 3),
        "intake_median_abs_error": round(intake.abs().median(), 3),
        "share_intake_over_by_10pct": round((intake > 0.10).mean(), 3),
        "share_intake_under_by_10pct": round((intake < -0.10).mean(), 3),
        "error_ratio_vs_rest": {
            name: round(float(error_ratio_vs_rest(forecast["YIELD"], forecast["YIELD_PRED"], col, name)), 2)
            for name, col in groups.items()
        },
    }


def main(target_year: int = 2024) -> dict:
    panel = build_panel(load_admissions(), load_directory())
    results = evaluate(naive_yield_forecast(panel[has_full_yield_history(panel)], target_year))
    for key, value in results.items():
        print(f"{key}: {value}")
    return results


if __name__ == "__main__":
    main()
