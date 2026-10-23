import numpy as np
import pandas as pd

from src.models.features import TARGET, LAG_COLUMN, fairness_group
from src.models.forecaster import YieldForecaster

OVERRUN_RISK = (1 - 0.80) / 2
GRID_POINTS = 200
MIN_OFFER_SHARE = 0.02
GOAL_FEASIBLE_SHARE = 0.90
FILL_SHORTFALL = 0.90


def offer_grid(applicants: float, points: int = GRID_POINTS) -> np.ndarray:
    low = max(1, int(np.floor(applicants * MIN_OFFER_SHARE)))
    return np.unique(np.linspace(low, int(applicants), points).round().astype(int))


def plan_offers(forecaster: YieldForecaster, frame: pd.DataFrame, capacity: pd.Series,
                points: int = GRID_POINTS) -> pd.DataFrame:
    groups = fairness_group(frame)
    grids = [offer_grid(a, points) for a in frame["APPLCN"].to_numpy()]
    lengths = np.array([len(g) for g in grids])
    position = np.repeat(np.arange(len(frame)), lengths)
    offers = np.concatenate(grids)

    scenarios = frame.iloc[position].reset_index(drop=True)
    scenarios["ADMIT_RATE"] = offers / scenarios["APPLCN"].to_numpy()
    interval = forecaster.predict_interval(scenarios, groups.iloc[position].reset_index(drop=True))

    curve = pd.DataFrame(
        {
            "row": position,
            "offers": offers,
            "forecast_yield": interval["yield"].to_numpy(),
            "upper_intake": offers * interval["upper"].to_numpy(),
            "expected_intake": offers * interval["yield"].to_numpy(),
            "lower_intake": offers * interval["lower"].to_numpy(),
        }
    )
    curve["capacity"] = capacity.to_numpy()[position]
    safe = curve[curve["upper_intake"] <= curve["capacity"]]
    chosen = safe.loc[safe.groupby("row")["offers"].idxmax()].set_index("row")

    plans = pd.DataFrame(index=pd.RangeIndex(len(frame)))
    plans = plans.join(chosen[["offers", "forecast_yield", "expected_intake", "lower_intake", "upper_intake"]])
    plans.insert(0, "capacity", capacity.to_numpy())
    plans.insert(0, "fairness_group", groups.to_numpy())
    plans["feasible"] = plans["offers"].notna()
    return plans.rename(columns={"offers": "planned_offers"})


def naive_offers(frame: pd.DataFrame, capacity: pd.Series) -> np.ndarray:
    offers = np.ceil(capacity.to_numpy() / frame[LAG_COLUMN].clip(lower=0.01).to_numpy())
    return np.minimum(offers, frame["APPLCN"].to_numpy())


def forecast_at(forecaster: YieldForecaster, frame: pd.DataFrame, offers) -> np.ndarray:
    scenario = frame.copy()
    scenario["ADMIT_RATE"] = np.asarray(offers, dtype=float) / scenario["APPLCN"].to_numpy()
    return forecaster.predict(scenario)


def realised(forecaster: YieldForecaster, frame: pd.DataFrame, offers, capacity: pd.Series) -> pd.DataFrame:
    """Intake under a plan if the institution keeps its actual forecast error and yield responds to offers as modelled."""
    offers = np.asarray(offers, dtype=float)
    shift = forecast_at(forecaster, frame, offers) - forecast_at(forecaster, frame, frame["ADMSSN"].to_numpy())
    achieved_yield = np.clip(frame[TARGET].to_numpy() + shift, 0.0, 1.0)
    fill = offers * achieved_yield / capacity.to_numpy()
    return pd.DataFrame({"fill": fill, "overrun": fill > 1.0, "shortfall": fill < FILL_SHORTFALL})


def plan_evaluation(forecaster: YieldForecaster, frame: pd.DataFrame, plans: pd.DataFrame,
                    capacity: pd.Series) -> pd.DataFrame:
    feasible = plans["feasible"].to_numpy()
    planned = plans["planned_offers"].fillna(frame["ADMSSN"]).to_numpy()
    planner = realised(forecaster, frame, planned, capacity)
    naive = realised(forecaster, frame, naive_offers(frame, capacity), capacity)
    groups = plans["fairness_group"]

    def summarise(mask: np.ndarray) -> dict:
        usable = mask & feasible
        return {
            "institutions": int(mask.sum()),
            "feasible_share": float(feasible[mask].mean()),
            "planner_overrun_share": float(planner["overrun"][usable].mean()),
            "naive_overrun_share": float(naive["overrun"][mask].mean()),
            "planner_median_fill": float(planner["fill"][usable].median()),
            "naive_median_fill": float(naive["fill"][mask].median()),
            "planner_shortfall_share": float(planner["shortfall"][usable].mean()),
            "naive_shortfall_share": float(naive["shortfall"][mask].mean()),
        }

    rows = {"All institutions": summarise(np.ones(len(frame), dtype=bool))}
    for group in ["All other institutions", "HBCU", "Under 1,000 students", "Private for-profit"]:
        mask = (groups == group).to_numpy()
        if mask.any():
            rows[group] = summarise(mask)
    return pd.DataFrame(rows).T.round(3)


def goal_two_met(evaluation: pd.DataFrame) -> bool:
    overall = evaluation.loc["All institutions"]
    return bool(overall["feasible_share"] >= GOAL_FEASIBLE_SHARE and overall["planner_overrun_share"] <= OVERRUN_RISK)
