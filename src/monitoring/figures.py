from pathlib import Path

import numpy as np
import pandas as pd

from src.explain.figures import FULL_WIDTH, GROUP_COLORS, INK, MUTED, NEUTRAL, ORANGE, BLUE, plt, save
from src.models.features import FEATURE_LABELS

STATUS_COLORS = {"green": "#1baf7a", "amber": "#e0a100", "red": "#d03b3b"}


def psi_chart(psi: pd.Series, path: Path) -> Path:
    ordered = psi.sort_values()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.0))
    colors = [STATUS_COLORS["red"] if v > 0.2 else STATUS_COLORS["amber"] if v > 0.1 else BLUE for v in ordered]
    ax.barh([FEATURE_LABELS[f] for f in ordered.index], ordered.to_numpy(), color=colors, height=0.6)
    for limit, label in ((0.1, "amber 0.1"), (0.2, "red 0.2")):
        ax.axvline(limit, color=INK, linestyle="--", linewidth=0.8)
        ax.text(limit, len(ordered) - 0.4, f" {label}", fontsize=8, color=INK, va="bottom")
    ax.set_xlim(0, max(0.25, float(ordered.max()) * 1.15))
    ax.set_xlabel("Population stability index, 2023 reference to 2024 current")
    ax.grid(axis="y", visible=False)
    return save(fig, path)


def error_ratio_chart(ratios: pd.DataFrame, limit: float, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.6))
    y = np.arange(len(ratios))
    ax.barh(y + 0.2, ratios["previous"], height=0.36, color=NEUTRAL, label="Previous cycle (2023, cross-validated)")
    ax.barh(y - 0.2, ratios["current"], height=0.36, color=[GROUP_COLORS[g] for g in ratios.index],
            label="Current cycle (2024)")
    for i, row in enumerate(ratios.itertuples()):
        ax.text(row.current + 0.05, i - 0.2, f"{row.current:.2f}", va="center", fontsize=8, color=INK)
        ax.text(row.previous + 0.05, i + 0.2, f"{row.previous:.2f}", va="center", fontsize=8, color=MUTED)
    ax.axvline(limit, color=INK, linestyle="--", linewidth=1)
    ax.text(limit, -0.8, f" limit {limit}", fontsize=8, color=INK, va="center")
    ax.set_yticks(y, ratios.index)
    ax.invert_yaxis()
    ax.set_xlabel("Median forecast error relative to all other institutions")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=8)
    ax.grid(axis="y", visible=False)
    return save(fig, path)


def coverage_and_alert_chart(coverage: pd.DataFrame, false_alarm_multiple: pd.Series, path: Path) -> Path:
    fig, (left, right) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.6))
    groups = [g for g in GROUP_COLORS if g in coverage.index]
    y = np.arange(len(groups))
    left.axvspan(0.75, 0.85, color="#e8f4ee", zorder=0)
    left.barh(y, coverage.loc[groups, "coverage"], color=[GROUP_COLORS[g] for g in groups], height=0.6)
    left.set_xlim(0.5, 1.0)
    left.set_yticks(y, groups)
    left.invert_yaxis()
    left.set_xlabel("80% interval coverage")
    left.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    left.grid(axis="y", visible=False)

    right.barh(y, false_alarm_multiple.reindex(groups), color=[GROUP_COLORS[g] for g in groups], height=0.6)
    right.axvline(2.0, color=INK, linestyle="--", linewidth=1)
    right.text(2.0, len(groups) - 0.45, " red 2x", fontsize=8, color=INK)
    right.set_yticks(y, [""] * len(groups))
    right.invert_yaxis()
    right.set_xlabel("False-alarm rate vs overall")
    right.grid(axis="y", visible=False)
    fig.tight_layout()
    return save(fig, path)


def planner_chart(evaluation: pd.DataFrame, path: Path) -> Path:
    rows = evaluation.drop(index="All institutions")
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.6))
    y = np.arange(len(rows))
    ax.barh(y + 0.2, rows["naive_overrun_share"], height=0.36, color=NEUTRAL, label="Last year's yield carried forward")
    ax.barh(y - 0.2, rows["planner_overrun_share"], height=0.36, color=ORANGE, label="Offer planner")
    ax.set_xlim(0, max(0.45, float(rows["naive_overrun_share"].max()) + 0.06))
    for i, row in enumerate(rows.itertuples()):
        ax.text(row.naive_overrun_share + 0.01, i + 0.2, f"{row.naive_overrun_share:.0%}", va="center", fontsize=8)
        ax.text(max(row.planner_overrun_share, 0.105) + 0.008, i - 0.2, f"{row.planner_overrun_share:.0%}", va="center",
                fontsize=8)
    ax.axvline(0.10, color=INK, linestyle="--", linewidth=1)
    ax.text(0.10, -0.75, " goal 10%", fontsize=8, color=INK, va="center")
    ax.set_yticks(y, rows.index)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Share of institutions whose intake would have exceeded capacity, 2024 cycle")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=8)
    ax.grid(axis="y", visible=False)
    return save(fig, path)
