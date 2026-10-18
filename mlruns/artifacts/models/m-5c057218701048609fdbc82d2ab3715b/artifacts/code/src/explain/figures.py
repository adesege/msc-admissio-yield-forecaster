from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL, INK, MUTED, GRID = "#a3a29c", "#0b0b0b", "#52514e", "#e4e3df"
GROUP_COLORS = {
    "All other institutions": NEUTRAL,
    "HBCU": BLUE,
    "Under 1,000 students": ORANGE,
    "Private for-profit": AQUA,
}
FULL_WIDTH = 6.5

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def model_comparison(rows: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.8))
    y = np.arange(len(rows))
    ax.barh(y + 0.2, rows["cv_mae"], height=0.36, color=BLUE, label="Cross-validated MAE (2023 cycle)")
    ax.barh(y - 0.2, rows["holdout_mae"], height=0.36, color=ORANGE, label="Hold-out MAE (2024 cycle)")
    for i, row in enumerate(rows.itertuples()):
        if not np.isnan(row.cv_mae):
            ax.text(row.cv_mae + 0.05, i + 0.2, f"{row.cv_mae:.2f}", va="center", fontsize=8, color=INK)
        ax.text(row.holdout_mae + 0.05, i - 0.2, f"{row.holdout_mae:.2f}", va="center", fontsize=8, color=INK)
    ax.axvline(4.8, color=INK, linestyle="--", linewidth=1)
    ax.text(4.8, len(rows) - 0.45, " release threshold 4.8", fontsize=8, color=INK, va="bottom")
    ax.set_yticks(y, rows["model"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute error in yield (percentage points)")
    ax.set_xlim(0, max(rows["holdout_mae"].max(), rows["cv_mae"].max()) + 1)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    return save(fig, path)


def predicted_vs_actual(frame: pd.DataFrame, path: Path) -> Path:
    fig, (left, right) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.0), gridspec_kw={"width_ratios": [1.1, 1]})
    for group, color in GROUP_COLORS.items():
        part = frame[frame["FAIRNESS_GROUP"] == group]
        size = 6 if group == "All other institutions" else 10
        left.scatter(100 * part["YIELD_PRED"], 100 * part["YIELD"], s=size, color=color, alpha=0.7,
                     linewidths=0, label=group)
    left.plot([0, 100], [0, 100], color=INK, linewidth=1)
    left.set_xlabel("Forecast yield (%)")
    left.set_ylabel("Actual 2024 yield (%)")
    left.set_xlim(0, 100)
    left.set_ylim(0, 100)
    left.set_title("Forecast against actual", loc="left")

    residual = 100 * (frame["YIELD"] - frame["YIELD_PRED"])
    order = list(GROUP_COLORS)
    data = [residual[frame["FAIRNESS_GROUP"] == g] for g in order]
    parts = right.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=False,
                          medianprops={"color": INK})
    for patch, group in zip(parts["boxes"], order):
        patch.set_facecolor(GROUP_COLORS[group])
        patch.set_edgecolor(MUTED)
    right.axvline(0, color=INK, linewidth=1)
    right.set_yticks(range(1, len(order) + 1), [f"{g}\n(n={len(d)})" for g, d in zip(order, data)])
    right.set_xlabel("Actual minus forecast (points)")
    right.set_title("Residuals by group", loc="left")
    right.grid(axis="y", visible=False)
    left.legend(loc="upper left", fontsize=7, markerscale=1.5)
    fig.tight_layout()
    return save(fig, path)


def fairness_ratios(table: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.6))
    groups = table["group"].tolist()
    y = np.arange(len(groups))
    series = [("baseline", "Naive baseline", NEUTRAL), ("model", "Selected model", BLUE)]
    for offset, (column, label, color) in zip((0.2, -0.2), series):
        ax.barh(y + offset, table[column], height=0.36, color=color, label=label)
        for i, value in enumerate(table[column]):
            ax.text(value + 0.05, i + offset, f"{value:.2f}", va="center", fontsize=8, color=INK)
    ax.axvline(1.25, color=INK, linestyle="--", linewidth=1)
    ax.text(1.25, -0.75, " limit 1.25", fontsize=8, color=INK)
    ax.set_yticks(y, groups)
    ax.invert_yaxis()
    ax.set_xlabel("Median absolute error, group divided by all other institutions")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    return save(fig, path)


def mitigation_tradeoff(scores: pd.DataFrame, selected: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.8))
    for name, row in scores.iterrows():
        color = ORANGE if name == selected else BLUE
        ax.scatter(row["cv_yield_mae_points"], row["cv_worst_error_ratio"], s=60, color=color, zorder=3,
                   edgecolors="white", linewidths=1.5)
        ax.annotate(name.replace("_", " "), (row["cv_yield_mae_points"], row["cv_worst_error_ratio"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK)
    ax.set_xlabel("Cross-validated yield MAE, 2023 cycle (points)")
    ax.set_ylabel("Worst group error ratio")
    ax.margins(x=0.25, y=0.2)
    return save(fig, path)


def lime_panels(explanations: list[tuple[str, dict]], path: Path) -> Path:
    fig, axes = plt.subplots(len(explanations), 1, figsize=(FULL_WIDTH, 2.0 * len(explanations)))
    for ax, (title, explanation) in zip(np.atleast_1d(axes), explanations):
        weights = pd.DataFrame(explanation["weights"]).iloc[::-1]
        colors = [BLUE if w >= 0 else ORANGE for w in weights["weight_points"]]
        ax.barh(weights["rule"], weights["weight_points"], color=colors, height=0.6)
        ax.axvline(0, color=INK, linewidth=0.8)
        ax.set_title(f"{title}: forecast yield {100 * explanation['forecast_yield']:.1f}%", loc="left")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=7.5)
    np.atleast_1d(axes)[-1].set_xlabel("Local contribution to forecast yield (points)")
    fig.tight_layout()
    return save(fig, path)


def sensitivity_chart(table: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.6))
    wide = table.pivot(index="feature", columns="relative_change", values="mean_abs_shift_points")
    wide = wide.loc[wide.max(axis=1).sort_values().index]
    y = np.arange(len(wide))
    ax.barh(y + 0.2, wide[0.05], height=0.36, color=BLUE, label="Input changed by 5%")
    ax.barh(y - 0.2, wide[0.10], height=0.36, color=ORANGE, label="Input changed by 10%")
    ax.set_yticks(y, wide.index)
    ax.set_xlabel("Mean absolute shift in forecast yield (points)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    return save(fig, path)


def interval_coverage_chart(table: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.4))
    groups = table["group"].tolist()
    colors = [GROUP_COLORS.get(g, INK) for g in groups]
    ax.barh(groups, 100 * table["coverage"], color=colors, height=0.6)
    for i, row in enumerate(table.itertuples()):
        ax.text(100 * row.coverage + 1, i, f"{100 * row.coverage:.0f}%  (median width {row.median_width_points:.1f} pts)",
                va="center", fontsize=8, color=INK)
    ax.axvline(80, color=INK, linestyle="--", linewidth=1)
    ax.set_xlim(0, 130)
    ax.set_xlabel("Share of 2024 actual yields inside the 80% interval")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    return save(fig, path)
