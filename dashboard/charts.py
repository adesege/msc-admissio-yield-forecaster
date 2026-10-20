import pandas as pd
import plotly.graph_objects as go

from dashboard.core import BASELINE_COLOR, GROUP_COLORS, GROUP_ORDER, INK, MODEL_COLOR, MUTED

RAISE_COLOR = "#2a78d6"
LOWER_COLOR = "#e34948"
LIMIT_COLOR = INK
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"


def _style(figure: go.Figure, height: int, x_title: str = "", y_title: str = "") -> go.Figure:
    figure.update_layout(
        height=height,
        margin={"l": 12, "r": 24, "t": 36, "b": 8},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": FONT, "color": INK, "size": 13},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "title": None},
        hoverlabel={"bgcolor": "white", "font": {"family": FONT, "color": INK}},
        bargap=0.3,
        bargroupgap=0.08,
    )
    figure.update_xaxes(title=x_title, gridcolor=GRID, zeroline=True, zerolinecolor="#c3c2b7", color=MUTED,
                        automargin=True)
    figure.update_yaxes(title=y_title, gridcolor=GRID, color=MUTED, automargin=True)
    figure.update_traces(cliponaxis=False, selector={"type": "bar"})
    return figure


def _room_for_labels(figure: go.Figure, headroom: float = 1.18) -> go.Figure:
    largest = max(max(trace.x) for trace in figure.data if trace.type == "bar")
    lowest = min(0, min(min(trace.x) for trace in figure.data if trace.type == "bar"))
    figure.update_xaxes(range=[lowest * headroom, largest * headroom])
    return figure


def baseline_vs_model_error(baseline: pd.DataFrame, model: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for name, table, color in (("Carry last year forward", baseline, BASELINE_COLOR),
                               ("Forecasting model", model, MODEL_COLOR)):
        figure.add_bar(
            y=GROUP_ORDER, x=table.loc[GROUP_ORDER, "mean_error"], orientation="h", name=name, marker_color=color,
            text=[f"{v:.1f}" for v in table.loc[GROUP_ORDER, "mean_error"]], textposition="outside",
            hovertemplate="%{y}<br>" + name + ": %{x:.1f} points<extra></extra>",
        )
    figure.update_yaxes(autorange="reversed")
    return _room_for_labels(_style(figure, 340, x_title="Average error (points of yield)"))


def forecast_vs_actual(holdout: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_scatter(x=[0, 100], y=[0, 100], mode="lines", line={"color": INK, "width": 1}, name="Perfect forecast",
                       hoverinfo="skip")
    for group in GROUP_ORDER:
        part = holdout[holdout["fairness_group"] == group]
        figure.add_scatter(
            x=100 * part["forecast_yield"], y=100 * part["actual_yield"], mode="markers", name=group,
            marker={"color": GROUP_COLORS[group], "size": 8 if group != GROUP_ORDER[0] else 6, "opacity": 0.75,
                    "line": {"color": SURFACE, "width": 1}},
            customdata=part[["institution", "state"]],
            hovertemplate="%{customdata[0]} (%{customdata[1]})<br>Forecast %{x:.1f}%<br>Actual %{y:.1f}%<extra></extra>",
        )
    figure.update_xaxes(range=[0, 101])
    figure.update_yaxes(range=[0, 101])
    return _style(figure, 480, x_title="Forecast yield for 2024 (%)", y_title="Actual 2024 yield (%)")


def driver_contributions(table: pd.DataFrame, top: int = 8) -> go.Figure:
    shown = table.head(top).iloc[::-1]
    colors = [RAISE_COLOR if v > 0 else LOWER_COLOR for v in shown["points"]]
    figure = go.Figure(
        go.Bar(
            y=shown["feature"], x=shown["points"], orientation="h", marker_color=colors,
            text=[f"{v:+.1f}" for v in shown["points"]], textposition="outside", customdata=shown["value"],
            hovertemplate="%{y}: %{customdata}<br>Effect on forecast: %{x:+.2f} points<extra></extra>",
        )
    )
    return _room_for_labels(_style(figure, 60 + 38 * len(shown), x_title="Effect on the forecast (percentage points)"),
                            headroom=1.3)


def global_drivers(table: pd.DataFrame) -> go.Figure:
    shown = table.iloc[::-1]
    figure = go.Figure(
        go.Bar(
            y=shown["feature"], x=shown["points"], orientation="h", marker_color=MODEL_COLOR,
            text=[f"{v:.1f}" for v in shown["points"]], textposition="outside",
            hovertemplate="%{y}<br>Average effect: %{x:.2f} points<extra></extra>",
        )
    )
    return _room_for_labels(_style(figure, 460, x_title="Average size of effect on a forecast (percentage points)"))


def enrollment_scenarios(curve: pd.DataFrame, current_offers: int) -> go.Figure:
    figure = go.Figure()
    figure.add_scatter(x=curve["offers"], y=curve["enrolled_upper"], mode="lines", line={"width": 0},
                       hoverinfo="skip", showlegend=False)
    figure.add_scatter(
        x=curve["offers"], y=curve["enrolled_lower"], mode="lines", line={"width": 0}, fill="tonexty",
        fillcolor="rgba(42,120,214,0.18)", name="Likely range (80%)", hoverinfo="skip",
    )
    figure.add_scatter(
        x=curve["offers"], y=curve["expected_enrolled"], mode="lines", line={"color": MODEL_COLOR, "width": 2},
        name="Expected new students", customdata=curve[["enrolled_lower", "enrolled_upper"]],
        hovertemplate="%{x:,.0f} offers<br>Expected %{y:,.0f} students<br>Range %{customdata[0]:,.0f} to "
                      "%{customdata[1]:,.0f}<extra></extra>",
    )
    figure.add_vline(x=current_offers, line={"color": MUTED, "dash": "dot", "width": 1},
                     annotation_text="Offers entered", annotation_position="top left")
    figure.update_yaxes(rangemode="tozero")
    return _style(figure, 380, x_title="Offers made", y_title="New students enrolling")


def ratio_against_limit(ratios: pd.DataFrame, limit: float) -> go.Figure:
    groups = list(ratios.index)
    figure = go.Figure(
        go.Bar(
            y=groups, x=ratios["median_error_ratio"], orientation="h",
            marker_color=[GROUP_COLORS[g] for g in groups],
            text=[f"{v:.2f}" for v in ratios["median_error_ratio"]], textposition="outside",
            hovertemplate="%{y}<br>Typical error is %{x:.2f} times the error for other institutions<extra></extra>",
        )
    )
    figure.add_vline(x=limit, line={"color": LIMIT_COLOR, "dash": "dash", "width": 1.5},
                     annotation_text=f"Limit {limit:.2f}", annotation_position="top right")
    figure.add_vline(x=1, line={"color": MUTED, "width": 1}, annotation_text="Parity", annotation_position="top left")
    figure.update_yaxes(autorange="reversed")
    return _room_for_labels(_style(figure, 280, x_title="Typical error compared with all other institutions (ratio)"))


def coverage_against_target(coverage: pd.DataFrame, target: float = 0.8) -> go.Figure:
    groups = GROUP_ORDER
    figure = go.Figure(
        go.Bar(
            y=groups, x=100 * coverage.loc[groups, "coverage"], orientation="h",
            marker_color=[GROUP_COLORS[g] for g in groups],
            text=[f"{100 * v:.0f}%" for v in coverage.loc[groups, "coverage"]], textposition="outside",
            customdata=coverage.loc[groups, "median_width_points"],
            hovertemplate="%{y}<br>%{x:.1f}% of actual results fell inside the range<br>"
                          "Typical range width %{customdata:.0f} points<extra></extra>",
        )
    )
    figure.add_vline(x=100 * target, line={"color": LIMIT_COLOR, "dash": "dash", "width": 1.5},
                     annotation_text="Target 80%", annotation_position="top left")
    figure.update_xaxes(range=[0, 105])
    figure.update_yaxes(autorange="reversed")
    return _style(figure, 280, x_title="Share of actual 2024 results inside the forecast range (%)")


def alert_rates(by_group: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for column, name, color in (("true_positive_rate", "Real drops flagged", MODEL_COLOR),
                                ("false_positive_rate", "False alarms", LOWER_COLOR)):
        figure.add_bar(
            y=GROUP_ORDER, x=100 * by_group.loc[GROUP_ORDER, column], orientation="h", name=name, marker_color=color,
            text=[f"{100 * v:.0f}%" for v in by_group.loc[GROUP_ORDER, column]], textposition="outside",
            hovertemplate="%{y}<br>" + name + ": %{x:.1f}%<extra></extra>",
        )
    figure.update_xaxes(range=[0, 105])
    figure.update_yaxes(autorange="reversed")
    return _style(figure, 340, x_title="Percent of institutions")
