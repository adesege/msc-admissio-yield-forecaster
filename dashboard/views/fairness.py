import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard.core import (
    GROUP_ORDER,
    MONITORED_GROUPS,
    alert_fairness,
    coverage_by_group,
    error_by_group,
    error_ratio_table,
    single_threshold_alerts,
)
from dashboard.session import dashboard_data, release_banner
from src.fairness.alerts import FOUR_FIFTHS_RATIO
from src.fairness.audit import MAX_ERROR_RATIO

PREDICTIONS = {"Forecasting model": "forecast_yield", "Carry last year forward": "baseline_yield"}
ALERT_RULES = ("Single threshold (deployed)", "Equalized-odds thresholds (tested, not deployed)")
GROUP_PHRASES = {"HBCU": "HBCUs", "Under 1,000 students": "institutions under 1,000 students",
                 "Private for-profit": "private for-profit institutions"}

data = dashboard_data()
holdout = data.holdout


def status(passed: bool) -> str:
    return "✔ Within limit" if passed else "✘ Breaches limit"


st.title("Ethical compliance")
st.caption(
    "Fairness checks from the Ethical AI Charter, computed live with Fairlearn on the 2024 test cycle. The protected "
    "groups are institution types, because the model forecasts for institutions rather than people."
)
release_banner()

errors, ranges, alerts, representation, mitigation = st.tabs(
    ["Forecast accuracy by group", "Forecast ranges", "Yield-drop alert", "Data representation", "Mitigation tested"]
)

with errors:
    choice = st.radio("Forecast", list(PREDICTIONS), horizontal=True)
    ratios = error_ratio_table(holdout, PREDICTIONS[choice])
    st.markdown(
        f"**Rule:** each monitored group's typical error must stay within {MAX_ERROR_RATIO} times the typical error "
        "for every other institution."
    )
    st.plotly_chart(charts.ratio_against_limit(ratios, MAX_ERROR_RATIO), width="stretch", theme=None)
    table = pd.DataFrame(
        {
            "Institutions": ratios["institutions"],
            "Typical error, group (points)": ratios["group_median_error_points"],
            "Typical error, everyone else (points)": ratios["rest_median_error_points"],
            "Ratio": ratios["median_error_ratio"],
            "Status": [status(p) for p in ratios["within_limit"]],
        }
    ).rename_axis("Group")
    st.dataframe(table, width="stretch")
    noise = data.summary["sampling_noise_points"]
    st.markdown(
        f"Small and for-profit institutions admit few students, so their yield moves a lot by chance alone. The typical "
        f"chance variation is {noise['Private for-profit']:.1f} points for for-profit institutions and "
        f"{noise['Under 1,000 students']:.1f} points for small institutions, against "
        f"{noise['All other institutions']:.1f} points for everyone else. No model can remove chance variation, so part "
        "of the gap is a limit of the data rather than of the model."
    )
    with st.expander("Average and typical error by group (Fairlearn MetricFrame)"):
        by_group = error_by_group(holdout, PREDICTIONS[choice]).rename_axis("Group").rename(
            columns={"institutions": "Institutions", "mean_error": "Average error (points)",
                     "median_error": "Typical error (points)"}
        )
        st.dataframe(by_group.round(2), width="stretch")
        st.caption("Each institution counts once here, in the first matching group: for-profit, small, or HBCU, in order.")

with ranges:
    coverage = coverage_by_group(holdout)
    st.markdown(
        "Every forecast comes with an 80% range, calibrated separately for each group. A fair range should contain the "
        "actual result about four times in five for every group, not only on average."
    )
    st.plotly_chart(charts.coverage_against_target(coverage), width="stretch", theme=None)
    st.dataframe(
        coverage.rename_axis("Group").rename(columns={"institutions": "Institutions", "coverage": "Share inside range",
                                 "median_width_points": "Typical range width (points)"}).style.format(
            {"Share inside range": "{:.1%}", "Typical range width (points)": "{:.1f}"}
        ),
        width="stretch",
    )
    st.caption(
        "Wide ranges for small and for-profit institutions are deliberate. They tell small and for-profit partners how uncertain "
        "their forecast is instead of giving a falsely precise number."
    )

with alerts:
    st.markdown(
        "The alert warns a partner when the forecast falls by several points. It is checked against whether yield "
        "actually fell by 3 points or more in 2024."
    )
    rule = st.radio("Alert rule", ALERT_RULES, horizontal=True)
    if rule == ALERT_RULES[0]:
        threshold = st.slider("Raise the alert when the forecast falls by at least (points)", 1.0, 6.0, 3.0, 0.5)
        decisions = single_threshold_alerts(holdout, threshold)
    else:
        decisions = holdout["alert_equalized_odds"].to_numpy()
        st.caption("Group-specific thresholds chosen by Fairlearn's ThresholdOptimizer on the 2023 cycle.")

    by_group, overall, summary = alert_fairness(holdout, decisions)
    first, second, third, fourth = st.columns(4)
    first.metric("Disparate impact ratio", f"{summary['disparate_impact_ratio']:.2f}",
                 "Meets four-fifths rule" if summary["meets_four_fifths_rule"] else "Fails four-fifths rule",
                 delta_color="normal" if summary["meets_four_fifths_rule"] else "inverse", delta_arrow="off",
                 help=f"Lowest group alert rate divided by the highest. The four-fifths rule asks for {FOUR_FIFTHS_RATIO} "
                      "or more.")
    second.metric("Demographic parity difference", f"{summary['demographic_parity_difference']:.2f}",
                  help="Largest gap between groups in how often the alert fires.")
    third.metric("Equalized odds difference", f"{summary['equalized_odds_difference']:.2f}",
                 help="Largest gap between groups in caught drops or false alarms.")
    fourth.metric("False alarms overall", f"{100 * overall['false_positive_rate']:.1f}%",
                  f"Balanced accuracy {overall['balanced_accuracy']:.2f}", delta_color="off", delta_arrow="off")
    st.plotly_chart(charts.alert_rates(by_group), width="stretch", theme=None)
    worst = by_group.loc[MONITORED_GROUPS, "false_positive_rate"].idxmax()
    reference = by_group.loc[GROUP_ORDER[0], "false_positive_rate"]
    if reference > 0:
        st.markdown(
            f"False alarms for {GROUP_PHRASES[worst]} run at "
            f"{by_group.loc[worst, 'false_positive_rate'] / reference:.1f} times the rate for all other institutions. "
            "A false alarm could push an admissions team to make more offers than it needs."
        )

with representation:
    report = pd.DataFrame(data.summary["representation"])
    report = report[report["dimension"].isin(["HBCU_GROUP", "SIZE_GROUP", "CONTROL_GROUP", "TRIBAL_GROUP"])]
    st.markdown(
        "Data checks can remove some institution types more than others. A representation ratio below 0.90 means a "
        "group makes up a noticeably smaller share of the modelling data than of the source data."
    )
    st.dataframe(
        report[["group", "source_rows", "modelling_rows", "representation_ratio", "status"]].rename(
            columns={"group": "Group", "source_rows": "Rows in source", "modelling_rows": "Rows used",
                     "representation_ratio": "Representation ratio", "status": "Status"}
        ),
        width="stretch", hide_index=True,
    )

with mitigation:
    scores = pd.DataFrame(data.summary["mitigation_scores"])
    st.markdown(
        "Five versions of the model were compared on the 2023 cycle. The rule, fixed before scoring, kept versions "
        "within 5% of the unmitigated model's error and chose the version with the smallest worst-group ratio."
    )
    st.dataframe(
        scores[["candidate", "cv_yield_mae_points", "cv_worst_error_ratio", "holdout_yield_mae_points",
                "holdout_ratio_HBCU", "holdout_ratio_Under 1,000 students", "holdout_ratio_Private for-profit"]].rename(
            columns={
                "candidate": "Version",
                "cv_yield_mae_points": "2023 average error",
                "cv_worst_error_ratio": "2023 worst ratio",
                "holdout_yield_mae_points": "2024 average error",
                "holdout_ratio_HBCU": "2024 ratio, HBCU",
                "holdout_ratio_Under 1,000 students": "2024 ratio, small",
                "holdout_ratio_Private for-profit": "2024 ratio, for-profit",
            }
        ),
        width="stretch", hide_index=True,
    )
    st.caption(
        "Group reweighting was selected. It closed the HBCU gap but did not bring small or for-profit institutions "
        "within the limit, which is why the model remains a candidate awaiting the Ethics Review Panel."
    )
