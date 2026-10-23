import streamlit as st

from dashboard import charts
from dashboard.core import error_by_group
from dashboard.session import dashboard_data, percent, release_banner

data = dashboard_data()
baseline = data.summary["holdout"]["baseline"]
model = data.summary["holdout"]["model"]
holdout = data.holdout

st.title("Admissio Yield Forecaster")
st.caption(
    "How many admitted students will actually enroll? The forecaster estimates enrollment yield for each institution "
    "after applications close and before deposits arrive, so admissions teams can plan offer volumes."
)
release_banner()

st.subheader("Results on the 2024 admission cycle")
st.caption(f"Tested on {len(holdout):,} US institutions the model never saw during training.")
first, second, third, fourth = st.columns(4)
first.metric(
    "Average forecast error",
    f"{model['yield_mae_points']:.1f} points",
    f"{model['yield_mae_points'] - baseline['yield_mae_points']:+.1f} vs last-year method",
    delta_color="inverse",
    help="Average gap between forecast and actual yield, in percentage points.",
)
second.metric(
    "Intake missed by more than 10%",
    percent(model["share_intake_missed_by_over_10pct"], 1),
    f"{100 * (model['share_intake_missed_by_over_10pct'] - baseline['share_intake_missed_by_over_10pct']):+.1f} "
    "points vs last-year method",
    delta_color="inverse",
)
coverage = (holdout["actual_yield"].between(holdout["lower"], holdout["upper"])).mean()
third.metric("Results inside the forecast range", percent(coverage, 1), "Target 80%", delta_color="off", delta_arrow="off")
fourth.metric("Release threshold for average error", "4.8 points", "Not met", delta_color="off", delta_arrow="off")

left, right = st.columns([1.1, 1])
with left:
    st.markdown("**Average error by type of institution**")
    st.plotly_chart(
        charts.baseline_vs_model_error(error_by_group(holdout, "baseline_yield"), error_by_group(holdout, "forecast_yield")),
        width="stretch", theme=None,
    )
    st.caption(
        "The model is more accurate than carrying last year's yield forward for every group. Errors remain far larger "
        "for institutions under 1,000 students and for private for-profit institutions."
    )
with right:
    st.markdown("**Forecast against what actually happened**")
    st.plotly_chart(charts.forecast_vs_actual(holdout), width="stretch", theme=None)
    st.caption("Points on the diagonal line were forecast exactly. Hover over a point to see the institution.")

with st.expander("How the model works", expanded=True):
    st.markdown(
        """
The forecaster starts from what happened last year: the share of admitted students who enrolled. It learns,
from three years of public admissions records, how the share tends to move when the funnel changes. More
applications, a lower admit rate, or a change in test-score policy all shift the expected yield in patterns
recurring across institutions.

Each forecast comes with a range. For four forecasts in five, the actual yield landed inside the range in the 2024
test. Ranges are wider for small and for-profit institutions because their yield swings more from year to year.

The model never sees individual applicants. It works only with institution totals, and it advises people rather than
making offer decisions.
        """
    )
