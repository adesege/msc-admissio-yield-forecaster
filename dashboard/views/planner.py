import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard.core import cycle_from_row, offer_plan
from dashboard.session import dashboard_data, institution_picker, model, release_banner

data = dashboard_data()

st.title("Offer planner")
st.caption(
    "Enter the places available and the planner suggests the largest number of offers that keeps the chance of "
    "admitting more students than there are places to about one in ten."
)
release_banner()

row = institution_picker(data)
cycle = cycle_from_row(row)
capacity = st.number_input(
    "Places available for new students", min_value=1, value=int(row["actual_enrolled"]), step=1,
    key=f"capacity_{row['unitid']}",
    help="Defaults to the number of students who actually enrolled in 2024.",
)

curve, plan = offer_plan(model(), cycle, int(capacity))
if plan is None:
    st.error(
        "Even the smallest offer volume carries more than a one-in-ten chance of exceeding the places available. "
        "Review capacity with the admissions director before any offers are made.", icon=":material/warning:"
    )
else:
    first, second, third = st.columns(3)
    first.metric("Suggested offers", f"{int(plan['offers']):,}",
                 f"{int(plan['offers']) - cycle.offers:+,} vs offers made", delta_color="off")
    second.metric("Expected new students", f"{plan['expected_enrolled']:,.0f}",
                  f"{plan['expected_enrolled'] / capacity:.0%} of places", delta_color="off", delta_arrow="off")
    third.metric("Likely range", f"{plan['enrolled_lower']:,.0f} to {plan['enrolled_upper']:,.0f}",
                 "80% range", delta_color="off", delta_arrow="off")
    if plan["offers"] >= cycle.applicants:
        st.info("Offering a place to every applicant still leaves room, so the plan is limited by demand, not risk.",
                icon=":material/info:")

st.plotly_chart(charts.offer_plan_curve(curve, int(capacity), None if plan is None else plan["offers"], cycle.offers),
                width="stretch", theme=None)
st.caption(
    "The shaded band is the 80% range for new students at each offer volume. The planner stops where the top of the "
    "band meets the places available. A named member of staff approves every offer plan."
)

st.subheader("How the planner would have performed in 2024")
backtest = pd.DataFrame(data.summary["offer_planner"])
overall = backtest[backtest["group"] == "All institutions"].iloc[0]
st.markdown(
    f"Replaying the 2024 cycle with capacity set to each institution's actual intake, sizing offers from last "
    f"year's yield would have overfilled {overall['naive_overrun_share']:.0%} of institutions. The planner overfilled "
    f"{overall['planner_overrun_share']:.0%}. The price is unfilled places: the typical institution would have filled "
    f"{overall['planner_median_fill']:.0%} of its places, against {overall['naive_median_fill']:.0%} for the simple rule."
)
st.plotly_chart(charts.planner_backtest(backtest[backtest["group"] != "All institutions"]), width="stretch", theme=None)
st.caption(
    "Wider ranges for small and for-profit institutions make the planner more cautious for them, so they are "
    "overfilled less often but leave more places empty. Whether that trade is acceptable is a decision for each "
    "institution and the Ethics Review Panel."
)
