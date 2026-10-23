import numpy as np
import streamlit as st

from api.main import forecast
from dashboard import charts
from dashboard.core import TEST_POLICY_NAMES, cycle_from_row, enrollment_curve, scenario
from dashboard.session import dashboard_data, institution_picker, model, percent, release_banner

CURVE_POINTS = 40

data = dashboard_data()

st.title("What-if analysis")
st.caption(
    "Change the inputs an admissions team can see before deposits arrive and watch how the forecast responds. "
    "The starting point is what the institution actually reported for 2024."
)
release_banner()

row = institution_picker(data)
original = cycle_from_row(row)

controls, results = st.columns([1, 1.4])
with controls:
    applicants = st.slider(
        "Applications received", min_value=max(1, int(original.applicants * 0.5)),
        max_value=int(original.applicants * 1.5), value=original.applicants, key=f"applicants_{original.institution_id}",
    )
    offers = st.slider(
        "Offers made", min_value=1, max_value=applicants, value=min(original.offers, applicants),
        key=f"offers_{original.institution_id}",
    )
    policy = st.radio(
        "Test-score policy", list(TEST_POLICY_NAMES), index=list(TEST_POLICY_NAMES).index(original.test_policy),
        format_func=TEST_POLICY_NAMES.get, key=f"policy_{original.institution_id}",
    )

as_reported = forecast(model(), original)
changed = forecast(model(), scenario(original, offers=offers, applicants=applicants, test_policy=policy))

with results:
    first, second = st.columns(2)
    first.metric("Forecast yield", percent(changed.forecast_yield, 1),
                 f"{100 * (changed.forecast_yield - as_reported.forecast_yield):+.1f} points vs as reported",
                 delta_color="off")
    second.metric("Expected new students", f"{changed.expected_enrolled:,}",
                  f"{changed.expected_enrolled - as_reported.expected_enrolled:+,} vs as reported", delta_color="off")
    st.markdown(
        f"Likely range: {percent(changed.yield_interval_80.lower)} to {percent(changed.yield_interval_80.upper)} "
        f"yield, or {changed.expected_enrolled_interval_80.lower:,.0f} to "
        f"{changed.expected_enrolled_interval_80.upper:,.0f} new students."
    )
    st.markdown(f"As reported, the institution made {original.offers:,} offers to {original.applicants:,} applicants "
                f"and {row['actual_enrolled']:,} students enrolled.")

offer_counts = np.unique(np.linspace(max(1, applicants * 0.1), applicants, CURVE_POINTS).astype(int))
curve = enrollment_curve(model(), scenario(original, applicants=applicants, test_policy=policy), offer_counts)
st.subheader("Expected enrollment across offer volumes")
st.plotly_chart(charts.enrollment_scenarios(curve, offers), width="stretch", theme=None)
st.caption(
    "Each point is a separate forecast with applications and test policy held at the values above. In past data, "
    "institutions admitting a larger share of applicants tended to see a smaller share of them enroll."
)
st.info(
    "The scenarios show how the model's forecasts move. They are not evidence of how real enrollment would respond "
    "to a change in offers. The Offer planner page sizes offers against the places available.", icon=":material/info:",
)
