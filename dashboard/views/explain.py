import streamlit as st

from api.main import forecast
from dashboard import charts
from dashboard.core import contributions, cycle_from_row, driver_sentence, forecast_summary, global_importance
from dashboard.session import dashboard_data, institution_picker, model, percent, release_banner

data = dashboard_data()

st.title("Explain a forecast")
st.caption("Pick an institution to see its forecast, how uncertain it is, and what pushed it up or down.")
release_banner()

row = institution_picker(data)
result = forecast(model(), cycle_from_row(row))

first, second, third, fourth = st.columns(4)
first.metric("Forecast yield", percent(result.forecast_yield, 1),
             f"Range {percent(result.yield_interval_80.lower)} to {percent(result.yield_interval_80.upper)}",
             delta_color="off", delta_arrow="off")
second.metric("Expected new students", f"{result.expected_enrolled:,}",
              f"Range {result.expected_enrolled_interval_80.lower:,.0f} to "
              f"{result.expected_enrolled_interval_80.upper:,.0f}", delta_color="off", delta_arrow="off")
third.metric("Actual 2024 yield", percent(row["actual_yield"], 1), f"{row['actual_enrolled']:,} students enrolled",
             delta_color="off", delta_arrow="off")
fourth.metric("Yield-drop alert", "Raised" if result.yield_drop_alert else "Not raised",
              "Forecast fall of 3 points or more", delta_color="off", delta_arrow="off")

st.markdown(forecast_summary(result, row["last_cycle_yield"]))
inside = row["lower"] <= row["actual_yield"] <= row["upper"]
st.markdown(
    f"In 2024 the institution's actual yield was {percent(row['actual_yield'], 1)}, which fell "
    f"{'inside' if inside else 'outside'} the forecast range."
)
if result.fairness_group != "All other institutions":
    st.info(
        f"The institution is in a monitored group ({result.fairness_group}). See the Ethical compliance page for how "
        "accurate forecasts are for the group.", icon=":material/balance:",
    )

st.subheader("What drove the forecast")
st.markdown("\n".join(f"- {driver_sentence(driver)}" for driver in result.top_drivers))
st.plotly_chart(charts.driver_contributions(contributions(row)), width="stretch", theme=None)
st.caption(
    "Blue bars pushed the forecast up and red bars pushed it down, measured against the average change across all "
    "institutions. The patterns come from past data and show association, not cause."
)

with st.expander("Which inputs matter most across all institutions"):
    st.plotly_chart(charts.global_drivers(global_importance(data.holdout)), width="stretch", theme=None)
    st.caption(
        "Average size of each input's effect on the 2024 forecasts. Last year's yield and the movement in "
        "applications matter most; the type and location of an institution matter far less."
    )
