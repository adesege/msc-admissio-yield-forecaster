import streamlit as st

from api.main import forecast
from dashboard import charts
from dashboard.core import contributions, cycle_from_row, driver_sentence, forecast_summary, global_importance
from dashboard.session import (counterfactuals, dashboard_data, institution_picker, lime_explanation, model, percent,
                               release_banner)

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

shap_tab, lime_tab, counterfactual_tab = st.tabs(["What drove the forecast (SHAP)", "Second opinion (LIME)",
                                                  "What would change the outcome"])

with shap_tab:
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

with lime_tab:
    st.markdown(
        "LIME fits a simple model to thousands of small variations of this institution's inputs. When its leading "
        "inputs agree with the SHAP drivers, the explanation is more trustworthy."
    )
    if st.button("Run LIME for this institution", key=f"lime_{row['unitid']}"):
        lime = lime_explanation(row["unitid"])
        st.plotly_chart(charts.lime_weights(lime["weights"]), width="stretch", theme=None)
        fit = lime["local_model_r2"]
        st.caption(
            f"The local model explains {fit:.0%} of the variation in the forecaster's output around this institution. "
            + ("A low share means the explanation is a rough guide only." if fit < 0.5 else
               "The local approximation is reasonable.")
        )

with counterfactual_tab:
    last_yield = float(row["last_cycle_yield"])
    st.markdown(
        f"Which admit rates would keep this institution's yield at or above last cycle's {percent(last_yield, 1)}? "
        "DiCE varies only the admit rate, the one input the institution controls before deposits arrive."
    )
    if st.button("Find alternative offer volumes", key=f"cf_{row['unitid']}"):
        options = counterfactuals(row["unitid"], last_yield, min(1.0, last_yield + 0.05))
        if options.empty:
            st.info("No admit rate between 5% and 100% brings the forecast into that range.", icon=":material/info:")
        else:
            st.dataframe(
                options.assign(
                    admit_rate=options["ADMIT_RATE"].map(lambda v: f"{v:.0%}"),
                    forecast=options["forecast_yield"].map(lambda v: f"{v:.1%}"),
                    offer_count=options["offers"].map(lambda v: f"{v:,}"),
                    students=options["implied_enrolled"].map(lambda v: f"{v:,}"),
                )[["admit_rate", "offer_count", "forecast", "students"]].rename(columns={
                    "admit_rate": "Admit rate", "offer_count": "Offers", "forecast": "Forecast yield",
                    "students": "Implied new students"}),
                hide_index=True, width="stretch",
            )
            st.caption(
                f"As reported, {int(row['offers']):,} offers were made. Fewer offers raise the forecast yield but bring "
                "fewer students in total. The options describe the model's behaviour, not a proven response."
            )
