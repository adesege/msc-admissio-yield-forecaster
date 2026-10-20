import pandas as pd
import streamlit as st

from api.main import LoadedModel, load_model
from dashboard.core import DashboardData, load_dashboard_data

SELECTED_KEY = "selected_unitid"
RELEASE_NOTICE = (
    "Candidate model, not approved for release. It was trained on public US data and has not been checked on "
    "Nigerian partner data. Forecasts are advisory, and offer decisions stay with named institution staff."
)


@st.cache_resource
def model() -> LoadedModel:
    return load_model()


@st.cache_data
def dashboard_data() -> DashboardData:
    return load_dashboard_data()


def release_banner() -> None:
    st.warning(RELEASE_NOTICE, icon=":material/info:")


def institution_picker(data: DashboardData) -> pd.Series:
    examples = data.example_unitids()
    holdout = data.holdout.sort_values("institution")
    others = [u for u in holdout["unitid"] if u not in examples.values()]
    options = list(examples.values()) + others
    names = dict(zip(data.holdout["unitid"], data.holdout["institution"] + " (" + data.holdout["state"] + ")"))
    sample_names = {unitid: f"Sample: {label}" for label, unitid in examples.items()}

    current = st.session_state.get(SELECTED_KEY, options[0])

    def remember_choice() -> None:
        st.session_state[SELECTED_KEY] = st.session_state["institution_choice"]

    st.selectbox(
        "Institution from the 2024 test cycle",
        options,
        index=options.index(current),
        format_func=lambda unitid: sample_names.get(unitid, names[unitid]),
        key="institution_choice",
        on_change=remember_choice,
    )
    st.session_state[SELECTED_KEY] = st.session_state["institution_choice"]
    return data.institution(st.session_state[SELECTED_KEY])


def percent(value: float, digits: int = 0) -> str:
    return f"{100 * value:.{digits}f}%"
