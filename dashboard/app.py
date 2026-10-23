import sys
from pathlib import Path

# Streamlit puts only the dashboard folder on the import path; the api and src packages live at the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

st.set_page_config(page_title="Admissio Yield Forecaster", page_icon=":material/school:", layout="wide")

navigation = st.navigation(
    [
        st.Page("views/overview.py", title="Overview", icon=":material/dashboard:", default=True),
        st.Page("views/explain.py", title="Explain a forecast", icon=":material/search_insights:"),
        st.Page("views/what_if.py", title="What-if analysis", icon=":material/tune:"),
        st.Page("views/planner.py", title="Offer planner", icon=":material/event_seat:"),
        st.Page("views/fairness.py", title="Ethical compliance", icon=":material/balance:"),
        st.Page("views/about.py", title="About the model", icon=":material/description:"),
    ]
)
navigation.run()
