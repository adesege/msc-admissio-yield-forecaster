import streamlit as st

from dashboard.session import dashboard_data, release_banner

REPOSITORY = "https://github.com/adesege/msc-admissio-yield-forecaster"

data = dashboard_data()
metadata = data.summary["model"]

st.title("About the model")
release_banner()

st.subheader("Transparency statement")
st.markdown(
    """
The Admissio Yield Forecaster estimates the share of admitted students who will enroll at an institution in the
coming cycle, with a range showing how uncertain the estimate is. It is built to support admissions directors,
enrollment analysts and finance leads when they plan offer volumes. It advises; it never decides. Every offer plan is
approved by a named member of institution staff, in line with the right to human review of automated decisions in
the Nigeria Data Protection Act 2023.
    """
)

can, cannot = st.columns(2)
with can:
    st.markdown("**What the model can do**")
    st.markdown(
        """
- Forecast next-cycle yield for an institution with at least one earlier cycle of admissions data.
- Give an 80% range containing the actual result for 78% to 82% of institutions in every group in the 2024 test.
- Show which inputs pushed each forecast up or down.
- Raise an early warning when yield looks likely to fall by 3 points or more.
        """
    )
with cannot:
    st.markdown("**What the model cannot do**")
    st.markdown(
        """
- Score, rank or decide on individual applicants. It never sees applicant records.
- Forecast for an institution in its first admission cycle.
- Speak for Nigerian institutions yet. It was trained on US data and must be revalidated on partner data first.
- Prove cause and effect. Its drivers and what-if scenarios describe patterns in past data.
- Compare institutions for funding, accreditation or league tables.
        """
    )

st.subheader("Model facts")
st.markdown(
    f"""
| Item | Detail |
|---|---|
| Version | {metadata['model_version']}, registry version {metadata['registered_model_version']} |
| Release status | {'Approved' if metadata['approved_for_release'] else 'Candidate, not approved for release'} |
| Method | Gradient-boosted trees (LightGBM) forecasting the change from last cycle's yield |
| Training data | IPEDS admissions, 2023 cycle with 2022 history |
| Test data | IPEDS admissions, 2024 cycle, {len(data.holdout):,} institutions |
| Approval | Ethics Review Panel |
    """
)

st.subheader("Data source")
st.markdown(
    "National Center for Education Statistics, Integrated Postsecondary Education Data System (IPEDS), admissions "
    "files for 2022 to 2024 and the 2023 institutional directory. All figures are institution totals."
)

st.markdown(
    f"Documentation: [model card]({REPOSITORY}/blob/main/docs/model_card.md), "
    f"[fairness report]({REPOSITORY}/blob/main/docs/fairness_report.md), [source code]({REPOSITORY})."
)
