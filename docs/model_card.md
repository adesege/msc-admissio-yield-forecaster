# Model card

## Model

Admissio Yield Forecaster: institution-level forecasts of next-cycle applications and enrolment yield, with 80% prediction intervals, plus an offer planner that converts a target intake into a recommended number of offers.

## Intended users

Admissions directors, enrolment analysts and finance leads at Admissio partner institutions.

## Out-of-scope uses

- Ranking, scoring or rejecting individual applicants.
- Any automated decision made without named human approval.

## Evaluation

| Metric | Naive baseline (2024 holdout) | Release threshold |
|---|---|---|
| Yield mean absolute error | 6.0 points | 4.8 points or lower |
| Institutions with intake error above 10% | 51.1% | Lower than baseline |
| Capacity overshoot risk below 10% | Not modelled | 90% of institutions |
| Median error ratio, each fairness group vs rest | HBCU 1.39, under 1,000 students 2.71, private for-profit 3.81 | 1.25 or lower |

Baseline figures are reproduced by `python -m src.models.baseline`.

## Explainability

Each forecast is shown with its three largest SHAP contributions. Drivers describe association, not cause.

## Release approval

Every model version is approved by the Ethics Review Panel before it reaches a partner dashboard.
