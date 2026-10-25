# Model card: Admissio Yield Forecaster 0.3.0

## Model details

| Item | Value |
|---|---|
| Name and version | `admissio-yield`, semantic version 0.3.0, MLflow registry version 2 |
| Registry alias | `candidate` (the `champion` alias is reserved for a version the Ethics Review Panel approves) |
| Release status | Not approved for release |
| Algorithm | LightGBM gradient boosting with absolute-error objective, trained with group-balanced sample weights |
| Output | Next-cycle enrolment yield, an 80% prediction interval, a yield-drop alert and the three largest SHAP drivers |
| Serialised model | `models/admissio_yield.joblib`, served by `api/main.py` at `POST /predict` |
| Owner | Admissio data science; approval by the Ethics Review Panel |

The model forecasts the change in yield from last cycle and adds it to last cycle's yield. Forecasts are clipped to the range 0 to 1.

## Intended use

Admissions directors, enrolment analysts and finance leads at Admissio partner institutions use the forecast to decide how many offers to make for a target intake. The forecast is made after applications close and offers are planned, and before enrolment deposits are confirmed.

## Out-of-scope uses

- Ranking, scoring or rejecting individual applicants. The model never sees applicant records.
- Automated offer decisions without named human approval.
- Comparing institutions against each other for funding, accreditation or league tables.
- Forecasting for institutions with no previous cycle of admissions data.

## Training and evaluation data

IPEDS Admissions and Test Scores files for 2022 to 2024 joined to the 2023 directory file, read from `analytics.model_ready_panel` produced by the Module 3 pipeline. The unit of analysis is one institution in one admission cycle.

| Split | Cycle | Institutions | Role |
|---|---|---|---|
| Training | 2023 (lags from 2022) | 1,852 | Fitting, 5-fold GroupKFold tuning by `UNITID`, mitigation selection, interval calibration |
| Hold-out | 2024 (lags from 2023) | 1,852 | Final evaluation only |

Features: last cycle's yield, admit rate this cycle and last cycle, change in applicants, applicant volume, test-score policy and whether it changed, control, size band, locale, region and Carnegie class. Every column measured on enrolled students (`ENRL*`, SAT and ACT submission counts and percentiles) is excluded and rejected by `assert_no_leakage`. The HBCU flag is used for fairness monitoring only and is never a model input.

## Performance on the 2024 hold-out

| Metric | Naive baseline | This model | Release threshold | Met |
|---|---|---|---|---|
| Yield mean absolute error | 5.99 points | 5.33 points | 4.8 points or lower | No |
| Yield RMSE | 12.82 points | 10.68 points | Not set | |
| R² | 0.791 | 0.855 | Not set | |
| Institutions with intake missed by more than 10% | 51.1% | 47.7% | Lower than baseline | Yes |
| 80% interval coverage | Not modelled | 80.5% | 75% to 85% in every group | Yes |

Cross-validated MAE on the 2023 cycle was 5.72 points. Retraining with five random seeds moved hold-out MAE by 0.07 points.

## Fairness

| Group (vs all other institutions) | Institutions | Median error ratio | Baseline ratio | Limit | Group MAE | Baseline group MAE |
|---|---|---|---|---|---|---|
| HBCU | 62 | 1.01 | 1.39 | 1.25 | 3.42 | 4.69 |
| Under 1,000 students | 596 | 3.76 | 2.71 | 1.25 | 10.93 | 11.73 |
| Private for-profit | 136 | 4.75 | 3.81 | 1.25 | 13.44 | 14.02 |

The model closes the HBCU gap and lowers absolute error for every monitored group. The ratios for small and for-profit institutions still breach the 1.25 limit because errors fall faster for large institutions. Median binomial sampling noise in yield is 3.24 points for for-profit institutions and 1.85 points for small institutions, against 0.61 points for other institutions, so part of the gap cannot be removed by any model. Intervals are calibrated by group, so 80% intervals are about 45 points wide for for-profit institutions and 36 points for small institutions, against 7 points elsewhere.

The yield-drop alert (forecast fall of 3 points or more) fails the four-fifths rule with a disparate impact ratio of 0.29, mostly because small and for-profit institutions see more real drops and more false alarms. Equalized-odds thresholds raise the ratio to 0.82 but quadruple the false-alarm rate, so they are documented and not deployed. Full results are in `docs/fairness_report.md`.

## Explainability

Every forecast returns its three largest SHAP contributions. Last cycle's yield, the change in applicants, applicant volume and the two admit rates account for most of the model's output; institutional descriptors contribute a few tenths of a point. Drivers describe association, not cause. The SHAP notebook is `notebooks/shap_analysis.ipynb`; LIME explanations, counterfactual offer scenarios and sensitivity tests are produced by `python -m src.explain.run`.

## Limitations and risks

- Three IPEDS cycles allow one training cycle and one test cycle, so the model has not been checked across several years or through an unusual cycle.
- The training population is US institutions. Nigerian partner institutions differ in funnel structure and scale, so the model must be revalidated on partner aggregates before any partner sees a forecast.
- For-profit institutions are under-represented in the modelling set (representation ratio 0.83 from Module 3).
- Counterfactual offer scenarios show how forecasts move with admit rate in historical data; they do not prove that changing offers will change yield.

## Acceptance criteria for release

1. Hold-out yield MAE of 4.8 points or lower.
2. Share of institutions with intake missed by more than 10% below the baseline.
3. Median error ratio of 1.25 or lower for each monitored group, or a noise-adjusted criterion approved by the Ethics Review Panel.
4. 80% interval coverage between 75% and 85% for every monitored group.
5. Written approval from the Ethics Review Panel recorded as the `release_status` tag in the registry.

## Offer planner

The planner turns the forecast into a recommended offer volume: the largest number of offers whose upper 80% bound on intake stays within the places available, which holds the modelled chance of exceeding capacity to about 10%. Replayed on the 2024 cycle with capacity set to each institution's actual intake, it exceeded capacity at 7.2% of institutions against 33.2% when offers were sized from last year's yield, and stayed at or below 10% in every monitored group. The cost is caution: the typical institution filled 83% of its places against 98% under the simple rule, and wider intervals make the planner more cautious for small and for-profit institutions. The replay assumes each institution keeps its actual forecast error and that yield responds to offer volume only as the model expects.

## Retraining

`python -m src.models.retrain` refits this configuration on every available cycle and registers the result with the `challenger` alias. On the pooled 2023 and 2024 cycles the challenger reached a cross-validated MAE of 5.32 points, with error ratios of 0.89 (HBCU), 3.82 (under 1,000 students) and 5.40 (private for-profit), so it fails the same gates. Promotion needs Ethics Review Panel approval through `python -m src.models.lifecycle promote`.

## Monitoring after deployment

Monitoring runs weekly and on every merge, and publishes to the monitoring dashboard at https://adesege.github.io/msc-admissio-yield-forecaster/. The full plan is in `docs/monitoring_plan.md`.

| Signal | Trigger |
|---|---|
| Population stability index on each input | Above 0.2 for any feature |
| Rolling yield MAE once actual enrolment lands | Above 6.0 points |
| Median error ratio by group | Above the limit for two cycles in a row |
| Interval coverage by group | Outside 70% to 90% |
| Alert false-positive rate by group | Any group more than double the overall rate |
| Share of rows or requests with a missing input | More than 5 points above the training cycle (8.3%, mostly Carnegie class) |
| Offer planner overrun by group | Above 15% of institutions |

Every forecast served by the API is stored with its inputs, interval and drivers in a hash-chained explanation store, retrievable at `GET /explanations/{request_id}`. On the 2024 cycle no input drifted (largest PSI 0.040) and MAE stayed at 5.33 points, but the small and for-profit error ratios and alert false-alarm multiples are red, which opens an incident for the Ethics Review Panel.
