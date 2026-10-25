# Monitoring plan

The plan describes how the forecaster is checked after deployment. It watches four things: whether the inputs still look like the data the model learned from, whether forecasts stay accurate, whether accuracy and alerts stay fair across institution types, and whether the offer planner keeps intake within capacity. Every threshold below comes from the model card, and the code that applies them is `src/monitoring/checks.py`.

## Where monitoring runs

| Layer | Tool | What it shows | Where |
|---|---|---|---|
| Cycle monitoring | `src/monitoring/run.py` with Evidently and Fairlearn | Drift, accuracy, fairness, coverage, false alarms, representation, planner overrun | Monitoring dashboard on GitHub Pages, rebuilt weekly and on every push to `main` |
| Live service | Prometheus and Grafana in Docker Compose | Forecasts served by group, alert rate by group, forecast distribution, interval width, missing inputs, latency, and the latest cycle status | `http://localhost:3000` after `docker compose up` |
| Audit | Explanation store | Every forecast with its inputs, interval and drivers, hash-chained | `GET /explanations/{request_id}` |

## Key performance indicators

| KPI | Target | Owner |
|---|---|---|
| Yield MAE on the latest cycle with actual enrolment | 4.8 points or lower for release; red above 6.0 | Admissio data science |
| Share of institutions missing intake by more than 10% | Below the last-year baseline | Admissio data science |
| Offer planner: share of institutions over capacity | 10% or lower in every monitored group | Admissio product |
| Offer planner: median share of places filled | Reported each cycle; no target until partners set their own tolerance | Partner admissions directors |
| 80% interval coverage | 75% to 85% in every group | Admissio data science |
| Median error ratio by group | 1.25 or lower, or a noise-adjusted criterion the Ethics Review Panel approves | Ethics Review Panel |
| Forecasts with a stored explanation | 100% | Admissio engineering |

## Continuous fairness monitoring

Fairness is measured every cycle, not only at release. The monitored groups are historically Black colleges and universities (HBCUs), institutions under 1,000 students and private for-profit institutions. Tribal colleges stay unassessed until at least 30 appear. For partner institutions in Nigeria the panel will name equivalent groups before the shadow pilot, for example state universities, newer private universities and institutions outside the main cities.

| Signal | Amber | Red |
|---|---|---|
| Median error ratio against all other institutions | Above 1.25 for one cycle | Above 1.25 for two cycles in a row |
| Bias drift: rise in a group's error ratio since the previous cycle | More than 0.25 | Not used alone |
| 80% interval coverage by group | Outside 75% to 85% | Outside 70% to 90% |
| Yield-drop alert false-alarm rate relative to the overall rate | Above 1.5 times | Above 2 times |
| Representation in the modelling set | Below 0.90 | Below 0.80 |
| Offer planner overrun share | Above 10% | Above 15% |

Fairlearn computes the group metrics. The previous-cycle reference is the cross-validated error on the training cycle, so the first live cycle already has a comparison point.

## Model drift and bias drift detection

| Signal | Method | Amber | Red |
|---|---|---|---|
| Input drift | Population stability index for each of the 12 inputs, and Evidently column tests (Wasserstein distance for numeric inputs, Jensen-Shannon divergence for categorical inputs, threshold 0.1) | PSI above 0.1 | PSI above 0.2 |
| Concept drift | Yield MAE once actual enrolment lands, with an Evidently regression report against the reference cycle | Above 5.5 points | Above 6.0 points |
| Missing inputs | Share of rows or requests with a blank input, relative to the training cycle (8.3%, mostly Carnegie class) | More than 2.5 points higher | More than 5 points higher |
| Bias drift | Change in each group's error ratio between cycles | See the fairness table | See the fairness table |

The 2024 cycle, treated as newly arrived data, shows no input drift (largest PSI 0.040, no Evidently column flagged) and a stable MAE of 5.33 points. The red signals are the small and for-profit error ratios, which were already above the limit on the training cycle, and the alert false-alarm multiples for the same groups.

## Responses

| Outcome | Automated response | Human response |
|---|---|---|
| All green | Monitoring site updated | None |
| Any amber | Monitoring site updated | Data science reviews at the next cycle meeting |
| Red drift or error | GitHub issue labelled `incident`; the `retrain` workflow trains a challenger on every available cycle and registers it with the `challenger` alias | Ethics Review Panel decides on promotion with `python -m src.models.lifecycle promote` |
| Red fairness, alert or planner signal | GitHub issue labelled `incident` | Incident response plan; retraining alone does not fix sampling noise, so the panel decides on scope, thresholds or withdrawal |

## Retraining

`python -m src.models.retrain` refits the served configuration (LightGBM with group reweighting) on every available cycle, calibrates group intervals from out-of-fold errors, checks the automated release gates (error, intake misses and group error ratios) and registers the result as a challenger. Promotion is never automatic. On the pooled 2023 and 2024 cycles the challenger reached a cross-validated MAE of 5.32 points with small and for-profit ratios of 3.82 and 5.40, so it also fails the release gates. One more cycle of the same population does not close the gap, which is why the shadow pilot on partner data comes before any release.

## Cadence

| When | What |
|---|---|
| Weekly and on every merge | Monitoring workflow rebuilds the dashboard from the latest data |
| Each admissions cycle, after enrolment deposits | Actual yield lands, accuracy and fairness checks run on real outcomes, and the panel reviews the monitoring dashboard |
| On a red signal | Incident response plan |
| Every pipeline run | Audit log entries older than 1,095 days are deleted |
| Every API start | Explanation store records older than 1,095 days are deleted |
