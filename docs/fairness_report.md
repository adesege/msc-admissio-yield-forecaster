# Fairness metrics report: Admissio Yield Forecaster 0.3.0

Reproduce with `python -m src.models.train` followed by `python -m src.fairness.report`. Numbers below come from `reports/training_summary.json` and `reports/alert_fairness.json` for the 2024 hold-out cycle.

## Protected groups and why they are institution attributes

The forecaster works on institutions, not applicants, so its protected groups are institution attributes that track who those institutions serve. Historically Black colleges and universities (HBCU) carry the Module 1 fairness objective. Institutions under 1,000 students and private for-profit institutions were added in Module 2 after the naive baseline showed the largest error gaps for them. Tribal colleges are reported as unassessed because only two appear in the panel.

For group-level metrics each institution is placed in one group, in this order: private for-profit, under 1,000 students, HBCU, all other institutions. Group error ratios compare each attribute with every institution outside it, so the 13 HBCUs with fewer than 1,000 students count as HBCUs in the ratio table and as small institutions in the alert and interval tables.

## 1. Forecast error by group (primary metric)

The Module 2 rule is that each group's median absolute error must stay within 1.25 times the median error for every other institution.

| Group | Institutions | Baseline ratio | Model ratio | Baseline group MAE | Model group MAE | Model MAE for the rest |
|---|---|---|---|---|---|---|
| HBCU | 62 | 1.39 | 1.01 | 4.69 | 3.42 | 5.39 |
| Under 1,000 students | 596 | 2.71 | 3.76 | 11.73 | 10.93 | 2.66 |
| Private for-profit | 136 | 3.81 | 4.75 | 14.02 | 13.44 | 4.68 |

HBCUs now meet the limit. Small and for-profit institutions are better served in absolute terms than under the baseline, but their ratios worsen because the rest of the sector improves by more. Median binomial sampling noise in observed yield sets a floor under these errors:

| Group | Median sampling noise in yield (points) |
|---|---|
| All other institutions | 0.61 |
| HBCU | 0.42 |
| Under 1,000 students | 1.85 |
| Private for-profit | 3.24 |

A for-profit institution with 81 offers has a sampling standard deviation of about 3 points before any modelling error is added, five times the noise at a large institution. A ratio limit of 1.25 cannot be met for these groups by improving the model alone.

## 2. Bias mitigation applied to the forecaster

Five candidates built on the tuned LightGBM model were scored with out-of-fold predictions on the 2023 cycle. The rule fixed before scoring was: keep candidates whose cross-validated MAE is within 5% of the unmitigated model, then pick the lowest worst-group ratio.

| Candidate | CV MAE | CV worst ratio | Hold-out MAE | Hold-out ratios (HBCU, small, for-profit) |
|---|---|---|---|---|
| Unmitigated | 5.45 | 4.97 | 5.23 | 0.91, 3.71, 4.69 |
| Group reweighted (selected) | 5.72 | 4.82 | 5.33 | 1.01, 3.76, 4.75 |
| Bounded group loss 0.06 | 5.74 | 5.63 | 5.24 | 0.87, 4.16, 5.53 |
| Bounded group loss 0.08 | 5.55 | 5.29 | 5.28 | 0.86, 3.99, 5.25 |
| Bounded group loss 0.10 | 5.45 | 4.97 | 5.23 | 0.91, 3.71, 4.69 |

Group reweighting gives each fairness group equal total weight in training. Bounded group loss uses Fairlearn's exponentiated gradient reduction; the bound of 0.10 never binds and reproduces the unmitigated model. Tighter bounds cannot be met because the for-profit group's error sits above them, so the reduction trades accuracy for no fairness gain. On the hold-out cycle the selected model costs 0.10 points of MAE against the unmitigated model and does not improve the ratios, which confirms that the remaining gap is driven by noise rather than by the model under-weighting small institutions.

The mitigation that does work is group-calibrated uncertainty. Intervals are calibrated separately for each group from out-of-fold residuals:

| Group | Interval half-width (points) | 2024 coverage | Median width (points) |
|---|---|---|---|
| All other institutions | 3.39 | 79.7% | 6.8 |
| HBCU | 3.49 | 77.6% | 7.0 |
| Under 1,000 students | 18.15 | 82.2% | 36.3 |
| Private for-profit | 22.75 | 81.6% | 45.5 |

Every group lands close to the 80% target, so partners at small institutions are told how uncertain their forecast is instead of receiving a falsely precise number.

## 3. Yield-drop alert (classification metrics)

The product raises an alert when the forecast falls 3 points or more below last cycle's yield. The ground truth is whether yield actually fell by 3 points or more. Demographic parity, equalized odds and the disparate impact ratio are computed on this decision with Fairlearn.

| Group | Institutions | Actual drop rate | Alert rate | True positive rate | False positive rate |
|---|---|---|---|---|---|
| All other institutions | 1,183 | 20.3% | 10.7% | 37.1% | 3.9% |
| HBCU | 49 | 32.7% | 22.4% | 56.2% | 6.1% |
| Under 1,000 students | 484 | 31.2% | 27.7% | 42.4% | 21.0% |
| Private for-profit | 136 | 38.2% | 36.8% | 48.1% | 29.8% |

| Metric | Single threshold | Equalized-odds thresholds |
|---|---|---|
| Demographic parity difference | 0.261 | 0.097 |
| Equalized odds difference | 0.258 | 0.167 |
| Disparate impact ratio | 0.29 | 0.82 |
| Meets four-fifths rule | No | Yes |
| Overall alert rate | 17.3% | 45.0% |
| Overall false positive rate | 9.6% | 39.3% |
| Balanced accuracy | 0.656 | 0.615 |

The disparate impact ratio fails the four-fifths rule, but the direction matters. Alerts fire more often for small and for-profit institutions, and their real drop rates are also higher, so demographic parity would require suppressing warnings that are partly justified. Equalized odds is the more relevant test, and it shows the real harm: for-profit institutions receive false alarms at 7.6 times the rate of other institutions, which could push them to over-offer.

Fairlearn's `ThresholdOptimizer` with an equalized-odds constraint, fitted on 2023 out-of-fold scores, brings the ratio to 0.82 and narrows the equalized odds gap. It does so by raising false alarms for every group to about 40%, lowering balanced accuracy, and randomising alerts near the threshold. The deployed service therefore keeps the single threshold, reports the interval next to every alert, and flags the false-alarm gap to the Ethics Review Panel.

## 4. Decision for the Ethics Review Panel

Version 0.3.0 is registered with the alias `candidate` and the tag `release_status: not approved`. Two questions need the panel's ruling before release: whether a noise-adjusted error ratio should replace the raw 1.25 rule for small and for-profit institutions, and whether the yield-drop alert should be shown to those groups only when the interval excludes last cycle's yield.
