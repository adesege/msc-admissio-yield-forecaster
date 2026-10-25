# Algorithmic impact assessment

| Item | Detail |
|---|---|
| System | Admissio Yield Forecaster 0.3.0 with offer planner |
| Decision supported | How many offers an institution makes for a target intake |
| Decision maker | Named admissions staff at each partner institution |
| Automation level | Advisory; no decision is made without human approval |
| Data | Public IPEDS institution totals (2022 to 2024); partner aggregates after anonymisation |
| Assessment date | October 2026; repeat before the shadow pilot and at every promotion |

## Who could be affected

| Stakeholder | Possible benefit | Possible harm |
|---|---|---|
| Applicants | More offers made when yield is likely to fall, fewer late rejections from overcrowding | Fewer offers at an institution whose forecast is too high, especially where forecasts are least accurate |
| Partner institutions | Fewer intake misses and steadier fee income | Over-reliance on a model trained on US data; unfilled places when the planner is cautious |
| Small and for-profit institutions | Wider ranges that reflect the real uncertainty of their forecasts | Larger errors and more false alarms than other institutions |
| Admissio | Evidence that analytics improve partner outcomes | Reputational and legal risk if forecasts are misused |
| Regulators (NUC, JAMB, NDPC) | Better use of approved admission quotas | None identified directly |

## Impact findings

| Risk | Likelihood | Severity | Evidence | Mitigation | Residual risk |
|---|---|---|---|---|---|
| Unequal accuracy for small and for-profit institutions | High | Medium | Error ratios 3.76 and 4.75; sampling noise of 1.85 and 3.24 points against 0.61 elsewhere | Group reweighting, group-calibrated intervals, monitoring, open incident | High until the panel sets a noise-adjusted criterion |
| Alert fires more often, and falsely, for the same groups | High | Medium | Disparate impact 0.29; false alarms 2.19 and 3.09 times the overall rate | Equalized-odds thresholds tested and rejected (false alarms quadrupled); proposed suppression unless the interval excludes last year's yield | Medium |
| Over-cautious offer plans leave places empty | Medium | Medium | Planner filled a median 83% of capacity against 98% for the naive rule | Capacity set by each institution; the 10% risk tolerance reviewed by the panel; planner always shown with its range | Medium |
| Transfer from US data to Nigerian institutions | High | High | No partner validation yet | Shadow pilot with two or three partners for one cycle before any release | High until the pilot ends |
| Misuse to score applicants or rank institutions | Low | High | Out-of-scope uses listed in the model card | No applicant inputs exist; advisory notice; partner agreements | Low |
| Personal data exposure | Low | High | Partner extracts deleted after each run; explanation store holds institution totals only | Governance policy, audit log, access matrix | Low |
| Unexplained or altered explanations | Low | Medium | Every forecast stored with a hash chain | `ExplanationStore.verify()` run at every audit and incident review | Low |

## Conclusion

The forecaster may proceed to a shadow pilot, where forecasts are compared with real enrolment but no partner acts on them. It may not be released to partners until the panel resolves the small and for-profit fairness criterion and the pilot confirms accuracy on Nigerian data.
