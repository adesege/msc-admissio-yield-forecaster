# Incident response plan for ethical failures

An ethical incident is any event in which the forecaster, its explanations or its data handling could treat an institution or applicant unfairly, mislead a decision maker or expose personal data. Incidents are tracked as GitHub issues labelled `incident`, and the monitoring workflow opens one automatically when a red signal fires.

## What counts as an incident

| Severity | Examples | First response |
|---|---|---|
| Critical | Partner applicant data exposed; forecasts used to score or reject individual applicants; offers issued from a forecast without named approval | Stop serving forecasts within 4 hours; notify the partner's data protection lead |
| High | A red fairness signal (error ratio, false-alarm multiple, planner overrun) for a monitored group; a stored explanation that does not match the forecast it accompanied | Triage within 2 working days; panel decides whether to pause partner-facing outputs for the group |
| Medium | Red input drift or MAE; a broken explanation chain reported by `ExplanationStore.verify()` | Triage within 5 working days; retraining workflow runs |
| Low | Amber signals that persist for two cycles; a complaint that an explanation was unclear | Review at the next cycle meeting |

## Steps

1. **Detect.** Monitoring opens or updates the incident issue with the red signals. Partners, staff and applicants can also report concerns through the partner's admissions office, which forwards them to Admissio.
2. **Contain.** For critical incidents, retire the served model with `python -m src.models.lifecycle retire` and redeploy, after which `POST /predict` returns 410 for every caller, and tell the affected partner to plan from last year's yield. Dashboard forecasts carry the release banner at all times.
3. **Assess.** Data science reproduces the signal from the monitoring summary and the explanation store, identifies affected institutions and cycles, and records whether offers were already issued.
4. **Notify.** Partners affected by a high or critical incident are told what happened and what they should check. If personal data was involved, the partner and Admissio follow the Nigeria Data Protection Act breach rules, which require notifying the Nigeria Data Protection Commission within 72 hours of becoming aware of a breach that risks individuals' rights, and the GDPR where Admissio's EU establishment processes the data.
5. **Decide.** The Ethics Review Panel chooses one of: accept with a recorded rationale, narrow the scope (for example, stop yield-drop alerts for a group), retrain, or retire. The decision and its reference are recorded on the issue and, for model changes, as an MLflow version tag.
6. **Recover and learn.** The fix ships through a pull request, the monitoring workflow is rerun, and the issue closes only when the signal is green or the panel has recorded why it accepts it. Lessons go into the model card and this plan.

## Open incident at handover

The monitoring run on the 2024 cycle opens a high-severity incident: error ratios of 3.76 for institutions under 1,000 students and 4.75 for private for-profit institutions, above 1.25 for a second cycle, and yield-drop false alarms at 2.19 and 3.09 times the overall rate for the same groups. The proposed containment, pending the panel, is to keep the model at candidate status, suppress yield-drop alerts for those groups unless the 80% interval excludes last year's yield, and decide on a noise-adjusted fairness criterion before the shadow pilot.
