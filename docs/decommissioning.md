# Decommissioning plan

The forecaster will be retired when it can no longer be trusted or is no longer needed. Retirement must not leave partners without a way to plan offers, and it must keep enough evidence to answer later questions about decisions the model informed.

## Triggers

- The Ethics Review Panel judges a critical incident cannot be fixed.
- MAE stays red for two cycles after retraining.
- A partner-trained model replaces the US-trained one.
- A partner withdraws consent for its aggregates, or a regulator directs Admissio to stop.

## Steps

| Step | Action | Owner |
|---|---|---|
| 1. Decide | Panel records the reason and an approval reference | Ethics Review Panel |
| 2. Notify | Partners receive at least one admissions cycle of notice, except after a critical incident, together with the fallback below | Admissio product |
| 3. Fallback | Partners plan offers from last year's yield and their own records; the dashboard explains the method | Partner admissions directors |
| 4. Stop serving | `python -m src.models.lifecycle retire --reason ... --approval-reference ...` marks the model retired, archives its files under `models/archive/<version>/`, removes every registry alias and tags the MLflow version as retired. Once the change is deployed or the API restarts, it answers `POST /predict` with 410 and `/health` reports `retired` | Admissio engineering |
| 5. Wind down | Remove the Render service and the monitoring schedule; keep the monitoring site's last report | Admissio engineering |
| 6. Retain | Keep the archived model, model card, fairness report and MLflow runs for 1,095 days after the last forecast, so any stored forecast can be reproduced until its own record in the explanation store expires; the audit log and explanation store keep their 1,095-day rule | Data engineering |
| 7. Delete | After the retention period, delete partner aggregates, the explanation store and the archived model; log each deletion in the audit log | Data engineering |
| 8. Close | Publish a short retirement note in the repository and inform partners that deletion is complete | Ethics Review Panel |

Public IPEDS files are kept, because they are public and needed to reproduce published results.
