# Ethical AI framework

The framework brings together every ethical commitment made for the Admissio Yield Forecaster, shows where each one is enforced in the system, and names who oversees it. It follows the four functions of the NIST AI Risk Management Framework: Govern, Map, Measure and Manage.

## Scope

The forecaster predicts the share of admitted students who will enrol at an institution in the next cycle, with an 80% range, a yield-drop alert, an offer plan and the drivers behind each forecast. It is advisory. It never sees applicant records, never ranks applicants and never issues offers.

## Ethical AI Charter

| Principle | NIST function | Commitment | Where it is enforced |
|---|---|---|---|
| Accountability | Govern | The Ethics Review Panel approves every release, and a named owner answers for each forecast shown to a partner | Registry aliases (`candidate`, `challenger`, `champion`) and `src/models/lifecycle.py`, which refuses promotion without a panel reference |
| Transparency | Map | Each release documents its intended use, its limits and the institution types for which evidence is thin | Model card, transparency page in the dashboard, release banner, `/health` release status |
| Fairness | Measure | Forecast error is reported for every monitored group, not as one overall figure | Fairness report, Ethical compliance page, monitoring checks by group |
| Responsiveness | Manage | A release that breaches a fairness objective is withheld until the gap is fixed or openly explained | Candidate status of version 0.3.0, incident issues, decommissioning plan |

## Traceability from commitment to evidence

| Commitment (module) | Evidence in the system | Status |
|---|---|---|
| Forecasts advise, humans decide (1) | Advisory notice on every API response; staff approval wording in the dashboard and offer planner | Met |
| HBCU error within 1.25 times other institutions (1) | Ratio 1.01 on the 2024 hold-out | Met |
| Small and for-profit institutions monitored (2) | Ratios 3.76 and 4.75, red in monitoring | Not met; open incident |
| Representation bias detected in the pipeline (3) | For-profit representation ratio 0.83, amber in monitoring | Disclosed, monitored |
| Personal data removed before modelling (3) | Partner extracts deleted after each run; the model trains on public institution totals, and only suppressed partner aggregates are kept for later training | Met |
| Explanations for every forecast (4) | Top three SHAP drivers in every response; LIME and counterfactual tabs; explanation store | Met |
| Release gate of 4.8 MAE (4) | 5.33 hold-out, 5.32 for the retrained challenger | Not met; candidate only |
| Capacity overrun below 10% (1) | Offer planner overruns 7.2% of institutions against 33.2% for the naive rule | Met, at a cost of unfilled places |

## Ethical governance structure

| Body or role | Membership | Responsibilities | Cadence |
|---|---|---|---|
| Ethics Review Panel | The Admissio co-founder who does not build the model, the data protection lead of each partner institution and one external adviser in education data | Approves releases, promotions and retirements; rules on fairness criteria; closes incidents | Each admissions cycle and on any high or critical incident |
| Model owner | Admissio data science | Runs training, monitoring and retraining; prepares evidence for the panel | Continuous |
| Data protection lead (partner) | Named by each partner | Approves partner data sharing, receives breach notices, can veto use for its tenant | Each cycle |
| Partner admissions director | Named by each partner | Approves every offer plan; reports concerns | Each cycle |
| Engineering | Admissio engineering | Keeps the API, explanation store and pipelines running to the governance policy | Continuous |

Changes to thresholds, monitored groups or the governance policy go through a pull request reviewed by the model owner and approved by the panel. Panel decisions are recorded on the relevant GitHub issue and, for model changes, as MLflow version tags.

## Related documents

- [Algorithmic impact assessment](algorithmic_impact_assessment.md)
- [Regulatory compliance report](regulatory_compliance.md)
- [Public trust statement](public_trust_statement.md)
- [Monitoring plan](monitoring_plan.md), [incident response plan](incident_response.md) and [decommissioning plan](decommissioning.md)
- [Model card](model_card.md), [fairness report](fairness_report.md), [data governance](governance.md) and [anonymisation plan](anonymization_plan.md)
