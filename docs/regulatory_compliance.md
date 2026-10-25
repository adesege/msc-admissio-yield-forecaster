# Regulatory compliance report

The forecaster's partners are Nigerian institutions, and Admissio operates from Lisbon, so the Nigeria Data Protection Act 2023 (NDPA) and the EU General Data Protection Regulation (GDPR) apply. The brief also asks about the California Consumer Privacy Act (CCPA) and HIPAA; both are assessed below.

| Law | Applies? | Why | Controls in place | Gaps |
|---|---|---|---|---|
| NDPA 2023 | Yes, to partner data | Partners process personal data of applicants in Nigeria | Applicant-level extracts are deleted at the end of each run; only suppressed, pseudonymised aggregates reach the model; forecasts are advisory and every offer plan has named human approval, consistent with the Act's limits on decisions based solely on automated processing; audit log and access matrix | Data protection impact assessment to be completed with each partner before the pilot; breach procedure (72-hour notice to the Commission) rehearsed with partner data protection leads |
| GDPR | Yes, to processing by Admissio's EU establishment | Article 3(1) covers processing in the context of an establishment in the Union, wherever it takes place | Data minimisation by design (institution totals only), retention limits, no solely automated decisions with legal or similar effects on individuals (Article 22), records of processing through the audit log | Record the lawful basis and joint-controller terms in each partner agreement |
| CCPA | No | The forecaster processes no personal information of California residents; IPEDS inputs are institution totals | None needed | Reassess if Admissio serves US institutions |
| HIPAA | No | Admissio is not a health plan, health care clearinghouse or health care provider, and no health information is processed | None needed | None |

## Control map

| Requirement theme | Implementation |
|---|---|
| Data minimisation | Model inputs are 12 institution-level features; applicant fields never enter the panel |
| Purpose limitation | Model card lists out-of-scope uses; partner exports are classified confidential and cannot be read by analysts |
| Storage limitation | `src/governance/policy.yaml` retention rules; explanation store records deleted after 1,095 days |
| Transparency and explanation | SHAP drivers on every forecast, LIME and counterfactuals in the dashboard, public trust statement |
| Human oversight | Advisory notice on every response; named approval of every offer plan; panel approval of releases |
| Security and integrity | Hash-chained explanation store; audit log of every data access; role-based access checked in CI |
| Accountability | Ethics Review Panel, MLflow version tags for every release decision, incident issues |
