# Data dictionary

Preliminary schema for every source used by the Admissio Yield Forecaster. Raw IPEDS files live in `data/raw/ipeds/` and are loaded by `src/ingest/ipeds.py`.

## Sources

| Source | File | Grain | Records | Publisher | Licence |
|---|---|---|---|---|---|
| IPEDS Admissions and Test Scores 2022 (revised) | `adm2022_rv.csv` | Institution | 1,991 | NCES | Public domain |
| IPEDS Admissions and Test Scores 2023 (revised) | `adm2023_rv.csv` | Institution | 1,973 | NCES | Public domain |
| IPEDS Admissions and Test Scores 2024 | `adm2024.csv` | Institution | 1,956 | NCES | Public domain |
| IPEDS Institutional Characteristics, Directory 2023 | `hd2023.csv` | Institution | 6,163 | NCES | Public domain |
| Admissio partner aggregates | Planned | Institution-cycle | Grows with partners | Partner institutions | Partner data processing agreement |

## IPEDS admissions (ADM)

| Field | Type | Description | Role |
|---|---|---|---|
| `UNITID` | Integer | Unique institution identifier | Join key |
| `APPLCN` | Integer | Applicants, total | Demand target and feature |
| `APPLCNM`, `APPLCNW`, `APPLCNAN`, `APPLCNUN` | Integer | Applicants by gender (men, women, another gender, unknown) | Feature |
| `ADMSSN` | Integer | Admissions (offers), total | Denominator of yield |
| `ADMSSNM`, `ADMSSNW`, `ADMSSNAN`, `ADMSSNUN` | Integer | Admissions by gender | Feature |
| `ENRLT` | Integer | Admitted applicants who enrolled, total | Numerator of yield |
| `ENRLM`, `ENRLW`, `ENRLAN`, `ENRLUN` | Integer | Enrolled by gender | Feature |
| `ENRLFT`, `ENRLPT` | Integer | Enrolled full time and part time | Feature |
| `ADMCON1` to `ADMCON12` | Code | Admission considerations: GPA, rank, record, college-prep programme, recommendations, competencies, test scores, English proficiency, other test, work experience, essay, legacy | Feature |
| `ADMCON7` | Code | Admission test scores: 1 required, 5 test optional, 3 test blind | Feature; policy-change flag |
| `SATNUM`, `SATPCT`, `ACTNUM`, `ACTPCT` | Integer | Number and percent of enrollees submitting SAT or ACT scores | Feature |
| `SATVR25` to `ACTMT75` | Integer | 25th, 50th and 75th percentile SAT and ACT scores | Feature |
| `X*` columns | Code | Imputation flags (R reported, A not applicable, and others) | Data-quality check |

## IPEDS directory (HD2023)

| Field | Type | Description | Role |
|---|---|---|---|
| `UNITID` | Integer | Unique institution identifier | Join key |
| `INSTNM` | Text | Institution name | Display only |
| `STABBR` | Text | State abbreviation | Feature |
| `OBEREG` | Code | Bureau of Economic Analysis region | Feature |
| `SECTOR`, `CONTROL` | Code | Sector; control (1 public, 2 private nonprofit, 3 private for-profit) | Feature; fairness group |
| `HBCU` | Code | Historically Black college or university (1 yes, 2 no) | Fairness group |
| `TRIBAL` | Code | Tribal college (1 yes, 2 no) | Fairness group, reported as unassessed below 30 institutions |
| `LOCALE` | Code | Degree of urbanisation | Feature |
| `INSTSIZE` | Code | Enrolment size category (1 under 1,000 to 5 over 20,000) | Feature; fairness group |
| `C21BASIC` | Code | Carnegie Classification 2021, basic | Feature |

## Derived panel fields (`src/features/panel.py`)

| Field | Type | Definition | Role |
|---|---|---|---|
| `YEAR` | Integer | IPEDS collection year | Time index |
| `YIELD` | Float | `ENRLT / ADMSSN`, missing when `ADMSSN` is 0 or missing | Primary target |
| `YIELD_LAG1` | Float | Previous year's yield for the same institution | Naive baseline and feature |
| `APPLCN_CHANGE` | Float | Year-on-year proportional change in applicants | Feature |
| `TEST_POLICY_CHANGED` | Boolean | `ADMCON7` differs from the previous year | Feature |
| `SIZE_GROUP`, `CONTROL_GROUP`, `HBCU_GROUP`, `TRIBAL_GROUP` | Text | Labelled fairness groups | Fairness audit and bias detection |
| `FUNNEL_COMPLETE` | Boolean | Applicants, admits and enrolled all reported, with admits above zero | Data-quality flag |
| `FULL_YIELD_HISTORY` | Boolean | Usable yield in every year of the panel | Modelling eligibility |

## Model features (`src/models/features.py`)

| Field | Type | Definition | Role |
|---|---|---|---|
| `ADMIT_RATE` | Float | `ADMSSN / APPLCN` for the current cycle, known once offers are planned | Feature; counterfactual lever |
| `ADMIT_RATE_LAG1` | Float | Previous cycle's admit rate | Feature |
| `LOG_APPLCN` | Float | Natural log of one plus applicants | Feature |
| `APPLCN_CHANGE` | Float | Year-on-year change in applicants, clipped to -0.9 and 3.0 | Feature |
| `FAIRNESS_GROUP` | Text | Private for-profit, under 1,000 students, HBCU or all other institutions, assigned in that order | Mitigation weights, intervals and alert audit; never a feature |
| Model target | Float | `YIELD - YIELD_LAG1`; the forecast adds it back to `YIELD_LAG1` | Learned change in yield |

Columns measured on enrolled students (`ENRL*`, `SATNUM`, `SATPCT`, `ACTNUM`, `ACTPCT` and score percentiles) are rejected as features.

## Pipeline outputs (`data/processed/warehouse.duckdb`)

| Table or view | Grain | Rows (2022 to 2024) | Purpose |
|---|---|---|---|
| `analytics.yield_panel` | Institution-year | 5,908 | Validated panel for analysis |
| `analytics.model_ready_panel` | Institution-year | 5,556 | View of institutions with full yield history, the Module 4 training input |
| `quarantine.admissions` | Institution-year | 12 | Rows with no directory record, held back with the reason |
| `public.funnel_by_gender` | Institution-year | 5,908 | Gender funnel counts after small-cell suppression |
| `reports.representation` | Group | 12 | Bias detection results |
| `analytics.partner_aggregates` | Tenant-cycle-programme | Varies | Anonymised partner counts |

## Planned Admissio partner aggregates

| Field | Type | Definition |
|---|---|---|
| `tenant_id` | Text | Partner institution identifier, never shown to other tenants |
| `cycle` | Text | Admission cycle, for example 2026/2027 |
| `applications`, `offers`, `acceptances`, `enrolled` | Integer | Counts per cycle and programme, aggregated before leaving the tenant |
| `programme_code` | Text | Programme identifier |

No applicant-level fields enter the modelling pipeline.
