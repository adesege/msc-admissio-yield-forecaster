# Data governance framework

The framework covers who may touch each dataset, for what purpose, and for how long it is kept. The machine-readable policy is `src/governance/policy.yaml`. Every read, write, anonymisation and deletion in the pipeline is checked against that policy before it happens and recorded in the privacy audit log.

## Roles

| Role | Holder | Responsibility |
|---|---|---|
| `pipeline_service` | Automated service account | Runs the scheduled pipeline with the narrowest rights each step needs |
| `data_engineer` | Pipeline maintainer | Maintains code, schemas and validation suites; cannot write analytical outputs by hand |
| `analyst` | Model developer | Reads the validated panel and public outputs; never sees partner applicant records |
| `ethics_review_panel` | Reviewers named in the Ethical AI Charter | Reads every internal output and the audit log; approves releases |
| `partner_admin` | Admissions staff at one partner | Exports its own tenant's records and reads public outputs |

## Dataset classification and access

| Dataset | Classification | pipeline_service | data_engineer | analyst | ethics_review_panel | partner_admin |
|---|---|---|---|---|---|---|
| `ipeds_raw` | Public | read | read, write | read | read | none |
| `quarantine` | Internal | write, delete | read | none | read | none |
| `yield_panel` | Internal | read, write, delete | read | read | read | none |
| `public_funnel` | Public | write, delete | read | read | read | read |
| `representation_report` | Internal | write, delete | read | read | read | none |
| `partner_extract` | Confidential | read, anonymise, delete | none | none | none | write |
| `partner_aggregates` | Internal | write, delete | read | read | read | none |
| `audit_log` | Restricted | append, delete | read | none | read | none |

A request outside this matrix raises `AccessDenied` and is still written to the audit log with the outcome `denied`.

## Retention

| Dataset | Rule | Reason |
|---|---|---|
| `ipeds_raw` | Keep indefinitely | Public source files; needed to reproduce any past run |
| `partner_extract` | Delete at the end of every run | Applicant-level personal data is needed only long enough to aggregate it |
| `yield_panel`, `quarantine`, `public_funnel`, `partner_aggregates`, `representation_report` | Keep the latest two versions | One current release and one for rollback |
| `audit_log` | Delete after 1,095 days (three years) | Long enough to answer a partner or regulator query about any training cycle |

Retention runs as the last task of every pipeline run, and each deletion is itself logged.

## Change control

Changes to `policy.yaml` go through a pull request reviewed by the data engineer and approved by the Ethics Review Panel. Continuous integration runs the governance tests, so a change that grants analysts access to partner extracts fails the build.
