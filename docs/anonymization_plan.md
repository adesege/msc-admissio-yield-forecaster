# Data anonymisation plan

## What counts as personal data here

IPEDS files contain institution-level counts and no direct identifiers. Two risks remain. Small gender cells, such as one applicant of another gender at a small college, can point to a real person when combined with other published facts. Partner data from Admissio tenants is applicant-level and contains names, contact details, national identification numbers and dates of birth.

## Controls

| Data | Control | Implementation |
|---|---|---|
| IPEDS gender splits | Cells between 1 and 9 are suppressed. When only one cell in a row is suppressed, the next smallest positive cell is also suppressed so the hidden value cannot be recovered from the total. Rows with a total under 10 lose every split | `src/privacy/suppression.py` |
| Partner direct identifiers | `full_name`, `email`, `phone_number`, `nin`, `date_of_birth` and `home_address` are dropped before any aggregation | `src/privacy/anonymize.py` |
| Partner quasi-identifiers | `state_of_origin` is dropped | `src/privacy/anonymize.py` |
| Tenant identity | Tenant IDs are replaced with a keyed HMAC-SHA256 pseudonym. The key comes from `ADMISSIO_PSEUDONYM_KEY` and is never stored with the data | `pseudonymise` |
| Partner aggregates | Records are reduced to tenant-cycle-programme counts, and any count between 1 and 9 is suppressed | `aggregate_partner_extract` |
| Leak check | The pipeline stops if any identifying column survives, and a Great Expectations suite requires the exact aggregate column set | `assert_no_personal_data`, `partner_aggregates` suite |
| Raw extract | Deleted at the end of the run under the retention policy | `apply_retention` task |

## Demonstration data

No partner has shared applicant data yet. `src/privacy/partner_mock.py` generates a seeded synthetic export of 1,200 applicants across three fictional Nigerian tenants with Faker's `en_NG` locale. The mock exists only to exercise these controls and never enters model training.

## Legal basis

Partner processing follows the Nigeria Data Protection Act 2023 principles of purpose limitation, data minimisation and storage limitation, and the GDPR requirement for data protection by design and by default, because Admissio operates from Portugal. A data protection impact assessment precedes the first real partner extract.
