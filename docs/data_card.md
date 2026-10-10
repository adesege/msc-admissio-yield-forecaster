# Data card

## Source

Integrated Postsecondary Education Data System (IPEDS), National Center for Education Statistics, U.S. Department of Education. Admissions and Test Scores files for 2022, 2023 and 2024, joined to the 2023 Institutional Characteristics directory file. All files are public domain.

## Coverage

- 5,920 institution-year records across three collection years.
- 1,852 institutions report usable yield in all three years.

## Known quality issues

- Header spelling differs between years, including trailing spaces in some 2022 column names.
- Files use different encodings, and the directory file starts with a byte-order mark.
- 89 records report applicants without usable admit or enrolment counts.
- Gender splits exist for every funnel stage; race and ethnicity splits do not.
- Two tribal colleges appear in 2023, too few for group-level evaluation.

## Intended use

Training and evaluating institution-level forecasts of applications and enrolment yield. The data never inform decisions about individual applicants.

## Versioning

Raw files are committed unchanged under `data/raw/ipeds/` and tagged by release year (for example `data-ipeds-2024`). Revised NCES releases replace the provisional file in a new tagged commit.
