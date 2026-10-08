# Admissio Yield Forecaster

Institution-level forecasting of applications and enrolment yield for admissions teams on the Admissio platform. Partner institutions use the forecasts to set offer volumes that fill places without overcrowding.

Built on public IPEDS admissions data (2022 to 2024) while partner data accumulates. Forecasts advise people; they never decide admissions.

## Repository layout

```
admissio-yield-forecaster/
├── data/          # raw and processed IPEDS files, tagged by release year
├── docs/          # data card, model card, data dictionary, architecture, RAID log
├── src/           # ingest/, features/, models/, fairness/
├── dashboard/     # React front end served by Cloudflare Workers
├── tests/         # unit and data validation tests
├── CHANGELOG.md
└── README.md
```

## Setup

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Reproduce the baseline

```bash
python -m src.models.baseline
```

This evaluates the carry-last-year's-yield-forward baseline on the 2024 cycle and prints its error, intake misses and fairness ratios.

## Roadmap

| Sprint | Course module | Deliverable |
|---|---|---|
| 1 | Module 2 | Project plan, architecture, repository, data dictionary, RAID log |
| 2 | Module 3 | Validated institution-year panel in DuckDB, data card |
| 3 | Module 4 | Demand and yield models, offer planner, SHAP and fairness audit, model card |
| 4 | Module 5 | Forecast API on Cloudflare Workers, React dashboard, stakeholder presentation |
| 5 | Final Project | Deployed, monitored release |

## Governance

Releases follow GitHub Flow. Models use Semantic Versioning in the MLflow registry. Every release is approved by the Ethics Review Panel.

## Data source

National Center for Education Statistics. Integrated Postsecondary Education Data System. U.S. Department of Education. https://nces.ed.gov/ipeds/
