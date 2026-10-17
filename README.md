# Admissio Yield Forecaster

Institution-level forecasting of applications and enrolment yield for admissions teams on the Admissio platform. Partner institutions use the forecasts to set offer volumes that fill places without overcrowding.

Built on public IPEDS admissions data (2022 to 2024) while partner data accumulates. Forecasts advise people; they never decide admissions.

## Repository layout

```
msc-admissio-yield-forecaster/
├── data/          # raw IPEDS files; interim and processed outputs are written at run time
├── docs/          # data card, model card, data dictionary, governance, anonymisation plan, architecture, RAID log
├── src/
│   ├── ingest/      # landing with checksums, IPEDS readers
│   ├── cleaning/    # deduplication, quarantine, reserved-code handling
│   ├── features/    # yield panel and fairness groups
│   ├── validation/  # Great Expectations suites and checkpoint runner
│   ├── fairness/    # representation bias suite and group error audit
│   ├── privacy/     # small-cell suppression, anonymisation, audit log
│   ├── governance/  # access policy and retention rules
│   ├── pipeline/    # Prefect flow, DuckDB warehouse, lineage
│   └── models/      # forecasting models
├── dashboard/     # React front end served by Cloudflare Workers
├── tests/         # pytest suite for every pipeline module
├── Dockerfile
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

## Run the pipeline

```bash
python -m src.pipeline.flow
```

The Prefect flow lands the raw files, validates them with Great Expectations, cleans and integrates the panel, checks representation bias, suppresses small cells, anonymises the partner extract, loads DuckDB and applies retention. Outputs:

| Output | Location |
|---|---|
| Warehouse with `analytics.yield_panel` and the `analytics.model_ready_panel` view | `data/processed/warehouse.duckdb` |
| Versioned Parquet copies of every published table | `data/processed/<dataset>/` |
| Great Expectations Data Docs | `reports/gx/uncommitted/data_docs/local_site/index.html` |
| Lineage graph and run summary | `reports/lineage.json`, `reports/run_summary.json` |
| Privacy audit log | `logs/audit/audit-<date>.jsonl` |

Set `ADMISSIO_PSEUDONYM_KEY` to keep tenant pseudonyms stable between runs. To see the flow graph, start `prefect server start` and run the flow with `PREFECT_API_URL=http://127.0.0.1:4200/api`.

## Run in Docker

```bash
docker build --tag admissio-pipeline .
docker run --rm \
  -v "$PWD/output/processed:/app/data/processed" \
  -v "$PWD/output/reports:/app/reports" \
  -v "$PWD/output/logs:/app/logs" \
  admissio-pipeline
```

Run the tests inside the image with `docker run --rm admissio-pipeline python -m pytest -q`.

## Reproduce the baseline

```bash
python -m src.models.baseline
```

This evaluates the carry-last-year's-yield-forward baseline on the 2024 cycle and prints its error, intake misses and fairness ratios.

## Roadmap

| Sprint | Course module | Deliverable |
|---|---|---|
| 1 | Module 2 | Project plan, architecture, repository, data dictionary, RAID log |
| 2 | Module 3 | Prefect pipeline, Great Expectations suites, bias detection, anonymisation, governance, audit log, Docker image |
| 3 | Module 4 | Demand and yield models, offer planner, SHAP and fairness audit, model card |
| 4 | Module 5 | Forecast API on Cloudflare Workers, React dashboard, stakeholder presentation |
| 5 | Final Project | Deployed, monitored release |

## Governance

Releases follow GitHub Flow. Models use Semantic Versioning in the MLflow registry. Every release is approved by the Ethics Review Panel.

## Data source

National Center for Education Statistics. Integrated Postsecondary Education Data System. U.S. Department of Education. https://nces.ed.gov/ipeds/
