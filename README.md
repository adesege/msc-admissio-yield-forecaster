# Admissio Yield Forecaster

Institution-level forecasting of applications and enrolment yield for admissions teams on the Admissio platform. Partner institutions use the forecasts to set offer volumes that fill places without overcrowding.

Built on public IPEDS admissions data (2022 to 2024) while partner data accumulates. Forecasts advise people; they never decide admissions.

## Repository layout

```
msc-admissio-yield-forecaster/
├── api/           # FastAPI service with the /predict endpoint
├── data/          # raw IPEDS files; interim and processed outputs are written at run time
├── docs/          # model card, fairness report, data card, data dictionary, governance, figures, screenshots
├── mlruns/        # MLflow tracking store and model registry (SQLite plus artifacts)
├── models/        # serialised forecaster, SHAP background sample and metadata
├── notebooks/     # SHAP analysis notebook
├── scripts/       # API demo
├── src/
│   ├── ingest/      # landing with checksums, IPEDS readers
│   ├── cleaning/    # deduplication, quarantine, reserved-code handling
│   ├── features/    # yield panel and fairness groups
│   ├── validation/  # Great Expectations suites and checkpoint runner
│   ├── fairness/    # representation bias, group error audit, alert fairness and threshold mitigation
│   ├── privacy/     # small-cell suppression, anonymisation, audit log
│   ├── governance/  # access policy and retention rules
│   ├── pipeline/    # Prefect flow, DuckDB warehouse, lineage
│   ├── models/      # features, tuning, mitigation, forecaster, training and MLflow registry
│   └── explain/     # SHAP, LIME, counterfactuals, sensitivity analysis and figures
├── dashboard/     # Streamlit stakeholder dashboard, its data export and slim requirements
├── tests/         # pytest suite
├── .streamlit/    # dashboard theme
├── Dockerfile     # pipeline image
├── Dockerfile.api # API image
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

| Output                                                                            | Location                                                 |
| --------------------------------------------------------------------------------- | -------------------------------------------------------- |
| Warehouse with `analytics.yield_panel` and the `analytics.model_ready_panel` view | `data/processed/warehouse.duckdb`                        |
| Versioned Parquet copies of every published table                                 | `data/processed/<dataset>/`                              |
| Great Expectations Data Docs                                                      | `reports/gx/uncommitted/data_docs/local_site/index.html` |
| Lineage graph and run summary                                                     | `reports/lineage.json`, `reports/run_summary.json`       |
| Privacy audit log                                                                 | `logs/audit/audit-<date>.jsonl`                          |

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

Run the tests inside the image with `docker run --rm admissio-pipeline python -m pytest -q --ignore-glob='tests/test_dashboard*'`.

## Reproduce the baseline

```bash
python -m src.models.baseline
```

This evaluates the carry-last-year's-yield-forward baseline on the 2024 cycle and prints its error, intake misses and fairness ratios.

## Train, explain and audit the model

Run the pipeline first so `analytics.model_ready_panel` exists, then:

```bash
python -m src.models.train      # tune Ridge, random forest and LightGBM, mitigate, evaluate, register
python -m src.fairness.report   # yield-drop alert fairness and equalized-odds threshold test
python -m src.explain.run       # LIME, counterfactuals, sensitivity analysis and report figures
```

Training uses the 2023 cycle with 5-fold GroupKFold by institution and holds out the 2024 cycle. Every candidate, metric and artifact is logged to MLflow, and the selected model is registered as `admissio-yield`. A version receives the `champion` alias only when it passes every acceptance check; otherwise it carries `candidate`. Add `--quick` to `train` for a short smoke run.

Browse the experiment and registry with:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Open `notebooks/shap_analysis.ipynb` for the SHAP analysis. The model card and fairness report are in `docs/`.

## Serve forecasts

```bash
uvicorn api.main:app --port 8000
```

```bash
curl -X POST 127.0.0.1:8000/predict -H "content-type: application/json" -d '{
  "institution_id": "163453", "applicants": 23366, "offers": 19214,
  "last_cycle_applicants": 22741, "last_cycle_offers": 18865, "last_cycle_yield": 0.121,
  "test_policy": 5, "last_cycle_test_policy": 5, "control": 1, "size_band": 3,
  "locale": 11, "region": 2, "carnegie_class": 16, "hbcu": true
}'
```

The response holds the forecast yield, its 80% interval, expected enrolment, the yield-drop alert, the fairness group and the three largest SHAP drivers. `GET /health` reports the model version and release status. Build the container with `docker build --file Dockerfile.api --tag admissio-api .`.

With the API running, send the three sample institutions from the stakeholder presentation and print each forecast in plain language:

```bash
python -m scripts.api_demo
```

## Explore the dashboard

Live dashboard: https://msc-admissio-yield-forecaster.streamlit.app, deployed on Streamlit Community Cloud from dashboard/app.py.

The Streamlit dashboard presents the forecaster to admissions leaders and the Ethics Review Panel. It has five pages: an overview of results against the last-year baseline, forecast explanations for any 2024 test institution, what-if analysis of offers, applications and test policy, an ethical compliance view that recomputes the fairness metrics with Fairlearn, and the model's transparency statement. Every forecast goes through the same `forecast()` function that serves `POST /predict`.

```bash
streamlit run dashboard/app.py
```

The dashboard reads `dashboard/data/`, a snapshot of the 2024 hold-out cycle with forecasts, intervals, alerts and SHAP values. Refresh it after retraining:

```bash
python -m dashboard.build_data
```

## Roadmap

| Sprint | Course module | Deliverable                                                                                                      |
| ------ | ------------- | ---------------------------------------------------------------------------------------------------------------- |
| 1      | Module 2      | Project plan, architecture, repository, data dictionary, RAID log                                                |
| 2      | Module 3      | Prefect pipeline, Great Expectations suites, bias detection, anonymisation, governance, audit log, Docker image  |
| 3      | Module 4      | Yield forecaster, MLflow registry, SHAP, LIME, counterfactual and fairness analysis, model card, FastAPI service |
| 4      | Module 5      | Streamlit stakeholder dashboard with ethical compliance view, API demo, stakeholder presentation                 |
| 5      | Final Project | Applicant demand forecast, offer planner, deployed and monitored release                                         |

## Governance

Releases follow GitHub Flow. Models use Semantic Versioning in the MLflow registry. Every release is approved by the Ethics Review Panel.

## Data source

National Center for Education Statistics. Integrated Postsecondary Education Data System. U.S. Department of Education. https://nces.ed.gov/ipeds/
