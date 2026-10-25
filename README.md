# Admissio Yield Forecaster

Institution-level forecasting of applications and enrolment yield for admissions teams on the Admissio platform. Partner institutions use the forecasts to set offer volumes that fill places without overcrowding.

Built on public IPEDS admissions data (2022 to 2024) while partner data accumulates. Forecasts advise people; they never decide admissions.

| Live service | URL |
| --- | --- |
| Stakeholder dashboard | https://msc-admissio-yield-forecaster.streamlit.app |
| Forecast API (`POST /predict`, docs at `/docs`) | https://admissio-yield-api.onrender.com |
| Monitoring dashboard | https://adesege.github.io/msc-admissio-yield-forecaster/ |

The model is version 0.3.0, registered as a candidate. It has not been approved for release because it misses the 4.8-point error gate and the fairness limit for small and for-profit institutions.

## Repository layout

```
msc-admissio-yield-forecaster/
├── api/           # FastAPI service: /predict, /explanations, /health, /metrics and the explanation store
├── data/          # raw IPEDS files; interim and processed outputs are written at run time
├── deploy/        # Prometheus and Grafana configuration
├── docs/          # model card, fairness report, ethical AI framework, monitoring, incident and decommissioning plans
├── mlruns/        # MLflow tracking store and model registry (SQLite plus artifacts)
├── models/        # serialised forecaster, SHAP background sample and metadata
├── notebooks/     # SHAP analysis notebook
├── scripts/       # API demo and hold-out replay
├── src/
│   ├── ingest/      # landing with checksums, IPEDS readers
│   ├── cleaning/    # deduplication, quarantine, reserved-code handling
│   ├── features/    # yield panel and fairness groups
│   ├── validation/  # Great Expectations suites and checkpoint runner
│   ├── fairness/    # representation bias, group error audit, alert fairness and threshold mitigation
│   ├── privacy/     # small-cell suppression, anonymisation, audit log
│   ├── governance/  # access policy and retention rules
│   ├── pipeline/    # Prefect flow, DuckDB warehouse, lineage
│   ├── models/      # features, tuning, mitigation, forecaster, training, retraining, registry and lifecycle
│   ├── explain/     # SHAP, LIME, counterfactuals, sensitivity analysis and figures
│   ├── planning/    # offer planner and its back-test
│   └── monitoring/  # drift, performance and fairness checks, Evidently reports, monitoring site
├── dashboard/     # Streamlit stakeholder dashboard, its data export and slim requirements
├── tests/         # pytest suite
├── .streamlit/    # dashboard theme
├── Dockerfile          # pipeline image
├── Dockerfile.api      # API image
├── Dockerfile.monitor  # monitoring image
├── docker-compose.yml  # pipeline, MLflow, monitoring, API, Prometheus and Grafana
├── render.yaml         # Render web service for the API
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

The response holds the forecast yield, its 80% interval, expected enrolment, the yield-drop alert, the fairness group, the three largest SHAP drivers and a `request_id`. `GET /explanations/{request_id}` returns the stored record of that forecast and its explanation, `GET /health` reports the model version and release status, and `GET /metrics/` exposes Prometheus metrics. Build the container with `docker build --file Dockerfile.api --tag admissio-api .`.

With the API running, send the three sample institutions from the stakeholder presentation and print each forecast as a short summary:

```bash
python -m scripts.api_demo
```

## Explore the dashboard

Live dashboard: https://msc-admissio-yield-forecaster.streamlit.app, deployed on Streamlit Community Cloud from dashboard/app.py.

The Streamlit dashboard presents the forecaster to admissions leaders and the Ethics Review Panel. It has six pages: an overview of results against the last-year baseline, forecast explanations for any 2024 test institution with SHAP drivers, a LIME check and DiCE counterfactual offer volumes, what-if analysis of offers, applications and test policy, an offer planner that sizes offers against the places available, an ethical compliance view that recomputes the fairness metrics with Fairlearn, and the model's transparency statement. Every forecast goes through the same `forecast()` function that serves `POST /predict`.

```bash
streamlit run dashboard/app.py
```

The dashboard reads `dashboard/data/`, a snapshot of the 2024 hold-out cycle with forecasts, intervals, alerts and SHAP values. Refresh it after retraining:

```bash
python -m dashboard.build_data
```

## Monitor, retrain and retire

Monitoring uses its own environment because Evidently pins an older Plotly than the dashboard.

```bash
python -m venv .venv-monitor
.venv-monitor/bin/pip install -r requirements-monitoring.txt
.venv-monitor/bin/python -m src.monitoring.run          # checks against the model card triggers
.venv-monitor/bin/python -m src.monitoring.site         # builds the monitoring dashboard in site/
```

The run compares the 2023 reference cycle with the 2024 current cycle: population stability index and Evidently drift tests for every input, Evidently regression performance, Fairlearn error ratios, interval coverage and alert false alarms by group, representation from the pipeline's bias report and the offer planner's overrun rate. Each signal is green, amber or red; thresholds and responses are in `docs/monitoring_plan.md`.

```bash
python -m src.models.retrain                                         # challenger on every available cycle
python -m src.models.lifecycle promote --approval-reference ERP-...  # after Ethics Review Panel approval
python -m src.models.lifecycle retire --reason ... --approval-reference ERP-...
```

## Run the whole system with Docker Compose

```bash
docker compose up --build
docker compose --profile demo up replay   # send 500 hold-out institutions through the API
```

| Service | Port | Purpose |
| --- | --- | --- |
| `pipeline` | | Runs the Prefect pipeline once and exits |
| `monitor` | 8080 | Runs the monitoring checks, then serves the monitoring dashboard |
| `api` | 8000 | Forecast API with the explanation store |
| `mlflow` | 5000 | MLflow experiments and model registry |
| `prometheus` | 9090 | Scrapes the API's metrics |
| `grafana` | 3000 | Live dashboard of forecasts, alerts and monitoring status by group |

## Continuous integration and deployment

| Workflow | Trigger | What it does |
| --- | --- | --- |
| `ci` | Push and pull request | Tests, quick training run, pipeline image, dashboard tests, monitoring run, API image with explanation and metrics checks, full Docker Compose stack |
| `deploy-api` | Successful `ci` on `main`, or on demand | Deploys the API image to Render and waits for `/health` |
| `monitoring` | Weekly, on push to `main` and on demand | Runs the pipeline and monitoring, publishes the dashboard to GitHub Pages, opens or updates an `incident` issue on red signals, and calls `retrain` on red drift or error |
| `retrain` | On demand or from `monitoring` | Trains and registers a challenger and uploads it with the MLflow store |

The API runs on a free Render web service defined in `render.yaml`. The deploy workflow needs a `RENDER_DEPLOY_HOOK_URL` repository secret and an `API_URL` repository variable such as `https://admissio-yield-api.onrender.com`. Free instances sleep after 15 minutes without traffic, so the first request after a pause takes about a minute, and the explanation store starts empty after each restart. GitHub Pages must use GitHub Actions as its source.

## Roadmap

| Sprint | Course module | Deliverable                                                                                                      |
| ------ | ------------- | ---------------------------------------------------------------------------------------------------------------- |
| 1      | Module 2      | Project plan, architecture, repository, data dictionary, RAID log                                                |
| 2      | Module 3      | Prefect pipeline, Great Expectations suites, bias detection, anonymisation, governance, audit log, Docker image  |
| 3      | Module 4      | Yield forecaster, MLflow registry, SHAP, LIME, counterfactual and fairness analysis, model card, FastAPI service |
| 4      | Module 5      | Streamlit stakeholder dashboard with ethical compliance view, API demo, stakeholder presentation                 |
| 5      | Final Project | Offer planner, monitoring and retraining, explanation store, Docker Compose, CI/CD, ethical AI framework         |
| 6      | Partner pilot | Shadow pilot with Nigerian partners, applicant demand forecast, partner fairness groups                         |

## Governance

Releases follow GitHub Flow. Models use Semantic Versioning in the MLflow registry: a retrain on new data raises the patch version. Every release is approved by the Ethics Review Panel. The ethical AI framework, algorithmic impact assessment, regulatory compliance report, public trust statement, incident response plan and decommissioning plan are in `docs/`.

## Data source

National Center for Education Statistics. Integrated Postsecondary Education Data System. U.S. Department of Education. https://nces.ed.gov/ipeds/
