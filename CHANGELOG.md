# Changelog

All notable changes to this project are documented here. The format follows Keep a Changelog, and model releases follow Semantic Versioning.

## [0.5.0] - 2026-10-25

### Added

- Offer planner that sizes offers so the upper 80% bound on intake meets capacity, with a 2024 back-test and a dashboard page.
- LIME and DiCE counterfactual tabs on the dashboard's forecast explanation page.
- Monitoring job for input drift, performance, fairness by group, interval coverage, alert false alarms, representation and planner overrun, with Evidently reports and a static monitoring dashboard.
- Hash-chained explanation store, `GET /explanations/{request_id}` and Prometheus metrics on the API.
- Retraining on every available cycle with a challenger alias, and promote and retire commands.
- Docker Compose stack with the pipeline, MLflow, monitoring, API, Prometheus and Grafana.
- Render deployment of the API, and workflows for deployment, scheduled monitoring with GitHub Pages and incident issues, and retraining.
- Ethical AI framework, algorithmic impact assessment, regulatory compliance report, public trust statement, monitoring plan, incident response plan and decommissioning plan.

### Changed

- Missing-input monitoring compares against the training cycle's share.
- The API image installs only the packages it needs.

## [0.4.0] - 2026-10-20

### Added

- Streamlit stakeholder dashboard with an overview, forecast explanations, what-if analysis, an ethical compliance view built on Fairlearn and a transparency page.
- Dashboard data export of the 2024 hold-out cycle with forecasts, intervals, alerts and SHAP values.
- API demo script that sends the presentation's sample institutions to `POST /predict`.
- Dashboard tests, Streamlit page smoke tests and a CI job that installs only the dashboard requirements.
- Streamlit Community Cloud configuration.

### Changed

- Module 5 roadmap delivers the Streamlit dashboard; the applicant demand forecast and offer planner move to the Final Project.

## [0.3.0] - 2026-10-17

### Added

- Yield forecaster that predicts the change from last cycle's yield, with a leakage guard on every feature.
- Time-aware split (train on the 2023 cycle, hold out 2024) with coarse random search and fine grid search over Ridge, random forest and LightGBM using GroupKFold by institution.
- MLflow tracking for every tuning candidate, mitigation and evaluation, and an `admissio-yield` registry entry with version tags and aliases.
- Group-reweighted and bounded-group-loss mitigation, group-calibrated 80% prediction intervals and a yield-drop alert.
- Fairness metrics for forecast error by group and for the alert (demographic parity, equalized odds, disparate impact ratio), with an equalized-odds threshold test.
- SHAP notebook, LIME explanations, DiCE counterfactual offer scenarios and sensitivity analysis.
- FastAPI service with `/predict` and `/health`, an API Dockerfile and CI jobs for training and the API image.
- Model card and fairness metrics report.

## [0.2.0] - 2026-10-16

### Added

- Prefect flow that runs ingestion, validation, cleaning, integration, bias detection, anonymisation, warehouse loading and retention end to end.
- Great Expectations suites for raw admissions, the directory, the yield panel and partner aggregates, with Data Docs.
- Representation bias suite built on Fairlearn.
- Small-cell suppression, partner anonymisation and a synthetic partner export for testing.
- Governance policy with role-based access and retention rules, and a JSON-lines privacy audit log.
- DuckDB warehouse, versioned Parquet outputs and a lineage graph.
- Dockerfile and a CI job that builds the image and runs the pipeline inside it.
- Governance framework and anonymisation plan documents.

### Changed

- Data validation moved from Pandera to Great Expectations.

## [0.1.0] - 2026-10-07

### Added

- Repository structure, Python environment and continuous integration.
- Raw IPEDS admissions files for 2022 to 2024 and the 2023 directory file.
- Ingestion, schema validation, panel features and the naive yield baseline.
- Group error audit for fairness monitoring.
- Data dictionary, data card, model card, architecture diagram and RAID log.
