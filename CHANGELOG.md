# Changelog

All notable changes to this project are documented here. The format follows Keep a Changelog, and model releases follow Semantic Versioning.

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
