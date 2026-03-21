# Pulsecast: Probabilistic Demand Forecasting

Pulsecast is a high-performance, probabilistic shipment demand forecasting system designed for NYC TLC taxi trip records, augmented with live MTA GTFS-Realtime congestion signals. It provides p10/p50/p90 hourly demand forecasts for a 1–7 day horizon, served via a low-latency FastAPI endpoint.

## Project Overview

### Core Technologies
- **Languages:** Python 3.12+
- **Data/Storage:** TimescaleDB (PostgreSQL), Redis (Caching), Polars/Pandas
- **ML/Models:** LightGBM (Quantile Regression), Temporal Fusion Transformer (PyTorch Forecasting), ONNX Runtime (Inference)
- **API/Serving:** FastAPI, Uvicorn, Pydantic v2
- **Dashboard:** Streamlit, Plotly
- **Infrastructure:** Docker, Docker Compose, MLflow (Experiment Tracking)

### Architecture
1.  **Ingestion:** Scrapes NYC TLC Parquet files and polls MTA GTFS-RT feeds for real-time `delay_index`.
2.  **Feature Engineering:** Computes demand lags, rolling averages, calendar features, and congestion signals.
3.  **Modeling:** Trains LightGBM and TFT models to predict demand quantiles.
4.  **Export:** Converts models to ONNX format for efficient, framework-agnostic serving.
5.  **Serving:** A FastAPI layer fetches features, performs ONNX inference, and caches results in Redis.
6.  **Monitoring:** MLflow for tracking and Streamlit for visual inspection of forecasts and ablation studies.

---

## Building and Running

The project relies on a `Makefile` for most operations.

### Environment Setup
1.  **Dependencies:** Ensure Python 3.12 and `poetry` are installed.
    ```bash
    make install
    ```
2.  **Infrastructure:** Start the core services (TimescaleDB, Redis, MLflow).
    ```bash
    make up
    ```

### Data Pipeline
- **TLC Ingest:** `make ingest` (Downloads and aggregates NYC taxi data).
- **GTFS Backfill:** `make backfill` (Backfills historical congestion data from S3).
- **Stop Mapping:** `make build-stop-zone-map` (Maps GTFS stops to TLC zones).

### Development Workflow
- **Features:** `make features` (Verifies feature engineering modules).
- **Training:** `make train` (Trains forecasting models).
- **Export:** `make export` (Exports models to ONNX).
- **Serving:** `make serve` (Starts FastAPI at `http://localhost:8000`).
- **Dashboard:** `make dashboard` (Starts Streamlit at `http://localhost:8501`).
- **Testing:** `make test` (Runs `pytest`).

---

## Development Conventions

### Coding Standards
- **Linter:** `ruff` (configured in `pyproject.toml`, line length 100).
- **Type Checking:** `mypy` (strict mode is currently off, but recommended for new modules).
- **Format:** Adheres to modern Python practices (Pydantic v2, type hints, `pathlib`).

### Testing Practices
- Tests are located in the `tests/` directory.
- Use `pytest` for all functional and unit tests.
- **TDD:** When adding features or fixing bugs, follow test-driven development.

### Contribution Guidelines
- **Gitflow:** Follow Gitflow for branching and releases.
- **Commits:** Write concise, descriptive commit messages focusing on "why" rather than just "what".
- **Documentation:** Keep `ARCHITECTURE.md` and `DECISIONS.md` (ADRs) updated as the system evolves.
