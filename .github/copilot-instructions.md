# Pulsecast: Probabilistic Demand Forecasting

Pulsecast is a high-performance, probabilistic shipment demand forecasting system designed for NYC TLC taxi trip records, augmented with live MTA GTFS-Realtime congestion signals. It provides p10/p50/p90 hourly demand forecasts for a 1–7 day horizon, served via a low-latency FastAPI endpoint.

## Project Overview

### Core Technologies
- **Languages:** Python 3.11+
- **Data/Storage:** TimescaleDB (PostgreSQL), Redis (Caching), Polars/Pandas, Feast (Feature Store)
- **ML/Models:** LightGBM (Quantile Regression), Temporal Fusion Transformer (PyTorch Forecasting), ONNX Runtime (Inference)
- **API/Serving:** FastAPI, Uvicorn, Pydantic v2
- **Dashboard:** Streamlit, Plotly
- **Infrastructure:** Docker, Docker Compose, MLflow (Experiment Tracking)

### Architecture
1. **Ingestion:** Scrapes NYC TLC Parquet files and polls MTA GTFS-RT feeds for real-time `delay_index`.
2. **Feature Engineering:** Computes demand lags, rolling averages, calendar features, and congestion signals.
3. **Modeling:** Trains LightGBM and TFT models to predict demand quantiles.
4. **Export:** Converts models to ONNX format for efficient, framework-agnostic serving.
5. **Serving:** A FastAPI layer fetches features, performs ONNX inference, and caches results in Redis.
6. **Monitoring:** MLflow for tracking and Streamlit for visual inspection of forecasts and ablation studies.

### Repository Layout

```
pulsecast/
├── data/
│   ├── ingest/
│   │   ├── tlc.py              # Downloads TLC Yellow/Green Parquet files
│   │   ├── gtfs_rt.py          # Polls MTA GTFS-RT, computes delay_index
│   │   └── gtfs_rt_backfill.py # Backfills delay_index from S3 archives
│   └── schema.sql              # TimescaleDB hypertable definitions
├── features/
│   ├── demand.py           # Lags, rolling means, EWM trend, YoY ratio
│   ├── calendar.py         # dow, hour, week, holiday, event flag
│   └── congestion.py       # lag-1 delay_index, rolling-3h, disruption_flag
├── models/
│   ├── baseline.py         # MSTL + AutoARIMA (statsforecast)
│   ├── lgbm.py             # LightGBM quantile regression + CV
│   ├── tft.py              # Temporal Fusion Transformer (pytorch-forecasting)
│   └── export.py           # ONNX export with parity validation
├── serving/
│   ├── main.py             # FastAPI POST /forecast
│   ├── cache.py            # Redis cache (delay_index bucketing)
│   └── schemas.py          # Pydantic v2 models
└── dashboard/
    └── app.py              # Streamlit fan chart + ablation panel
```

All Python modules are under the `pulsecast/` package. Import with the `pulsecast.*` prefix (e.g., `from pulsecast.serving.cache import ForecastCache`).

---

## Building and Running

The project relies on a `Makefile` for most operations.

### Environment Setup
1. **Dependencies:** Ensure Python 3.11 and `uv` are installed.
   ```bash
   make install   # runs: uv sync
   ```
2. **Infrastructure:** Start the core services (TimescaleDB, Redis, MLflow).
   ```bash
   make up        # runs: docker compose up --build -d
   ```

### Data Pipeline
- **TLC Ingest:** `make ingest` — downloads the last 24 months of Yellow and Green taxi Parquet files from NYC TLC and aggregates them to hourly pickup counts per zone.
- **GTFS Backfill:** `make backfill` — backfills historical congestion data from S3.
- **Stop Mapping:** `make build-stop-zone-map` — maps GTFS stops to TLC zones.

### Development Workflow
- **Features:** `make features` — verifies feature engineering modules.
- **Training:** `make train` — trains forecasting models.
- **Export:** `make export` — exports models to ONNX.
- **Serving:** `make serve` — starts FastAPI at `http://localhost:8000`.
- **Dashboard:** `make dashboard` — starts Streamlit at `http://localhost:8501`.
- **Testing:** `make test` — runs `pytest tests/ -v --tb=short`.

---

## Development Conventions

### Coding Standards
- **Linter:** `ruff` (configured in `pyproject.toml`; line length 100, target py311, select E/F/I/UP, ignore E501). Run: `ruff check .`
- **Type Checking:** `mypy` (strict mode off; `ignore_missing_imports = true`). Run: `mypy .`
- **Formatting:** Modern Python practices — Pydantic v2, type hints throughout, `pathlib` for filesystem operations.
- **String quoting:** Follow the existing file's style; no strict preference enforced by tooling.

### Module and Import Conventions
- All source code lives under `pulsecast/` and must be imported as `pulsecast.<submodule>`.
- Use `polars` (not `pandas`) for tabular data except where `geopandas` or interoperability requires pandas.
- Use Pydantic v2 models for all API request/response schemas.
- Redis `ForecastCache` uses `decode_responses=True`; cache `get()` returns `str`, not `bytes`.
- ONNX models use `FloatTensorType([None, n_features])` to keep the batch dimension dynamic.
- `delay_index` is bucketed to the nearest 0.5 before constructing Redis cache keys (see ADR-003).

### Testing Practices
- Tests live in `tests/` and are discovered automatically by pytest.
- Use `pytest` for all unit and functional tests; async tests use `asyncio_mode = "auto"`.
- **TDD:** Write or update tests before (or alongside) implementing new features or bug fixes.
- Do not remove or weaken existing tests.

### Architecture Decision Records
Key design decisions are documented in `DECISIONS.md`:
- **ADR-001:** MTA GTFS-RT used as the primary congestion covariate (`delay_index`).
- **ADR-002:** LightGBM quantile models exported to ONNX for low-latency, portable serving.
- **ADR-003:** `delay_index` bucketed to nearest 0.5 in Redis cache keys to maximise hit rate.

When making changes that affect these decisions, update `DECISIONS.md` accordingly.

### Contribution Guidelines
- **Branching:** Follow Gitflow (feature branches off `develop`, releases off `main`).
- **Commits:** Write concise, descriptive commit messages that explain *why*, not just *what*.
- **Documentation:** Keep `ARCHITECTURE.md` and `DECISIONS.md` updated as the system evolves.
- **Dependencies:** Add new dependencies to `pyproject.toml` via `uv add <package>`; do not edit the lockfile manually.
