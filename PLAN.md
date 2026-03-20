# Pulsecast — Build Plan

---

## Build Plan

**Data sources:** NYC TLC Trip Records (demand backbone) + MTA GTFS-Realtime (congestion covariate)
**Timeline:**  5 phases, each with a clear deliverable and success criteria

---

### Phase 0 — Data ingestion: TLC + GTFS-RT

- Download 24 months of Yellow + Green taxi Parquet files from `nyc.gov/tlc` (~8GB); filter to `pickup_datetime`, `PULocationID`, `trip_distance`, `fare_amount` — **duckdb, pandas**
- Aggregate to route-level demand: define 20 logical routes from TLC zone pairs (e.g. JFK→Midtown, BK→Manhattan) mirroring a courier network topology — **geopandas, TLC zone shapefile**
- Set up GTFS-RT ingestion: poll MTA Subway + Bus trip updates feed every 60s, parse protobuf, compute per-zone `delay_index` = mean(arrival_delay) weighted by trip count — **gtfs-realtime-bindings, protobuf, apscheduler**
- Store demand timeseries in TimescaleDB hypertable; store `delay_index` in a separate hypertable; join on (zone, hour) at feature time — **timescaledb, docker compose**

**Deliverable:** 20-route demand table + hourly `delay_index` covariate, both in TimescaleDB. The GTFS-RT poller runs as a background service and backfills the last 30 days from MTA historical archives.

> MTA GTFS-RT feeds require a free API key from mta.info. Historical trip update archives cover ~18 months and let you backfill `delay_index` without live polling during dev.

---

### Phase 1 — Feature engineering pipeline

- Demand lags from TLC: `volume(t-1…t-7)`, rolling mean 7d/14d, EWM trend, yoy ratio (`volume_t / volume_{t-52w}`) — **polars**
- Calendar features: dow, hour_of_day, week_of_year, days_to_next_US_holiday, `is_nyc_event` flag (scraped from NYC Open Data events calendar) — **holidays, nyc open data api**
- GTFS-RT covariate: lag-1 `delay_index`, rolling 3h `delay_index`, binary `disruption_flag` (`delay_index > 2σ` from route mean) — **polars**
- Materialize as Parquet snapshots + register in Feast local feature store; version by training cutoff date — **feast (local), parquet**

**Deliverable:** Feature store with two source lineages clearly separated: TLC demand features vs GTFS-RT congestion features. Ablation later will show each source's marginal contribution.

---

### Phase 2 — Modeling layer: baselines → TFT 

- Baseline: MSTL + AutoARIMA per route via `statsforecast`; establishes the floor and quantifies how much the GTFS-RT covariate actually helps — **statsforecast**
- LightGBM quantile regression (q=0.1/0.5/0.9): train twice — with and without `delay_index` — to produce a covariate ablation table — **lightgbm, sktime, optuna**
- TFT (7-day horizon): TLC demand as target, GTFS-RT `delay_index` as known future covariate (available at inference time from real feed); quantile outputs P10/P50/P90 — **pytorch-forecasting**
- Log all runs to MLflow: pinball loss per quantile, calibration coverage, RMSE. Flag the ablation result explicitly as a tracked experiment — **mlflow**

**Deliverable:** Ablation table: MSTL vs LightGBM (no covariate) vs LightGBM+delay vs TFT+delay. The GTFS-RT marginal gain during disruption periods is the headline portfolio insight.

---

### Phase 3 — Serving layer: ONNX + FastAPI

- Export LightGBM to ONNX; validate numerical parity; benchmark CPU latency with ONNX Runtime vs native — **onnxruntime, onnxmltools**
- FastAPI: `POST /forecast` — accepts `route_id` + `horizon`, fetches live `delay_index` from GTFS-RT poller, returns P10/P50/P90 + `X-Latency-Ms` header — **fastapi, pydantic v2**
- Redis: cache per `(route_id, horizon, delay_index_bucket)` with TTL=30min; bucket `delay_index` to nearest 0.5 to maximise hit rate without staling on live signal — **redis**
- Docker Compose: API + GTFS-RT poller + Redis + TimescaleDB + MLflow UI — full stack in one command — **docker compose**

**Deliverable:** Live inference endpoint that consumes a real GTFS-RT feed at prediction time — not just at training time. p99 <50ms CPU target.

---

### Phase 4 — Demo UI + portfolio packaging

- Streamlit dashboard: route selector, 7-day fan chart (P10/P50/P90 band), actual vs forecast overlay, live `delay_index` badge, latency indicator — **streamlit, plotly**
- Ablation panel: side-by-side pinball loss bars for the four model variants — key visual for interviewers — **plotly**
- Calibration chart (Evidently): expected vs actual quantile coverage — shows the model is honest, not just accurate — **evidently**
- Repo: `ARCHITECTURE.md` (data lineage diagram, design decisions), `DECISIONS.md` (why TLC, why GTFS-RT as covariate, why ONNX, why `delay_index` bucketing), clean module structure — **git**
- Deploy to Fly.io with GitHub Actions retraining DAG (weekly, triggered on data freshness check) — **fly.io, github actions**

**Deliverable:** Public URL + repo. The live GTFS-RT feed means the demo behaves differently at rush hour vs midnight — interviewers who click around during different times see real variance.

---

