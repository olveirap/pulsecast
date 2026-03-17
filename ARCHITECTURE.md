# Architecture

## Overview

Pulsecast is a probabilistic shipment demand forecasting system that ingests
NYC TLC taxi trip records and live MTA GTFS-Realtime congestion data,
engineers features, trains multiple forecasting models, and serves
probabilistic predictions (p10/p50/p90) via a low-latency REST API.

---

## Data Flow

```
TLC Parquet (nyc.gov)
      │
      ▼
data/ingest/tlc.py
      │  (hourly pickup counts per zone)
      ▼
TimescaleDB ─────────────────────────────────────────────┐
      │                                                   │
      │          MTA GTFS-RT feed (api.mta.info)          │
      │                    │                              │
      │          data/ingest/gtfs_rt.py                   │
      │          (delay_index per zone/hour)              │
      │                    │                              │
      │◄───────────────────┘                              │
      │                                                   │
      ▼                                                   │
Feature Store (Feast)                                     │
  ├─ features/demand.py   (lags, rolling, EWM, YoY)      │
  ├─ features/calendar.py (dow, hour, holiday, event)     │
  └─ features/congestion.py (lag-1, rolling-3h, flag)    │
      │                                                   │
      ▼                                                   │
Models                                                    │
  ├─ models/baseline.py  (MSTL + AutoARIMA)              │
  ├─ models/lgbm.py      (LightGBM quantile regression)  │
  ├─ models/tft.py       (Temporal Fusion Transformer)   │
  └─ models/export.py    (ONNX export + parity check)    │
      │                                                   │
      ▼                                                   │
ONNX Runtime                                              │
      │                                                   │
      ▼                                                   │
FastAPI (serving/main.py)  ◄──── Redis cache ─────────────┘
      │
      ▼
Client / Streamlit Dashboard (dashboard/app.py)
```

---

## Component Responsibilities

| Component | Responsibility |
|---|---|
| `data/ingest/tlc.py` | Download last 24 months of Yellow/Green Parquet, aggregate to hourly pickup counts per zone. |
| `data/ingest/gtfs_rt.py` | Poll MTA GTFS-RT every 60 s, compute `delay_index` (mean weighted arrival delay) per zone/hour, upsert into TimescaleDB. |
| `data/schema.sql` | Define TimescaleDB hypertables `demand` and `delay_index`; configure compression and retention. |
| `features/demand.py` | Lag features, rolling means, EWM trend, year-over-year ratio from TLC hourly counts. |
| `features/calendar.py` | Day-of-week, hour-of-day, week-of-year, days-to-next-US-holiday, NYC event flag. |
| `features/congestion.py` | lag-1 `delay_index`, rolling 3-hour mean, disruption flag (>2σ over trailing 168 h). |
| `models/baseline.py` | MSTL decomposition + AutoARIMA per route via statsforecast. |
| `models/lgbm.py` | LightGBM quantile regression (q=0.1, 0.5, 0.9) with walk-forward CV. |
| `models/tft.py` | Temporal Fusion Transformer (pytorch-forecasting) for 7-day horizon. |
| `models/export.py` | Export fitted LightGBM models to ONNX with numerical parity validation. |
| `serving/main.py` | FastAPI app: fetch delay_index → Redis cache → ONNX inference → return p10/p50/p90 with `X-Latency-Ms` header. |
| `serving/cache.py` | Redis cache with `delay_index` bucketed to nearest 0.5. |
| `serving/schemas.py` | Pydantic v2 request/response models. |
| `dashboard/app.py` | Streamlit UI: route selector, fan chart, ablation panel, calibration chart. |

---

## Infrastructure (docker-compose)

| Service | Image | Purpose |
|---|---|---|
| `timescaledb` | `timescale/timescaledb:latest-pg16` | Time-series database |
| `redis` | `redis:7-alpine` | Forecast cache |
| `mlflow` | `ghcr.io/mlflow/mlflow` | Experiment tracking |
| `gtfs-poller` | Custom build | MTA GTFS-RT ingestion loop |
| `api` | Custom build | FastAPI serving layer |
