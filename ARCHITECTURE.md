# Architecture

## Overview

Pulsecast is a probabilistic shipment demand forecasting system that ingests
NYC TLC taxi trip records and live MTA congestion data,
engineers features, trains multiple forecasting models, and serves
probabilistic predictions (p10/p50/p90) via a low-latency REST API.

---

## Data Flow

```
TLC Parquet (nyc.gov)           NYC Bus Positions (S3)
      │                               │
      ▼                               ▼
data/ingest/tlc.py             data/ingest/bus_positions.py
      │  (hourly pickups)             │ (travel_time_var)
      ▼                               ▼
TimescaleDB ◄─────────────────────────┘
      │
      │          MTA Subway GTFS-RT (8 feeds)
      │                    │
      │          data/ingest/subway_rt.py
      │          (mean_delay per zone/hour)
      │                    │
      │◄───────────────────┘
      │
      ▼
Feature Engineering (Polars)
  ├─ features/demand.py     (lags, rolling, EWM, YoY)
  ├─ features/calendar.py   (dow, hour, holiday, event)
  └─ features/congestion.py (travel_time_var, flags)
      │
      ▼
Models
  ├─ models/baseline.py  (MSTL + AutoARIMA)
  ├─ models/lgbm.py      (LightGBM quantile regression)
  ├─ models/tft.py       (Temporal Fusion Transformer)
  └─ models/export.py    (ONNX export + parity check)
      │
      ▼
ONNX Runtime
      │
      ▼
FastAPI (serving/main.py)  ◄──── Redis cache ─── TimescaleDB
      │
      ▼
Client / Streamlit Dashboard (dashboard/app.py)
```

---

## Component Responsibilities

| Component | Responsibility |
|---|---|
| `data/ingest/tlc.py` | Download last 24 months of Yellow/Green Parquet, aggregate to hourly pickup counts per zone. |
| `data/ingest/bus_positions.py` | Poll/Read NYC bus positions from S3, spatial-join to TLC zones, compute travel time variance, upsert into `congestion` table. |
| `data/ingest/subway_rt.py` | Poll 8 MTA Subway GTFS-RT feeds every 60 s, compute `mean_delay` per zone/hour, upsert into `subway_delay` table. |
| `data/schema.sql` | Define TimescaleDB hypertables `demand`, `congestion`, and `subway_delay`. |
| `features/demand.py` | Lag features, rolling means, EWM trend, year-over-year ratio from TLC hourly counts. |
| `features/calendar.py` | Day-of-week, hour-of-day, week-of-year, days-to-next-US-holiday, NYC event flag. |
| `features/congestion.py` | travel_time_var lags, rolling 3-hour mean, disruption flag, low_confidence_flag. |
| `models/baseline.py` | MSTL decomposition + AutoARIMA per route via statsforecast. |
| `models/lgbm.py` | LightGBM quantile regression (q=0.1, 0.5, 0.9) with walk-forward CV. |
| `models/tft.py` | Temporal Fusion Transformer (pytorch-forecasting) for 7-day horizon. |
| `models/export.py` | Export fitted LightGBM models to ONNX with numerical parity validation. |
| `serving/main.py` | FastAPI app: fetch bus variance + subway delay → Redis cache → ONNX inference. |
| `serving/cache.py` | Redis cache with `travel_time_var` bucketed to nearest 10.0. |
| `serving/schemas.py` | Pydantic v2 request/response models. |
| `dashboard/app.py` | Streamlit UI: route selector, fan chart, ablation panel, calibration chart. |

---

## Infrastructure (docker-compose)

| Service | Image | Purpose |
|---|---|---|
| `timescaledb` | `timescale/timescaledb:latest-pg16` | Time-series database |
| `redis` | `redis:7-alpine` | Forecast cache |
| `mlflow` | `ghcr.io/mlflow/mlflow` | Experiment tracking |
| `gtfs-poller` | Custom build | MTA GTFS-RT + Bus positions ingestion loop |
| `api` | Custom build | FastAPI serving layer |
