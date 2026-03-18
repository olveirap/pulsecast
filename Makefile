.PHONY: ingest backfill features train export serve test

# ── Configuration ────────────────────────────────────────────────────────────
PYTHON   ?= python
UV       ?= uv
MONTHS   ?= 24
HORIZON  ?= 7
BACKFILL_START ?= $(shell date -d '18 months ago' +%Y-%m-%d 2>/dev/null || date -v-18m +%Y-%m-%d)
BACKFILL_END   ?= $(shell date +%Y-%m-%d)

# ── Data ingestion ────────────────────────────────────────────────────────────
ingest:
	$(PYTHON) -m data.ingest.tlc

# ── GTFS-RT backfill ─────────────────────────────────────────────────────────
backfill:
	$(PYTHON) -m data.ingest.gtfs_rt_backfill \
		--start $(BACKFILL_START) \
		--end   $(BACKFILL_END)

# ── Feature engineering ───────────────────────────────────────────────────────
features:
	$(PYTHON) -c "\
import polars as pl; \
from features.demand import build_demand_features; \
from features.calendar import build_calendar_features; \
from features.congestion import build_congestion_features; \
print('Feature modules imported successfully.')"

# ── Model training ────────────────────────────────────────────────────────────
train:
	$(PYTHON) -c "\
from models.baseline import BaselineForecaster; \
from models.lgbm import LGBMForecaster; \
print('Model modules imported successfully.')"

# ── ONNX export ───────────────────────────────────────────────────────────────
export:
	$(PYTHON) -c "\
from models.export import export_lgbm_to_onnx; \
print('Export module imported successfully.')"

# ── Serving (local dev) ───────────────────────────────────────────────────────
serve:
	uvicorn serving.main:app --reload --host 0.0.0.0 --port 8000

# ── Dashboard ─────────────────────────────────────────────────────────────────
dashboard:
	streamlit run dashboard/app.py

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

# ── Docker helpers ────────────────────────────────────────────────────────────
up:
	docker compose up --build -d

down:
	docker compose down -v

logs:
	docker compose logs -f

# ── Install dependencies ──────────────────────────────────────────────────────
install:
	$(UV) sync
