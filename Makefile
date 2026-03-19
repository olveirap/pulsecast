.PHONY: ingest backfill build-stop-zone-map features train export serve test

# ── Configuration ────────────────────────────────────────────────────────────
PYTHON   ?= python
UV       ?= uv
MONTHS   ?= 24
HORIZON  ?= 7
BACKFILL_START ?= $(shell date -d '18 months ago' +%Y-%m-%d 2>/dev/null || date -v-18m +%Y-%m-%d)
BACKFILL_END   ?= $(shell date +%Y-%m-%d)

# ── Data ingestion ────────────────────────────────────────────────────────────
ingest:
	$(PYTHON) -m pulsecast.data.ingest.tlc

# ── GTFS-RT backfill ─────────────────────────────────────────────────────────
backfill:
	$(PYTHON) -m pulsecast.data.ingest.gtfs_rt_backfill \
		--start $(BACKFILL_START) \
		--end   $(BACKFILL_END)

build-stop-zone-map:
	$(PYTHON) scripts/build_stop_zone_map.py

# ── Feature engineering ───────────────────────────────────────────────────────
features:
	$(PYTHON) -c "\
import polars as pl; \
from pulsecast.features.demand import build_demand_features; \
from pulsecast.features.calendar import build_calendar_features; \
from pulsecast.features.congestion import build_congestion_features; \
print('Feature modules imported successfully.')"

# ── Model training ────────────────────────────────────────────────────────────
train:
	$(PYTHON) -c "\
from pulsecast.models.baseline import BaselineForecaster; \
from pulsecast.models.lgbm import LGBMForecaster; \
print('Model modules imported successfully.')"

# ── ONNX export ───────────────────────────────────────────────────────────────
export:
	$(PYTHON) -c "\
from pulsecast.models.export import export_lgbm_to_onnx; \
print('Export module imported successfully.')"

# ── Serving (local dev) ───────────────────────────────────────────────────────
serve:
	uvicorn pulsecast.serving.main:app --reload --host 0.0.0.0 --port 8000

# ── Dashboard ─────────────────────────────────────────────────────────────────
dashboard:
	streamlit run pulsecast/dashboard/app.py

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
