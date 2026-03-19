.PHONY: ingest ingest-tlc ingest-subway ingest-bus backfill build-zone-maps features train export serve test

# ── Configuration ────────────────────────────────────────────────────────────
PYTHON   ?= python
POETRY   ?= poetry
MONTHS   ?= 24
HORIZON  ?= 7
BACKFILL_START ?= $(shell date -d '18 months ago' +%Y-%m-%d 2>/dev/null || date -v-18m +%Y-%m-%d)
BACKFILL_END   ?= $(shell date +%Y-%m-%d)

# ── Data ingestion ────────────────────────────────────────────────────────────
ingest: ingest-tlc ingest-subway ingest-bus

ingest-tlc:
	$(POETRY) run python -m pulsecast.data.ingest.tlc

ingest-subway:
	$(POETRY) run python -m pulsecast.data.ingest.subway_rt

ingest-bus:
	$(POETRY) run python -m pulsecast.data.ingest.bus_positions

# ── Backfill ─────────────────────────────────────────────────────────────────
backfill:
	$(POETRY) run python -m pulsecast.data.ingest.bus_positions_backfill \
		--start $(BACKFILL_START) \
		--end   $(BACKFILL_END)

# ── Spatial Mappings ──────────────────────────────────────────────────────────
# NOTE: requires [geo] extras (geopandas, shapely, etc.)
build-zone-maps:
	$(POETRY) run python scripts/build_bus_zone_map.py
	$(POETRY) run python scripts/build_subway_zone_map.py

# ── Feature engineering ───────────────────────────────────────────────────────
features:
	$(POETRY) run python scripts/run_features.py

# ── Model training ────────────────────────────────────────────────────────────
train:
	$(POETRY) run python scripts/run_train.py

# ── ONNX export ───────────────────────────────────────────────────────────────
export:
	$(POETRY) run python scripts/run_export.py

# ── Serving (local dev) ───────────────────────────────────────────────────────
serve:
	$(POETRY) run uvicorn pulsecast.serving.main:app --reload --host 0.0.0.0 --port 8000

# ── Dashboard ─────────────────────────────────────────────────────────────────
dashboard:
	$(POETRY) run streamlit run pulsecast/dashboard/app.py

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	$(POETRY) run pytest tests/ -v --tb=short

# ── Docker helpers ────────────────────────────────────────────────────────────
up:
	docker compose up --build -d

down:
	docker compose down -v

logs:
	docker compose logs -f

# ── Install dependencies ──────────────────────────────────────────────────────
install:
	$(POETRY) install
