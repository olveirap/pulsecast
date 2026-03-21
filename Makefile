.PHONY: ingest ingest-tlc ingest-subway ingest-bus backfill build-zone-maps features train export serve test

# ── Configuration ────────────────────────────────────────────────────────────
PYTHON   ?= python
POETRY   ?= poetry
MONTHS   ?= 24
HORIZON  ?= 7
# Dynamic backfill dates: defaults to 180 days ago until today
# Override with BACKFILL_START and BACKFILL_END environment variables for testing
TODAY        := $(shell date +%Y-%m-%d)
BACKFILL_DAYS := 180
BACKFILL_START ?= $(shell date -d "$(TODAY) - $(BACKFILL_DAYS) days" +%Y-%m-%d)
BACKFILL_END   ?= $(TODAY)

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
	$(POETRY) run python scripts/build_subway_zone_map.py

# ── Feature engineering ───────────────────────────────────────────────────────
features:
	$(POETRY) run python scripts/run_features.py

update-disruption-flag:
	$(POETRY) run python scripts/update_disruption_flag.py

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
