# Architecture Decision Records (ADRs)

---

## ADR-001 – GTFS-RT as Congestion Covariate

**Status:** Accepted

**Context:**  
Taxi demand in NYC is strongly correlated with subway and bus network
disruptions.  When transit is delayed or suspended, ride-hail demand
spikes — particularly for routes near major transit hubs.

**Decision:**  
Use the MTA GTFS-Realtime trip-updates feed as the primary congestion
covariate.  A `delay_index` metric — defined as the mean arrival delay
(in seconds) weighted by trip count, grouped by TLC zone and truncated
hour — is computed every 60 seconds and stored in TimescaleDB.

**Rationale:**
- GTFS-RT is published under an open-data licence, freely available at
  `https://api.mta.info/GTFS`.
- The `delay_index` aggregation reduces the high cardinality of raw
  stop-level updates to a single, zone-level signal that aligns with the
  granularity of TLC demand data.
- Real-time ingestion (60-second polling) keeps the signal current enough
  to influence short-term (1–7 day) forecasts.

**Alternatives considered:**
- Traffic speed APIs (e.g., HERE, TomTom): paid, higher latency.
- Weather APIs: complementary but not a direct proxy for transit demand.

---

## ADR-002 – ONNX for Model Serving

**Status:** Accepted

**Context:**  
The LightGBM quantile models must be served with low latency (< 50 ms p99)
at scale.  Python-native LightGBM inference requires the full Python stack
and is not easily portable across deployment targets.

**Decision:**  
Export the three quantile LightGBM models to ONNX format using `skl2onnx`
and run inference via `onnxruntime`.  A numerical parity check (max
absolute difference < 1e-3) is performed at export time.

**Rationale:**
- ONNX Runtime is a C++ inference engine with Python bindings; it delivers
  significantly lower per-sample latency than LightGBM's Python predict.
- ONNX models are portable: the same artefact can run in the FastAPI
  container, a Lambda function, or edge devices without code changes.
- The parity validation step catches silent numerical regressions during
  export.

**Alternatives considered:**
- Triton Inference Server: more operational complexity than warranted for
  v0.1.
- Bentoml: adds another abstraction layer; ONNX Runtime is simpler.

---

## ADR-003 – delay_index Bucketing in Cache Key

**Status:** Accepted

**Context:**  
The Redis cache key includes the current `delay_index` value.  If the raw
floating-point value is used, minor fluctuations in congestion (e.g.,
140.23 vs 140.27 seconds mean delay) would cause unnecessary cache misses
and redundant ONNX inferences.

**Decision:**  
Round the `delay_index` to the nearest 0.5 before constructing the cache
key.

**Rationale:**
- A rounding resolution of 0.5 seconds is finer than the measurement noise
  in the GTFS-RT feed (which is reported in whole seconds per stop).
- It dramatically increases cache hit rate — empirically, 90 %+ of
  real-time delay values fall within 0.25 seconds of the nearest 0.5
  boundary.
- The resulting forecast error introduced by this approximation is well
  below the model's inherent uncertainty (quantile interval width).

**Alternatives considered:**
- No bucketing (raw float): unacceptably low cache hit rate.
- Bucket to nearest 1.0: coarser than necessary; could mask genuine
  disruptions at the boundary (e.g., 1.9 → 2.0 disruption threshold).
- Ignore delay_index in cache key: incorrect — different congestion levels
  produce materially different forecasts.

---

## ADR-004 – TLC Zone to Route Corridor Mapping

**Status:** Accepted

**Context:**
The `demand` table stores `route_id`, while NYC TLC trip records provide
pickup geography as `PULocationID` (taxi zone IDs).  Without a stable mapping,
ingestion writes one sparse time series per zone and does not align with the
intended route-level modelling granularity.

**Decision:**
Add a versioned mapping file at `data/zone_routes.csv` that assigns each TLC
`PULocationID` (1..263) to one of 20 logical borough-pair corridor route IDs.
The mapping is loaded by `pulsecast.data.ingest.tlc` and applied during
`aggregate_hourly()` so downstream storage and modelling consume route-level
series directly.

**Rationale:**
- Keeps the mapping auditable and editable without code changes.
- Collapses high-cardinality zone traffic into a compact set of route IDs
  suitable for stable model training.
- Applies mapping early in the pipeline (during hourly aggregation), reducing
  duplicate downstream regrouping work.

**Alternatives considered:**
- Identity mapping (`route_id = PULocationID`): too sparse and does not match
  route-level intent.
- Hardcoded Python dict: difficult to review and maintain compared to CSV.
- Applying mapping only at DB write time: late transformation leaks zone-level
  structure into intermediate outputs and tests.
