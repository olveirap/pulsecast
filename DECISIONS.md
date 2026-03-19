# Architecture Decision Records (ADRs)

---

## ADR-001b – Bus Position Variance as Primary Congestion Covariate

**Status:** Accepted (Supersedes ADR-001)

**Context:**  
The original GTFS-RT `delay_index` design (ADR-001) was not viable for
training because no historical GTFS-RT archive exists.  Forecast models
require a congestion covariate that is available both historically (for
training) and in real-time (for inference).

**Decision:**  
Use the NYC Bus Positions dataset (`s3://nycbuspositions`) as the primary
congestion covariate.  A `travel_time_var` metric — defined as the
variance of segment travel times for buses within a TLC zone and truncated
hour — is computed and stored in the `congestion` table.  MTA subway
GTFS-RT delays are retained as a supplementary real-time signal in the
`subway_delay` table.

**Rationale:**
- **Training availability:** Historical bus positions exist for ~18 months
  on S3, enabling the covariate to be included in the training set.
- **Signal quality:** Travel time variance is a more robust proxy for
  unpredictable traffic congestion than mean arrival delay, which is
  often confounded by scheduled buffer time.
- **Asymmetric roles:** Bus positions provide the consistent signal
  required for the ONNX feature vector; subway RT provides high-frequency
  disruption signals for the serving layer.

**Alternatives considered:**
- GTFS-RT (ADR-001): Dropped due to lack of historical archive.
- Paid traffic APIs (HERE/TomTom): Dropped due to cost and licensing
  restrictions for open-source distribution.

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

## ADR-003b – Travel Time Variance Bucketing in Cache Key

**Status:** Accepted (Updates ADR-003)

**Context:**  
The Redis cache key includes the current congestion value.  With the shift
from mean delay (seconds) to travel time variance (seconds squared), the
original 0.5 bucketing resolution is too fine.

**Decision:**  
Round the `travel_time_var` to the nearest 10.0 before constructing the
cache key.

**Rationale:**
- Variance values are numerically larger and more volatile than mean
  delays.  A 10.0 bucket size (equivalent to ~3.2s standard deviation
  fluctuation) provides a stable cache key without masking meaningful
  congestion trends.
- Empirically, variance buckets of this size align with the sensitivity
  of the LightGBM models observed during Phase 0 EDA.

**Alternatives considered:**
- Previous 0.5 bucket: Results in near-zero cache hit rate for variance
  signals.
- Dynamic bucketing: Too complex for a low-latency serving layer.

---
