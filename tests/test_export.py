"""
tests/test_export.py – Unit tests for models/export.py ONNX export.

All tests run offline using a tiny LightGBM model trained on synthetic data.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Skip the entire module if optional ONNX dependencies are not installed.
pytest.importorskip("onnxmltools")
ort = pytest.importorskip("onnxruntime")

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_N_FEATURES = 5
_N_ROWS = 80

# ---------------------------------------------------------------------------
# Fixtures (module-scoped so training and export run only once per session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def forecaster():
    """Fit a tiny LightGBM forecaster once for the entire module."""
    from pulsecast.models.lgbm import LGBMForecaster

    rng = np.random.default_rng(0)
    X = rng.standard_normal((_N_ROWS, _N_FEATURES))
    y = rng.standard_normal(_N_ROWS)
    fc = LGBMForecaster(params={"n_estimators": 10})
    fc.fit(X, y)
    return fc


@pytest.fixture(scope="module")
def exported_paths(forecaster):
    """Export the fitted forecaster to ONNX once and yield the paths dict."""
    from pulsecast.models.export import export_lgbm_to_onnx

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = export_lgbm_to_onnx(
            forecaster, n_features=_N_FEATURES, output_dir=tmpdir, n_parity_rows=20
        )
        yield paths


# ---------------------------------------------------------------------------
# export_lgbm_to_onnx
# ---------------------------------------------------------------------------


def test_export_returns_three_paths(exported_paths):
    """export_lgbm_to_onnx should return paths for p10, p50, and p90."""
    assert set(exported_paths.keys()) == {"p10", "p50", "p90"}
    for path in exported_paths.values():
        assert isinstance(path, Path)


def test_export_files_exist(exported_paths):
    """Exported ONNX files must exist on disk."""
    for path in exported_paths.values():
        assert path.exists(), f"{path} does not exist"


def test_export_onnx_output_shape(exported_paths):
    """Each exported ONNX model must produce shape (n_rows, 1) for quantile outputs."""
    n_test = 15
    rng = np.random.default_rng(7)
    X_test = rng.standard_normal((n_test, _N_FEATURES)).astype(np.float32)

    for q_name, path in exported_paths.items():
        sess = ort.InferenceSession(str(path))
        input_name = sess.get_inputs()[0].name
        outputs = sess.run(None, {input_name: X_test})
        assert outputs[0].shape == (
            n_test,
            1,
        ), f"{q_name}: expected shape ({n_test}, 1), got {outputs[0].shape}"


def test_export_parity_with_lgbm_predict(forecaster, exported_paths):
    """ONNX predictions must match LightGBM predictions within tolerance."""
    from pulsecast.models.export import _PARITY_ATOL

    rng = np.random.default_rng(99)
    X_test = rng.standard_normal((50, _N_FEATURES))
    lgbm_preds = forecaster.predict(X_test)

    for q_name, path in exported_paths.items():
        sess = ort.InferenceSession(str(path))
        input_name = sess.get_inputs()[0].name
        onnx_preds = sess.run(None, {input_name: X_test.astype(np.float32)})[0].flatten()
        max_diff = float(np.max(np.abs(lgbm_preds[q_name] - onnx_preds)))
        assert max_diff <= _PARITY_ATOL, (
            f"{q_name}: parity check failed, max diff={max_diff:.6f} > {_PARITY_ATOL}"
        )
