"""
export.py – ONNX export for the LightGBM quantile models with parity
validation.

Usage
-----
>>> from pulsecast.models.export import export_lgbm_to_onnx
>>> export_lgbm_to_onnx(forecaster, n_features=42, output_dir="models/onnx")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pulsecast.models.lgbm import LGBMForecaster
logger = logging.getLogger(__name__)

_QUANTILE_NAMES = ("p10", "p50", "p90")
_QUANTILE_VALUES = (0.1, 0.5, 0.9)
_PARITY_ATOL = 1e-3  # absolute tolerance for ONNX parity check


def export_lgbm_to_onnx(
    forecaster: LGBMForecaster,
    n_features: int,
    output_dir: str | Path = "models/onnx",
    n_parity_rows: int = 200,
    seed: int = 42,
) -> dict[str, Path]:
    """
    Export all three LightGBM quantile models to ONNX and validate parity.

    Parameters
    ----------
    forecaster : LGBMForecaster
        A fitted instance of :class:`pulsecast.models.lgbm.LGBMForecaster`.
    n_features : int
        Number of input features (used to generate parity check data).
    output_dir : str | Path
        Directory where ONNX files are saved.
    n_parity_rows : int
        Number of synthetic rows used for parity validation.
    seed : int
        Random seed for parity data generation.

    Returns
    -------
    dict mapping quantile name (``"p10"``, ``"p50"``, ``"p90"``) to Path.
    """
    try:
        from onnxmltools import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError as exc:
        raise ImportError(
            "onnxmltools is required for ONNX export. "
            "Install it with: pip install onnxmltools onnxruntime"
        ) from exc

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for parity validation. "
            "Install it with: pip install onnxruntime"
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    X_parity = rng.standard_normal((n_parity_rows, n_features)).astype(np.float32)

    saved_paths: dict[str, Path] = {}

    for q_name, q_val in zip(_QUANTILE_NAMES, _QUANTILE_VALUES):
        lgbm_model = forecaster._models.get(q_val)
        if lgbm_model is None:
            logger.warning("No model found for quantile %.1f – skipping.", q_val)
            continue

        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model = convert_lightgbm(lgbm_model.booster_, initial_types=initial_type)

        onnx_path = output_dir / f"lgbm_{q_name}.onnx"
        onnx_path.write_bytes(onnx_model.SerializeToString())
        logger.info("Exported %s → %s", q_name, onnx_path)

        # --- parity validation ---
        lgbm_preds = lgbm_model.predict(X_parity.astype(np.float64))
        sess = ort.InferenceSession(str(onnx_path))
        input_name = sess.get_inputs()[0].name
        onnx_preds = sess.run(None, {input_name: X_parity})[0].flatten()

        max_diff = float(np.max(np.abs(lgbm_preds - onnx_preds)))
        if max_diff > _PARITY_ATOL:
            raise ValueError(
                f"ONNX parity check failed for {q_name}: "
                f"max |lgbm - onnx| = {max_diff:.6f} > {_PARITY_ATOL}"
            )
        logger.info(
            "Parity OK for %s (max diff=%.2e < %.2e)", q_name, max_diff, _PARITY_ATOL
        )
        saved_paths[q_name] = onnx_path

    return saved_paths
