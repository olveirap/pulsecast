"""
run_export.py – ONNX export for Pulsecast models.

Loads the fitted LGBMForecaster and exports it to ONNX format for serving.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

from pulsecast.models.export import export_lgbm_to_onnx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Number of features must match _LGBM_FEATURES in run_train.py
_N_FEATURES = 22


def main() -> None:
    models_dir = Path(os.getenv("MODELS_DIR", "models"))
    onnx_dir = models_dir / "onnx"
    pickle_path = models_dir / "lgbm_forecaster.pkl"

    if not pickle_path.exists():
        logger.error("Fitted LightGBM pickle not found at %s. Run 'make train' first.", pickle_path)
        return

    logger.info("Loading fitted LGBMForecaster from %s", pickle_path)
    with open(pickle_path, "rb") as f:
        forecaster = pickle.load(f)

    logger.info("Exporting to ONNX in %s", onnx_dir)
    saved_paths = export_lgbm_to_onnx(
        forecaster=forecaster,
        n_features=_N_FEATURES,
        output_dir=onnx_dir
    )
    
    for q, path in saved_paths.items():
        logger.info("Saved %s to %s", q, path)


if __name__ == "__main__":
    main()
