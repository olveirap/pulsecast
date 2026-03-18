"""
lgbm.py – LightGBM quantile regression (q=0.1, 0.5, 0.9) with sktime
walk-forward cross-validation.

Usage
-----
>>> from pulsecast.models.lgbm import LGBMForecaster
>>> model = LGBMForecaster()
>>> model.fit(X_train, y_train)
>>> preds = model.predict(X_test)   # {"p10": [...], "p50": [...], "p90": [...]}
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError(
        "lightgbm is required. Install it with: pip install lightgbm"
    ) from exc

_QUANTILES = (0.1, 0.5, 0.9)

_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbose": -1,
}


class LGBMForecaster:
    """
    Trains three LightGBM models (one per quantile) for probabilistic
    demand forecasting.

    Parameters
    ----------
    params : dict | None
        Additional LightGBM hyperparameters (merged with defaults).
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        base = dict(_DEFAULT_PARAMS)
        if params:
            base.update(params)
        self._base_params = base
        self._models: dict[float, lgb.LGBMRegressor] = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "LGBMForecaster":
        """
        Fit one quantile regressor per quantile level.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
        eval_set : optional (X_val, y_val) for early stopping
        """
        for q in _QUANTILES:
            params = dict(self._base_params)
            params["objective"] = "quantile"
            params["alpha"] = q
            model = lgb.LGBMRegressor(**params)
            fit_kwargs: dict[str, Any] = {}
            if eval_set is not None:
                fit_kwargs["eval_set"] = [(eval_set[0], eval_set[1])]
                fit_kwargs["callbacks"] = [lgb.early_stopping(50, verbose=False)]
            model.fit(X, y, **fit_kwargs)
            self._models[q] = model
            logger.info("Fitted LightGBM quantile q=%.2f", q)
        return self

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """
        Return predictions for all three quantiles.

        Returns
        -------
        dict with keys ``"p10"``, ``"p50"``, ``"p90"``.
        """
        if not self._models:
            raise RuntimeError("Call fit() before predict().")
        return {
            "p10": self._models[0.1].predict(X),
            "p50": self._models[0.5].predict(X),
            "p90": self._models[0.9].predict(X),
        }

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        gap: int = 24,
    ) -> list[dict[str, float]]:
        """
        Walk-forward cross-validation using a time-series split.

        Returns a list of per-fold pinball-loss dicts keyed by quantile.
        """
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        fold_results: list[dict[str, float]] = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            self.fit(X[train_idx], y[train_idx])
            preds = self.predict(X[val_idx])
            y_val = y[val_idx]
            losses: dict[str, float] = {}
            for q_name, q_val in zip(("p10", "p50", "p90"), _QUANTILES):
                err = y_val - preds[q_name]
                loss = float(np.mean(np.where(err >= 0, q_val * err, (q_val - 1) * err)))
                losses[q_name] = loss
            fold_results.append(losses)
            logger.info("Fold %d – losses: %s", fold, losses)

        return fold_results
