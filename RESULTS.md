# Results

Ablation study comparing model variants on a held-out 30-day test set
(hourly demand per TLC zone, NYC 2024).

> **Note:** Results are placeholders pending full experiment runs.
> Update this table after running `make train` and evaluating on the test set.

## Metrics

| Model | MAE | RMSE | Pinball p10 | Pinball p50 | Pinball p90 | Coverage 80% CI |
|---|---|---|---|---|---|---|
| MSTL (baseline) | — | — | — | — | — | — |
| LightGBM | — | — | — | — | — | — |
| LightGBM + delay_index | — | — | — | — | — | — |
| TFT + delay_index | — | — | — | — | — | — |

## Notes

- **MSTL**: Seasonal-Trend decomposition using LOESS with daily (24 h) and
  weekly (168 h) seasonality + AutoARIMA residual model.
- **LightGBM**: Quantile regression (q=0.1, 0.5, 0.9) trained on
  demand lag/rolling/EWM features and calendar features only.
- **LightGBM + delay_index**: Adds lag-1 and rolling-3h GTFS-RT congestion
  features.
- **TFT + delay_index**: Temporal Fusion Transformer (7-day horizon) trained
  on all features including real-time congestion covariates.

## Evaluation Protocol

- Walk-forward cross-validation: 5 folds, 24-hour gap between train/test.
- Test set: final 30 days withheld from all CV folds.
- Pinball loss reported as average across all zones and forecast steps.
