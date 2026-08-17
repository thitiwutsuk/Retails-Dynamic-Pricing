# Forecasting Report — LSTM vs. SARIMA

**Question**: can a global PyTorch LSTM forecast daily `Units Sold` more accurately than a per-series SARIMA
model, on a fair, apples-to-apples evaluation?

## Method

- Scope: 10 (Store ID, Product ID) series (top-10 by historical volume), held-out test window 2023-10-01 to
  2024-01-01 (92 days).
- Both models evaluated **one-step-ahead, walk-forward, with true history**: predict day *t*, then update the
  model with the *actual* value of day *t* before predicting day *t+1*. Neither model gets an easier or harder
  evaluation protocol than the other.
- SARIMA (`src/forecasting/arima_model.py`): `pmdarima.auto_arima(seasonal=True, m=7)`, fit **per series**
  (10 independent models).
- LSTM (`src/forecasting/lstm_model.py`): one **global** model trained across all 10 series (2-layer LSTM,
  hidden 64→32, 28-day window, Huber loss, early stopping on validation loss — stopped at epoch 9).
- Primary metric: **MASE** (scale-free, comparable across series of very different volume), scaled by each
  series' own seasonal-naive error computed on train only.

## Results

| Model | MAE | RMSE | sMAPE | MASE (macro) | MASE (volume-weighted) |
|---|---|---|---|---|---|
| DatasetForecast* | 8.35 | 10.07 | 15.5% | 0.066 | 0.067 |
| **LSTM** | 89.03 | 110.33 | 69.6% | **0.709** | 0.712 |
| SARIMA | 90.63 | 110.36 | 70.0% | 0.722 | 0.724 |
| Naive | 118.23 | 151.63 | 89.9% | 0.941 | 0.944 |
| Seasonal-Naive | 122.48 | 154.24 | 92.4% | 0.978 | 0.983 |

**LSTM wins**: MASE 0.709 vs. SARIMA's 0.722. A Diebold-Mariano test on the paired absolute errors confirms
the gap is statistically significant (DM = 3.833, **p = 0.0001**), not sampling noise. Both models comfortably
beat the naive and seasonal-naive baselines.

\* `DatasetForecast` is the CSV's own pre-existing `Demand Forecast` column, included only as an external
benchmark to beat — never used as a model feature or target. Its implausibly low MASE (15x better than either
model we built) is a strong signal it was constructed with information not realistically available at
forecast time (e.g. derived from the actual value plus small noise), so it is **not** treated as a fair
target to match — SARIMA-vs-LSTM is the comparison this project actually set out to make.

## Reproduce

`notebooks/retail_demand_forecasting.ipynb`, Stage 4a and the "Stage 5" forecast-comparison cells. Full
per-series orders and MLflow run history: `models/arima/orders.csv`, `mlflow ui --backend-store-uri
sqlite:///mlflow.db` (experiments `forecasting_arima`, `forecasting_lstm`).
