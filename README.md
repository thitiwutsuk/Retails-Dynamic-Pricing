<<<<<<< HEAD
# Retails-Dynamic-Pricing
=======
# Retail Store Inventory — Forecasting, Inventory Optimization & Dynamic Pricing

End-to-end data science project on a single retail panel dataset (`data/raw/retail_store_inventory.csv`):
5 stores × 20 products × 731 daily dates (2022-01-01 to 2024-01-01), 73,100 rows, no missing values.

## Stage 1 — Business Understanding

Three linked challenges, all evaluated on the same held-out test window (2023-10-01 to 2024-01-01):

| Challenge | Problem | Target | Success metric |
|---|---|---|---|
| 1. Demand Forecasting | Predict `Units Sold` per (Store ID, Product ID), 1-day and 7-day ahead | `Units Sold` | Beat SARIMA / seasonal-naive baselines on MASE |
| 2. Inventory Optimization | Choose reorder point / order quantity per (Store ID, Product ID) from Challenge 1's forecast + its uncertainty | Stockout rate, holding cost | Lower stockout rate and/or total cost vs. the historical `Units Ordered` policy |
| 3. Dynamic Pricing | Recommend a revenue-maximizing price per segment from estimated elasticity, competitor price, and discount behavior | `Price` recommendation | Positive expected revenue lift vs. historical average price, within a realistic price band |

The dataset's own `Demand Forecast` column is treated strictly as an **external benchmark to beat** — it is never used as a model feature (that would leak a competing forecast into our own models).

## Project layout

```
data/{raw,interim,processed}   raw is immutable; interim/processed are gitignored, regenerate via the pipeline
notebooks/                     EDA and per-stage analysis notebooks (import from src/, not the reverse)
src/{data,forecasting,inventory,pricing,viz}   reusable pipeline code
models/{arima,lstm}             saved model artifacts (gitignored)
mlruns/                         MLflow tracking store (gitignored)
reports/                        write-ups + figures per challenge
tests/                          pytest — feature/split leakage checks, formula correctness
```

## Reproducing the pipeline

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run notebooks in order: `00_data_audit` → `01_eda` → `02_data_preparation` → `10_arima_baseline` →
`11_lstm_forecasting` → `12_forecast_comparison` → `20_inventory_optimization` → `30_price_elasticity` →
`31_dynamic_pricing_optimization`.

Inspect experiment runs with:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

### Scope: subset vs. full 100 series

`src/config.py:N_SERIES_SUBSET` controls how many (Store ID, Product ID) series the pipeline runs on.
It starts at `10` (top-10 by historical volume) for fast iteration. Set it to `"all"` to re-run the
identical pipeline on all 100 series — no code changes needed elsewhere.

## Status

- [x] Stage 1 — Business understanding (this README)
- [x] Stage 2 — Data understanding / EDA
- [x] Stage 3 — Data preparation (shared feature pipeline)
- [ ] Stage 4a — Forecasting (SARIMA vs PyTorch LSTM, MLflow-tracked)
- [ ] Stage 4b — Inventory optimization
- [ ] Stage 4c — Dynamic pricing
- [ ] Stage 5 — Evaluation
- [ ] Stage 6 — Reporting / synthesis
>>>>>>> cfe9e1b (Initial commit: retail demand forecasting, inventory & pricing project)
