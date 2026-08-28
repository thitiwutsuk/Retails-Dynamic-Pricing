# Retails-Dynamic-Pricing

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.x-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![statsmodels](https://img.shields.io/badge/statsmodels-SARIMA-8CAAE6?logo=python&logoColor=white)](https://www.statsmodels.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-experiment%20tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![pytest](https://img.shields.io/badge/pytest-8%20passed-0A9EDC?logo=pytest&logoColor=white)](https://pytest.org/)

End-to-end data science project solving three linked retail decisions — **demand forecasting**, **inventory
optimization**, and **dynamic pricing** — on one panel dataset: 5 stores × 20 products × 731 days
(2022-01-01 → 2024-01-01), 73,100 rows, no missing values.

## Problem

For every (store, product) pair, three questions determine revenue and cost directly:

1. **Forecasting** — does a global LSTM beat classical per-series SARIMA on held-out demand?
2. **Inventory** — given that forecast and its uncertainty, what reorder policy minimizes stockouts without over-stocking?
3. **Pricing** — given estimated demand elasticity, what price maximizes revenue?

The dataset's own `Demand Forecast` column is treated strictly as an external benchmark, never as a
feature or target. All three challenges share one feature pipeline and are evaluated on the same held-out
window: **2023-10-01 → 2024-01-01**.

## Results

| Challenge | Method | Result |
|---|---|---|
| Forecasting | Global LSTM vs. per-series SARIMA, one-step-ahead walk-forward | **LSTM MASE 0.709 vs. SARIMA 0.722** (Diebold-Mariano p<0.001); both beat naive (0.941) and seasonal-naive (0.978) |
| Inventory | ROP/EOQ simulated against real demand, 3 policies | LSTM-driven policy has a *higher* stockout rate (14.4%) than naive-driven (11.6%) — a ~6% low bias in LSTM's mean forecast undersizes the reorder point despite its lower day-to-day error |
| Pricing | Log-log elasticity regression + bounded grid search, per category | VIF ≈ 41 and R² < 2% in every category; 3 of 5 categories return economically backwards (positive) elasticity — recommendations not reliable enough to act on |

Full write-ups: [`reports/forecasting_report.md`](reports/forecasting_report.md),
[`reports/inventory_report.md`](reports/inventory_report.md), [`reports/pricing_report.md`](reports/pricing_report.md).

**Takeaway**: a better accuracy metric (MASE) did not translate into a better downstream decision, because an
undiagnosed forecast bias directly undersized the reorder point. Every stage's caveats are reported honestly
rather than smoothed over.

## Project Structure

```
├── data/
│   ├── raw/retail_store_inventory.csv        # immutable source file
│   ├── interim/cleaned_panel.parquet          # typed, validated panel (gitignored)
│   └── processed/                             # features_full + train/val/test.parquet + forecasts/
│
├── notebooks/retail_demand_forecasting.ipynb  # single notebook, Thai explanations, runs top-to-bottom
│
├── src/                                       # tested pipeline code — source of truth
│   ├── config.py                              # paths, seed, TARGET_COL, split dates, series scope
│   ├── data/                                  # load.py, clean.py, features.py
│   ├── forecasting/                           # splits.py, arima_model.py, lstm_model.py, evaluate.py, backtest.py
│   ├── inventory/                             # policies.py (safety stock/ROP/EOQ), simulate.py
│   └── pricing/                               # elasticity.py, optimize.py
│
├── models/{arima,lstm}/                       # fitted orders / checkpoint + scalers (gitignored)
├── mlflow.db + mlruns/                        # experiment tracking (gitignored)
├── reports/{figures,*.md}                     # per-challenge write-ups + charts
├── tests/                                     # pytest — leakage, split-overlap, formula correctness
└── requirements.txt
```

## Pipeline

1. **Business understanding** — each challenge mapped to a target and success metric (table above).
2. **Data understanding** — audits the panel for a full dense grid and valid ranges; EDA surfaces weekly
   seasonality and a demand-censoring signal (`Units Sold` capped by `Inventory Level`) that shapes later features.
3. **Data preparation** — `src/data/features.py` builds calendar, lag/rolling (leakage-checked, shifted by 1
   day), price, and inventory features once, shared by all three challenges. Chronological split: train ≤
   2023-06-30, val ≤ 2023-09-30, test ≤ 2024-01-01.
4. **Modeling** — SARIMA + global LSTM (forecasting) → ROP/EOQ simulation (inventory) → elasticity + price
   search (pricing). See Results above.
5. **Evaluation** — MASE + Diebold-Mariano (forecasting), stockout/holding-cost tradeoff + service-level
   sensitivity (inventory), VIF/R² diagnostics (pricing) — all on the same test window.
6. **Reporting** — per-challenge reports plus a cross-challenge synthesis in the notebook.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run `notebooks/retail_demand_forecasting.ipynb` top to bottom (Stage 1 → 4c in one pass). Then:

```bash
pytest tests/ -q                                        # 8 tests: leakage, splits, formulas
mlflow ui --backend-store-uri sqlite:///mlflow.db        # inspect SARIMA/LSTM runs
```

`src/config.py:N_SERIES_SUBSET` controls scope — `10` (top-volume series) by default for fast iteration,
`"all"` reruns the identical pipeline on all 100 series with no other code changes.

## Status

All 3 challenges built, evaluated, and reported end to end, currently on a 10-series subset.
