# Retails-Dynamic-Pricing

End-to-end data science project on a single retail panel dataset (`data/raw/retail_store_inventory.csv`):
**5 stores × 20 products × 731 daily dates** (2022-01-01 to 2024-01-01), **73,100 rows**, no missing values,
15 raw columns (Date, Store ID, Product ID, Category, Region, Inventory Level, Units Sold, Units Ordered,
Demand Forecast, Price, Discount, Weather Condition, Holiday/Promotion, Competitor Pricing, Seasonality).

The project follows the standard data science lifecycle — **Business Understanding → Data Understanding →
Data Preparation → Modeling → Evaluation → Reporting** — to solve three linked challenges built on top of
one shared pipeline:

1. **Demand Forecasting** — predict `Units Sold`; LSTM vs. classical ARIMA/SARIMA.
2. **Inventory Optimization** — reorder point / order quantity policy that minimizes stockouts and overstock, built on Challenge 1's forecasts.
3. **Dynamic Pricing** — revenue-maximizing price recommendation from estimated demand elasticity, competitor pricing, and discount behavior.

---

## Project Structure

```
Retail Store Inventory/
├── data/
│   ├── raw/
│   │   └── retail_store_inventory.csv        # original file, treated as immutable, never overwritten
│   ├── interim/
│   │   └── cleaned_panel.parquet             # output of Stage 2 audit: typed, validated, sorted panel (gitignored, regenerate via 00_data_audit.ipynb)
│   └── processed/
│       ├── features_full.parquet             # output of Stage 3: full engineered feature set (gitignored, regenerate via 02_data_preparation.ipynb)
│       ├── train.parquet                     # 2022-01-01 -> 2023-06-30
│       ├── val.parquet                       # 2023-07-01 -> 2023-09-30
│       ├── test.parquet                      # 2023-10-01 -> 2024-01-01 (held out, touched only for final evaluation)
│       └── forecasts/                        # baseline/arima/lstm forecast outputs from Stage 4a
│
├── notebooks/                                # imports from src/, never the other way around
│   └── full_pipeline_th.ipynb                # single notebook, Thai explanations, Stage 1 -> 4a end-to-end (runs top-to-bottom);
│                                              #   Stage 4b/4c will be appended here too, not split into new files
│
├── src/                                      # reusable, tested pipeline code — the source of truth; notebooks call into this
│   ├── config.py                             # paths, random seed, TARGET_COL, split dates, N_SERIES_SUBSET scope flag
│   ├── data/
│   │   ├── load.py                           # read_raw() / read_cleaned() / read_features() helpers
│   │   ├── clean.py                          # clean_panel(): dtype casting + range validation (Stage 2)
│   │   └── features.py                       # select_series(), engineer_features(), build_features_full() (Stage 3)
│   ├── forecasting/
│   │   ├── splits.py                         # time_split(): chronological train/val/test, reused by every stage
│   │   ├── arima_model.py                    # per-series SARIMA, one-step-ahead walk-forward eval, MLflow logging
│   │   ├── lstm_model.py                     # PyTorch windowing + global LSTM, one-step-ahead eval, MLflow logging
│   │   ├── evaluate.py                       # MAE / RMSE / sMAPE / MASE metrics, shared per-series scale
│   │   └── backtest.py                       # 2-origin chronological robustness check
│   ├── inventory/
│   │   ├── policies.py                       # safety stock / reorder point / EOQ formulas (planned)
│   │   └── simulate.py                       # day-step inventory simulation across policies (planned)
│   ├── pricing/
│   │   ├── elasticity.py                     # log-log demand elasticity regression (planned)
│   │   └── optimize.py                       # revenue-maximizing price search (planned)
│   └── viz/
│       └── plots.py                          # shared matplotlib/plotly helpers (planned)
│
├── models/
│   ├── arima/                                # orders.csv (chosen SARIMA orders per series, gitignored)
│   └── lstm/                                 # lstm.pt checkpoint + per-series scalers (gitignored)
│
├── mlflow.db + mlruns/                       # MLflow experiment tracking (sqlite backend) — forecasting_arima (10 runs) + forecasting_lstm (1 run) logged (gitignored)
│
├── reports/
│   ├── figures/                              # PNG charts produced by notebooks (EDA + forecasting figures populated)
│   ├── forecasting_report.md                 # Stage 6 write-up (planned)
│   ├── inventory_report.md                   # Stage 6 write-up (planned)
│   └── pricing_report.md                     # Stage 6 write-up (planned)
│
├── tests/                                    # pytest — run before trusting any pipeline output
│   ├── test_features.py                      # lag/rolling leakage checks, subset-selection correctness, one-hot correctness
│   ├── test_splits.py                        # train/val/test cover every row exactly once, zero date overlap
│   ├── test_inventory_policies.py            # EOQ/ROP formula correctness on hand-checked cases (planned)
│   └── test_elasticity.py                    # elasticity model diagnostics (planned)
│
├── requirements.txt
├── .gitignore
└── README.md
```

Items marked **(planned)** are the remaining Stage 4b/4c/6 work; everything else already exists and runs
end-to-end (see [Status](#status)).

---

## How each stage works

### Stage 1 — Business Understanding

Before touching any data, each challenge is turned into a precise, measurable problem so there's an
unambiguous definition of "success" to test against later:

| Challenge | Problem | Target | Success metric |
|---|---|---|---|
| 1. Demand Forecasting | Predict `Units Sold` per (Store ID, Product ID), 1-day and 7-day ahead | `Units Sold` | Beat SARIMA / seasonal-naive baselines on MASE |
| 2. Inventory Optimization | Choose reorder point / order quantity per (Store ID, Product ID) from Challenge 1's forecast + its uncertainty | Stockout rate, holding cost | Lower stockout rate and/or total cost vs. the historical `Units Ordered` policy |
| 3. Dynamic Pricing | Recommend a revenue-maximizing price per segment from estimated elasticity, competitor price, and discount behavior | `Price` recommendation | Positive expected revenue lift vs. historical average price, within a realistic price band |

The dataset's own `Demand Forecast` column is treated strictly as an **external benchmark to beat** — it is
never used as a model feature or target (that would leak a competing forecast into our own models). All three
challenges are evaluated on the **same held-out test window** (2023-10-01 to 2024-01-01), since Challenge 2
consumes Challenge 1's output and Challenge 3 shares the same feature pipeline — a fair, single point of
comparison matters more than any one challenge in isolation.

### Stage 2 — Data Understanding (EDA)

Run once, shared by all three challenges. Covered by the "Stage 2a / 2b" sections of
`notebooks/full_pipeline_th.ipynb`:

**Data audit** — structural/quality checks before anything else:
- Confirms the panel is a full dense grid: every (Store ID, Product ID) pair has exactly one row per date,
  731 dates × 100 series = 73,100 rows, no gaps and no duplicates.
- Verifies zero missing values across all 15 columns.
- Range/sanity checks: Inventory Level / Units Sold / Units Ordered ≥ 0, Discount ∈ [0,100], Price and
  Competitor Pricing > 0, Holiday/Promotion is strictly binary.
- Casts dtypes (categoricals, int8 flags, real datetime) via `src/data/clean.py:clean_panel()` and persists
  the result to `data/interim/cleaned_panel.parquet` so no other notebook repeats this work.

**EDA analysis pass** — drives Stage 3's feature choices:
- Target behavior: daily total trend, average `Units Sold` by day-of-week (weekly seasonality is visible),
  by Category, and the effect of `Holiday/Promotion` / `Weather Condition`.
- Compares the dataset's own `Demand Forecast` against actual `Units Sold` (correlation + MAE) to
  characterize it as a benchmark, never a feature.
- Flags a **demand-censoring caveat**: a measurable share of rows have `Units Sold` at or near
  `Inventory Level`, suggesting stockouts may have suppressed true demand — carried forward explicitly into
  later modeling decisions rather than ignored.
- Pricing patterns: `Price` vs `Competitor Pricing` gap distribution, `Discount` vs `Units Sold` relationship
  by Category — these directly become the elasticity model's feature set in Challenge 3.
- Identifies representative high-volume / low-volume series to use as running examples in later notebooks.

Output: 7 charts saved to `reports/figures/`, plus the EDA findings that justify every feature built in
Stage 3.

### Stage 3 — Data Preparation

One reusable, pytest-covered pipeline consumed identically by all three challenges, so Forecasting,
Inventory, and Pricing never diverge on feature definitions:

**`src/data/features.py`**, computed per (Store ID, Product ID) group sorted by Date:
- **Calendar features** — day-of-week, is-weekend, month, week-of-year, plus sin/cos cyclical encodings so
  the model sees Sunday and Monday as adjacent rather than far apart.
- **Lag features** — `Units Sold` shifted 1 / 7 / 14 / 28 days.
- **Rolling features** — 7 / 14 / 28-day mean and std of `Units Sold`, rolling mean of `Inventory Level` and
  `Discount`. Every rolling window is computed on data **shifted by 1 day first**, so a given day's own value
  can never leak into its own rolling statistic — verified by `tests/test_features.py`.
- **Price features** — `price_gap`, `price_gap_pct`, `discount_pct`, `effective_price`.
- **Inventory features** — `days_of_supply`, a stockout proxy flag (from the Stage 2 censoring finding).
- **One-hot encoding** — Category, Region, Weather Condition, Seasonality.
- The dataset's `Demand Forecast` column is excluded from every feature set by design.

**Scope control** — `src/config.py:N_SERIES_SUBSET` selects a subset of (Store ID, Product ID) pairs (int N
= top-N by historical volume, a list of specific pairs, or `"all"`). Currently set to `10` for fast
iteration; switching it to `"all"` reruns the identical pipeline on all 100 series with zero code changes
elsewhere.

**Chronological split** — `src/forecasting/splits.py:time_split()` applies identical cut dates to every
series, no shuffling:

| Split | Range | Purpose |
|---|---|---|
| Train | 2022-01-01 → 2023-06-30 | model fitting |
| Validation | 2023-07-01 → 2023-09-30 | hyperparameter tuning / early stopping |
| Test | 2023-10-01 → 2024-01-01 | final held-out evaluation, touched once |

This exact split is reused unmodified by the Inventory simulation and Pricing evaluation stages.

The "Stage 3" section of `full_pipeline_th.ipynb` runs the full stage end-to-end: scope selection → feature
engineering → save `features_full.parquet` → time split → leakage/overlap assertions → save
`train/val/test.parquet`. `tests/test_features.py` and `tests/test_splits.py` (8 tests) assert lag/rolling
correctness, subset selection correctness, and zero-overlap splits — run before trusting any downstream
model output.

### Stage 4 — Modeling *(4a done, 4b/4c in progress)*

Three tracks built on Stage 3's output:

- **4a. Forecasting** (`src/forecasting/`, done) — naive/seasonal-naive/dataset baselines, per-series SARIMA
  (`pmdarima.auto_arima`, seasonal period 7), and a global PyTorch LSTM (28-day window, Huber loss, early
  stopping) trained once across all scoped series rather than per series. Both models are evaluated
  **one-step-ahead with true history** (walk-forward: predict 1 day, then feed the model the true observed
  value before predicting the next day) — an apples-to-apples protocol, not "LSTM gets true history while
  ARIMA gets a harder recursive rollout" or vice versa. Every SARIMA run (10 series) and the LSTM training run
  are logged to MLflow (`forecasting_arima`, `forecasting_lstm` experiments).

  **Result**: LSTM MASE 0.709 vs. SARIMA MASE 0.722 — LSTM wins, and a Diebold-Mariano test confirms the gap
  is statistically significant (p < 0.001). Both comfortably beat the naive (0.941) and seasonal-naive (0.978)
  baselines. The dataset's own `Demand Forecast` column scores implausibly well (MASE 0.066) — reported
  transparently in the "Stage 5" section of `full_pipeline_th.ipynb` as a likely artifact of how the
  synthetic dataset was generated, not a benchmark either model was realistically expected to beat.

- **4b. Inventory Optimization** (`src/inventory/`, planned) — will consume Track 1's LSTM forecast +
  backtested residual std as demand uncertainty; compute safety stock / reorder point / EOQ; simulate
  stockout rate and holding cost across policies (historical baseline vs. naive-forecast-driven vs.
  LSTM-forecast-driven).
- **4c. Dynamic Pricing** (`src/pricing/`, planned) — log-log elasticity regression per Category, discount
  effectiveness analysis, and a bounded grid search maximizing `price × predicted_demand`. Independent of
  Track 1, can run in parallel once Stage 3 is done.

### Stage 5 — Evaluation *(planned)*

Every track judged on the same Stage 3 test window: **MASE** as the primary forecasting metric (scale-free
across heterogeneous series) with rolling-origin backtesting and an optional Diebold-Mariano significance
test; stockout-rate vs. holding-cost tradeoff curves for inventory policies, validated against actual
`Units Sold`; elasticity sign/magnitude sanity checks and historical-range validation for pricing
recommendations.

### Stage 6 — Reporting *(planned)*

Per-challenge write-ups in `reports/`, an `mlflow ui` walkthrough of the logged ARIMA vs. LSTM comparison,
and a closing synthesis note tying the three challenges together: better forecasts → better inventory
decisions → informs pricing constraints.

---

## Reproducing the pipeline

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run `notebooks/full_pipeline_th.ipynb` top to bottom — it covers Stage 1 through the current end of Stage 4a
in one pass (Thai-language explanations throughout). Stage 4b (inventory) and 4c (pricing) will be appended
to the same notebook as later sections rather than split into new files, so the whole project stays in one
place.

Run the test suite before trusting any pipeline output:

```bash
pytest tests/ -q
```

Inspect experiment runs (SARIMA per-series + LSTM training) logged by Stage 4a:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Scope: subset vs. full 100 series

`src/config.py:N_SERIES_SUBSET` controls how many (Store ID, Product ID) series the pipeline runs on.
It starts at `10` (top-10 by historical volume) for fast iteration. Set it to `"all"` to re-run the
identical pipeline on all 100 series — no code changes needed elsewhere.

---

## Status

- [x] Stage 1 — Business understanding (this README)
- [x] Stage 2 — Data understanding / EDA
- [x] Stage 3 — Data preparation (shared feature pipeline)
- [x] Stage 4a — Forecasting (SARIMA vs PyTorch LSTM, MLflow-tracked — LSTM wins, MASE 0.709 vs 0.722, p < 0.001)
- [ ] Stage 4b — Inventory optimization
- [ ] Stage 4c — Dynamic pricing
- [ ] Stage 5 — Evaluation
- [ ] Stage 6 — Reporting / synthesis
