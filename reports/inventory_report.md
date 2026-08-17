# Inventory Optimization Report

**Question**: given a demand forecast (and its uncertainty), what reorder-point / order-quantity policy
minimizes stockouts without excessive overstock — and does a better forecast (LSTM, per the forecasting
report) actually translate into a better inventory outcome?

## Method

- Policies (`src/inventory/policies.py`): Safety Stock `SS = z·σ_d·√lead_time`, Reorder Point
  `ROP = avg_daily_demand·lead_time + SS`, `EOQ = √(2·D·S/H)`.
- Explicit assumptions (the raw dataset has no cost/lead-time columns): lead time = 5 days, ordering cost =
  $50/order, holding cost = 20% of unit Price per year, service level = 95% (z = 1.65).
- Simulation (`src/inventory/simulate.py`): day-step replay of the test window (92 days × 10 series) against
  **actual realized demand**, comparing three policies:
  1. **Historical (actual)** — read straight off the data, no simulation; the retailer's own real behavior.
  2. **Naive-forecast-driven** ROP/EOQ, using the Naive baseline's forecast statistics.
  3. **LSTM-forecast-driven** ROP/EOQ, using the (better-MASE) LSTM forecast's statistics.

## Results

| Policy | Avg. stockout rate | Avg. fill rate | Avg. inventory | Avg. holding cost | Orders placed |
|---|---|---|---|---|---|
| Historical (actual) | 5.6% | — | 279.6 | 794.2 | 930 |
| Naive-forecast-driven | 11.6% | 90.7% | 467.5 | 1322.5 | 189 |
| **LSTM-forecast-driven** | **14.4%** | 88.4% | 400.3 | **1135.0** | 189 |

Service-level sensitivity (LSTM-driven policy):

| Service level | Avg. stockout rate | Avg. holding cost |
|---|---|---|
| 90% | 15.1% | 1091.5 |
| 95% | 14.4% | 1135.0 |
| 99% | 14.3% | 1138.8 |

## An honest, counter-intuitive finding

The LSTM-driven policy has a **higher** stockout rate than the Naive-driven one, despite LSTM having the
better forecast accuracy (MASE) in the forecasting report. Diagnosed directly (`data/processed/
inventory_forecast_bias_check.csv`) rather than left unexplained:

| Forecast | Avg. daily demand | Avg. σ (residual std) | Avg. bias (pred − true) |
|---|---|---|---|
| Naive | 144.11 | 151.79 | −0.05 |
| LSTM | 135.60 | 109.72 | **−8.55** |

True average demand across the scoped series: **144.2**.

Naive's forecast is essentially unbiased (it's yesterday's actual value, which averages out to the true mean
over time) but very noisy — that noise inflates its safety-stock term, which *incidentally* protects it from
stockouts at the cost of the highest holding cost of the three policies. LSTM's day-to-day predictions are
genuinely tighter (lower σ), but its *average* forecast is biased about 6% low — a known behavior of
Huber/MSE-trained sequence models on spiky demand (they hedge toward the mean rather than chasing spikes).
Because the reorder point is driven directly by the forecast's mean, that bias undersizes it regardless of
how tight the day-to-day error is.

**Takeaway**: a lower MASE does not certify a forecast as safe to drive inventory decisions on its own — the
forecast's bias needs to be checked and corrected (e.g. a simple additive bias-correction term) before it is
used to set reorder points. This is the single most actionable engineering finding from this stage.

## Reproduce

`notebooks/retail_demand_forecasting.ipynb`, Stage 4b. Outputs: `data/processed/
{inventory_policy_comparison,inventory_sensitivity,inventory_forecast_bias_check}.csv`,
`reports/figures/inventory_policy_comparison.png`, `reports/figures/inventory_sensitivity_tradeoff.png`.
