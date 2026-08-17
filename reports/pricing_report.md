# Dynamic Pricing Report

**Question**: given how demand responds to price, discount, and competitor pricing, what price recommendation
would maximize revenue per Category — and how much should that recommendation actually be trusted?

## Method

- Elasticity (`src/pricing/elasticity.py`): log-log OLS per Category —
  `log(Units Sold+1) ~ log(Price) + discount_pct + log(Competitor Pricing) + Holiday/Promotion + Region + Weather Condition + Seasonality`.
  The coefficient on `log(Price)` is the price elasticity of demand.
- Diagnostics: p-values, R², and Variance Inflation Factor (VIF) on the three continuous price-related
  regressors, to check whether the coefficients can be trusted at all before using them.
- Optimization (`src/pricing/optimize.py`): bounded grid search over `price × predicted_demand`, candidates
  restricted to ±20% of the current average price **and** never outside that Category's historically observed
  price range (no extrapolation past what the regression was fit on).

## Results

| Category | Elasticity | p-value | Discount coef. | R² | n |
|---|---|---|---|---|---|
| Toys | −0.245 | 0.425 | 0.141 | 0.4% | 1522 |
| Groceries | −0.045 | 0.887 | −0.362 | 1.6% | 1476 |
| Clothing | +0.381 | 0.235 | −0.023 | 0.7% | 1448 |
| Furniture | +0.406 | 0.186 | 0.348 | 0.8% | 1459 |
| Electronics | **+0.641** | **0.044** | 0.138 | 0.8% | 1405 |

VIF (example category, Toys): `log_price` = 41.0, `discount_pct` = 1.0, `log_competitor_price` = 41.0.

| Category | Current price | Recommended price | Revenue lift |
|---|---|---|---|
| Electronics | 55.60 | 66.72 | +366.5% |
| Clothing | 54.81 | 65.77 | +84.4% |
| Furniture | 54.74 | 65.69 | +66.3% |
| Groceries | 55.65 | 66.78 | +14.7% |
| Toys | 55.52 | 44.42 | n/a |

## Reported honestly: these numbers should not be acted on

Three problems, found and stated directly in the notebook rather than smoothed over:

1. **VIF ≈ 41** for `log_price` and `log_competitor_price` — far above the conventional concern threshold of
   10. Price and Competitor Pricing move together too tightly in this dataset to separate their individual
   effects; the two coefficients are not individually trustworthy.
2. **R² under 2% in every category** — price explains almost none of the variation in `Units Sold` in this
   dataset. Something else (likely inherent to how the synthetic dataset was generated) dominates.
3. **3 of 5 categories return a *positive* elasticity** (price up, predicted demand also up) — backwards from
   basic demand theory — and the one category with a statistically significant result (Electronics, p = 0.044)
   is also positive, which is itself a red flag rather than a reassurance.

**Conclusion**: the pipeline (fit → diagnose → optimize → guardrail against extrapolation) is built and works
correctly end to end, but the elasticity estimates and price recommendations it produces on *this* dataset are
not reliable enough to act on. This is a methodology demonstration, not a production-ready recommendation. A
concrete fix path: drop one of the two collinear price variables (keep `log_price`, drop
`log_competitor_price`), or replace both with `price_gap` (their difference) to remove the collinearity
directly; ideally validate against a real pricing experiment (A/B test) rather than observational data alone,
given the well-known endogeneity risk in price-vs-demand regressions (retailers often change price *because*
demand shifted, which biases a plain OLS elasticity estimate).

## Reproduce

`notebooks/retail_demand_forecasting.ipynb`, Stage 4c. Output: `data/processed/pricing_recommendations.csv`,
`reports/figures/{pricing_elasticity_by_category,pricing_revenue_curve}.png`.
