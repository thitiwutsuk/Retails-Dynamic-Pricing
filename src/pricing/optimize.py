"""Stage 4c — revenue-maximizing price search.

Given a fitted per-Category elasticity model, predict demand at candidate
prices (holding every other feature at that category's mean/mode) and pick
the price maximizing `price * predicted_demand`. Uses a bounded grid search
(rather than a numerical optimizer) so the whole revenue curve is available
to plot and sanity-check, not just the optimum.

Two guardrails against unrealistic recommendations:
    - the candidate price range never leaves the category's *observed*
      historical [min, max] Price (avoids extrapolating the regression
      outside the data it was fit on)
    - candidates are additionally capped to +/- `price_band_pct` of the
      category's current average price
"""

import numpy as np
import pandas as pd

from src.pricing.elasticity import build_design_matrix, NUMERIC_FEATURES, CONTROL_CATEGORICALS


def _representative_row(df: pd.DataFrame) -> pd.Series:
    """A single 'typical day' row for this category: mean of numeric
    features, most common category value for each control."""
    row = df.iloc[0:1].copy()
    row["Discount"] = df["Discount"].mean()
    row["Competitor Pricing"] = df["Competitor Pricing"].mean()
    row["Holiday/Promotion"] = df["Holiday/Promotion"].mean()
    for col in CONTROL_CATEGORICALS:
        row[col] = df[col].mode().iloc[0]
    return row


def predict_demand_at_price(model, representative_row: pd.DataFrame, price: float) -> float:
    row = representative_row.copy()
    row["Price"] = price
    X, _ = build_design_matrix(row)
    # a single-row design matrix can't reproduce every dummy column the model
    # was fit on (a category with only one observed level here just vanishes
    # from get_dummies) -- align to the model's own columns, missing ones = 0
    X = X.reindex(columns=model.params.index, fill_value=0.0)
    # statsmodels echoes X's own row index on the predicted Series, which is
    # almost never 0 once X comes from a slice of a larger frame -- index
    # positionally instead of by label to avoid a KeyError
    log_demand = np.asarray(model.predict(X)).ravel()[0]
    return float(np.expm1(log_demand))


def optimize_price(
    model,
    category_df: pd.DataFrame,
    n_grid: int = 41,
    price_band_pct: float = 0.20,
) -> dict:
    """Grid-search the revenue-maximizing price for one category.

    Returns the full revenue curve plus the recommended price and the
    revenue lift vs. the category's current average price.
    """
    rep_row = _representative_row(category_df)

    current_price = float(category_df["Price"].mean())
    hist_min, hist_max = float(category_df["Price"].min()), float(category_df["Price"].max())

    band_low = current_price * (1 - price_band_pct)
    band_high = current_price * (1 + price_band_pct)
    low = max(band_low, hist_min)
    high = min(band_high, hist_max)

    price_grid = np.linspace(low, high, n_grid)
    demand_grid = np.array([predict_demand_at_price(model, rep_row, p) for p in price_grid])
    revenue_grid = price_grid * demand_grid

    curve = pd.DataFrame({"price": price_grid, "predicted_demand": demand_grid, "revenue": revenue_grid})

    best_idx = int(np.argmax(revenue_grid))
    best_price = float(price_grid[best_idx])
    best_revenue = float(revenue_grid[best_idx])

    current_demand = predict_demand_at_price(model, rep_row, current_price)
    current_revenue = current_price * current_demand

    revenue_lift_pct = (
        (best_revenue - current_revenue) / current_revenue * 100 if current_revenue > 0 else np.nan
    )

    return {
        "current_price": current_price,
        "current_revenue": current_revenue,
        "recommended_price": best_price,
        "recommended_revenue": best_revenue,
        "revenue_lift_pct": revenue_lift_pct,
        "price_bounds": (low, high),
        "curve": curve,
    }
