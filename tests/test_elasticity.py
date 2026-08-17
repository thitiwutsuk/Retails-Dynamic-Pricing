import numpy as np
import pandas as pd

from src.pricing.elasticity import fit_elasticity_by_category, elasticity_summary, build_design_matrix
from src.pricing.optimize import optimize_price


def _synthetic_category_df(n=2000, true_elasticity=-1.5, seed=0):
    rng = np.random.default_rng(seed)
    price = rng.uniform(20, 40, n)
    competitor = price + rng.normal(0, 2, n)
    discount = rng.integers(0, 30, n)
    holiday = rng.integers(0, 2, n)
    region = rng.choice(["North", "South"], n)
    weather = rng.choice(["Sunny", "Rainy"], n)
    season = rng.choice(["Summer", "Winter"], n)

    # intercept chosen so log1p(units_sold) stays comfortably positive
    # across the whole price range used here
    log_demand = 10 + true_elasticity * np.log(price) + 0.01 * discount + rng.normal(0, 0.05, n)
    units_sold = np.expm1(log_demand)

    return pd.DataFrame(
        {
            "Units Sold": units_sold,
            "Price": price,
            "Competitor Pricing": competitor,
            "Discount": discount,
            "Holiday/Promotion": holiday,
            "Region": region,
            "Weather Condition": weather,
            "Seasonality": season,
            "Category": "Toys",
        }
    )


def test_recovers_negative_elasticity_sign_and_magnitude():
    df = _synthetic_category_df(true_elasticity=-1.5)
    models = fit_elasticity_by_category(df, min_rows=50)
    summary = elasticity_summary(models)

    assert "Toys" in models
    fitted = summary.loc[summary["Category"] == "Toys", "elasticity"].iloc[0]
    assert fitted < 0, "elasticity should be negative: higher price -> lower demand"
    assert np.isclose(fitted, -1.5, atol=0.15)


def test_skips_categories_below_min_rows():
    df = _synthetic_category_df(n=10)
    models = fit_elasticity_by_category(df, min_rows=50)
    assert models == {}


def test_design_matrix_has_expected_columns():
    df = _synthetic_category_df(n=200)
    X, y = build_design_matrix(df)
    assert "log_price" in X.columns
    assert "discount_pct" in X.columns
    assert "log_competitor_price" in X.columns
    assert len(y) == len(df)


def test_optimize_price_stays_within_historical_bounds():
    df = _synthetic_category_df(n=2000)
    models = fit_elasticity_by_category(df, min_rows=50)
    result = optimize_price(models["Toys"], df)

    low, high = result["price_bounds"]
    assert df["Price"].min() <= low
    assert high <= df["Price"].max()
    assert low <= result["recommended_price"] <= high
