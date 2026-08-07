import numpy as np
import pandas as pd

from src.data.clean import clean_panel
from src.data.features import engineer_features, select_series


def _toy_raw_panel(n_days=40):
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    rows = []
    rng = np.random.default_rng(0)
    for store in ["S001", "S002"]:
        for product in ["P0001", "P0002"]:
            for i, d in enumerate(dates):
                units_sold = int(rng.integers(0, 50))
                rows.append(
                    {
                        "Date": d,
                        "Store ID": store,
                        "Product ID": product,
                        "Category": "Toys",
                        "Region": "North",
                        "Inventory Level": 200,
                        "Units Sold": units_sold,
                        "Units Ordered": 10,
                        "Demand Forecast": units_sold + 1.0,
                        "Price": 20.0,
                        "Discount": 10,
                        "Weather Condition": "Sunny",
                        "Holiday/Promotion": 0,
                        "Competitor Pricing": 22.0,
                        "Seasonality": "Winter",
                    }
                )
    return pd.DataFrame(rows)


def test_select_series_top_n_by_volume():
    raw = _toy_raw_panel()
    cleaned = clean_panel(raw)
    # force distinct volumes per series
    cleaned.loc[cleaned["Store ID"] == "S001", "Units Sold"] = 100
    cleaned.loc[cleaned["Store ID"] == "S002", "Units Sold"] = 1

    subset = select_series(cleaned, n_series_subset=2)
    kept_series = subset[["Store ID", "Product ID"]].drop_duplicates()
    assert len(kept_series) == 2
    assert set(kept_series["Store ID"]) == {"S001"}


def test_select_series_all_returns_everything():
    raw = _toy_raw_panel()
    cleaned = clean_panel(raw)
    subset = select_series(cleaned, n_series_subset="all")
    assert len(subset) == len(cleaned)


def test_lag_feature_does_not_leak_future():
    raw = _toy_raw_panel()
    cleaned = clean_panel(raw)
    features = engineer_features(cleaned)

    one_series = features[
        (features["Store ID"] == "S001") & (features["Product ID"] == "P0001")
    ].sort_values("Date").reset_index(drop=True)

    # lag_1 at row t must equal Units Sold at row t-1, and be NaN for the
    # very first observation (no prior day exists)
    assert pd.isna(one_series.loc[0, "units_sold_lag_1"])
    for t in range(1, len(one_series)):
        assert one_series.loc[t, "units_sold_lag_1"] == one_series.loc[t - 1, "Units Sold"]


def test_rolling_mean_excludes_current_day():
    raw = _toy_raw_panel()
    cleaned = clean_panel(raw)
    features = engineer_features(cleaned)

    one_series = features[
        (features["Store ID"] == "S001") & (features["Product ID"] == "P0001")
    ].sort_values("Date").reset_index(drop=True)

    # the rolling mean at row t should only be a function of rows < t
    t = 10
    expected = one_series.loc[: t - 1, "Units Sold"].tail(7).mean()
    assert np.isclose(one_series.loc[t, "units_sold_roll_mean_7"], expected)


def test_onehot_columns_created_and_raw_categoricals_dropped():
    raw = _toy_raw_panel()
    cleaned = clean_panel(raw)
    features = engineer_features(cleaned)

    assert "Category_Toys" in features.columns
    assert "Weather Condition_Sunny" in features.columns
    assert "Category" not in features.columns
