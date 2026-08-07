"""Stage 3 — shared feature engineering pipeline.

Consumed identically by Forecasting (Track 1), Inventory (Track 2), and
Pricing (Track 3) so the three challenges never diverge on feature
definitions. Everything here is computed per (Store ID, Product ID) group,
sorted by Date, with rolling/lag windows shifted to avoid leaking the
future into the present.
"""

import numpy as np
import pandas as pd

from src import config

CATEGORICAL_ONEHOT_COLS = ["Category", "Region", "Weather Condition", "Seasonality"]

LAG_DAYS = (1, 7, 14, 28)
ROLLING_WINDOWS = (7, 14, 28)


def select_series(df: pd.DataFrame, n_series_subset=None) -> pd.DataFrame:
    """Filter the panel down to the configured scope.

    n_series_subset:
        - "all": no filtering, use every (Store ID, Product ID) pair
        - int N: keep the top-N series by total historical Units Sold
        - list of (store_id, product_id) tuples: keep exactly those series
        - None: read the value from src.config.N_SERIES_SUBSET
    """
    if n_series_subset is None:
        n_series_subset = config.N_SERIES_SUBSET

    if n_series_subset == "all":
        return df

    if isinstance(n_series_subset, int):
        volume = (
            df.groupby(config.ID_COLS, observed=True)["Units Sold"]
            .sum()
            .sort_values(ascending=False)
        )
        keep = volume.head(n_series_subset).index
    else:
        keep = pd.MultiIndex.from_tuples(n_series_subset, names=config.ID_COLS)

    idx = pd.MultiIndex.from_frame(df[config.ID_COLS])
    return df.loc[idx.isin(keep)].reset_index(drop=True)


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df[config.DATE_COL].dt
    df["day_of_week"] = dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int8")
    df["month"] = dt.month
    df["week_of_year"] = dt.isocalendar().week.astype("int32")
    df["day_of_year"] = dt.dayofyear

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def _add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(config.ID_COLS, observed=True, group_keys=False)

    for lag in LAG_DAYS:
        df[f"units_sold_lag_{lag}"] = grp["Units Sold"].shift(lag)

    # shift(1) inside each group's transform so the current day's own value
    # never leaks into its own rolling window
    for window in ROLLING_WINDOWS:
        df[f"units_sold_roll_mean_{window}"] = grp["Units Sold"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f"units_sold_roll_std_{window}"] = grp["Units Sold"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=2).std()
        )

    df["inventory_roll_mean_7"] = grp["Inventory Level"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )
    df["discount_roll_mean_7"] = grp["Discount"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )

    return df


def _add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df["price_gap"] = df["Price"] - df["Competitor Pricing"]
    df["price_gap_pct"] = df["price_gap"] / df["Competitor Pricing"]
    df["discount_pct"] = df["Discount"] / 100.0
    df["effective_price"] = df["Price"] * (1 - df["discount_pct"])
    return df


def _add_inventory_features(df: pd.DataFrame) -> pd.DataFrame:
    denom = df["units_sold_roll_mean_7"].replace(0, np.nan)
    df["days_of_supply"] = df["Inventory Level"] / denom
    df["stockout_flag"] = (
        (df["Units Sold"] >= 0.95 * df["Inventory Level"]) & (df["Inventory Level"] > 0)
    ).astype("int8")
    return df


def _add_onehot(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=CATEGORICAL_ONEHOT_COLS, prefix=CATEGORICAL_ONEHOT_COLS)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full shared feature set to a cleaned panel.

    `df` must already be the output of `src.data.clean.clean_panel` (typed,
    validated, sorted by ID_COLS + Date).
    """
    df = df.sort_values(config.ID_COLS + [config.DATE_COL]).reset_index(drop=True)

    df = _add_calendar_features(df)
    df = _add_lag_and_rolling_features(df)
    df = _add_price_features(df)
    df = _add_inventory_features(df)
    df = _add_onehot(df)

    return df


def build_features_full(n_series_subset=None) -> pd.DataFrame:
    """Orchestrate Stage 3: load cleaned panel -> scope -> feature engineer -> save."""
    from src.data.load import read_cleaned

    cleaned = read_cleaned()
    scoped = select_series(cleaned, n_series_subset)
    features = engineer_features(scoped)

    config.FEATURES_FULL.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(config.FEATURES_FULL, index=False)
    return features


if __name__ == "__main__":
    out = build_features_full()
    print(f"saved {config.FEATURES_FULL} shape={out.shape}")
