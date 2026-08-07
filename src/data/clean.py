"""Stage 3 — cleaning: dtype casting and range validation of the raw panel."""

import pandas as pd

from src import config

CATEGORICAL_COLS = [
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality",
]


def clean_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Cast dtypes and validate ranges on the raw retail panel.

    Assumes `df` already passed the Stage 2 audit (full dense grid, no
    duplicates, no missing values) — this function only casts types and
    checks value ranges, it does not fix structural issues.
    """
    df = df.copy()

    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    df["Holiday/Promotion"] = df["Holiday/Promotion"].astype("int8")

    range_checks = {
        "Inventory Level": df["Inventory Level"] >= 0,
        "Units Sold": df["Units Sold"] >= 0,
        "Units Ordered": df["Units Ordered"] >= 0,
        "Discount": df["Discount"].between(0, 100),
        "Price": df["Price"] > 0,
        "Competitor Pricing": df["Competitor Pricing"] > 0,
    }
    for name, mask in range_checks.items():
        if not mask.all():
            raise ValueError(f"range check failed for column: {name}")

    df = df.sort_values(config.ID_COLS + [config.DATE_COL]).reset_index(drop=True)
    return df
