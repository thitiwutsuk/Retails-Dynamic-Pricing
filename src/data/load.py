"""Stage 3 — loading helpers shared across notebooks and src/ modules."""

import pandas as pd

from src import config


def read_raw() -> pd.DataFrame:
    return pd.read_csv(config.DATA_RAW, parse_dates=[config.DATE_COL])


def read_cleaned() -> pd.DataFrame:
    return pd.read_parquet(config.DATA_INTERIM)


def read_features() -> pd.DataFrame:
    return pd.read_parquet(config.FEATURES_FULL)
