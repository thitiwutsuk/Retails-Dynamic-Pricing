"""Stage 4a/5 — lightweight backtesting robustness check.

Both ARIMA (`arima_model.py`) and the LSTM (`lstm_model.py`) already produce
genuine one-step-ahead forecasts for *every day* of the test window (each
day's forecast uses true history only, walked forward day by day / window by
window) — this is already a much stronger evaluation than a single train/test
split. `split_into_origins` turns that single evaluation into a lightweight
"rolling-origin" robustness check by splitting the already-computed test-set
forecasts into contiguous chronological chunks and scoring each chunk
separately, so we can see whether error is stable across the test window
rather than concentrated in one sub-period — without the cost of re-fitting
either model multiple times.
"""

import numpy as np
import pandas as pd

from src import config
from src.forecasting.evaluate import evaluate_forecasts


def split_into_origins(forecasts_df: pd.DataFrame, n_origins: int = 2) -> list:
    """Split a [Store ID, Product ID, Date, y_true, y_pred] frame into
    `n_origins` contiguous chronological chunks (by unique Date)."""
    dates = np.sort(forecasts_df[config.DATE_COL].unique())
    chunks = np.array_split(dates, n_origins)
    return [forecasts_df[forecasts_df[config.DATE_COL].isin(chunk)] for chunk in chunks]


def backtest_summary(forecasts_df: pd.DataFrame, scales: dict, n_origins: int = 2) -> pd.DataFrame:
    """Per-origin overall MAE/RMSE/sMAPE/MASE, to check error stability over the test window."""
    origins = split_into_origins(forecasts_df, n_origins)
    rows = []
    for i, chunk in enumerate(origins, start=1):
        result = evaluate_forecasts(chunk, scales)
        overall = result["overall"]
        rows.append(
            {
                "origin": i,
                "start_date": chunk[config.DATE_COL].min(),
                "end_date": chunk[config.DATE_COL].max(),
                "MAE": overall["MAE"],
                "RMSE": overall["RMSE"],
                "sMAPE": overall["sMAPE"],
                "MASE_macro": overall["MASE_macro"],
            }
        )
    return pd.DataFrame(rows)
