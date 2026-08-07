"""Stage 5 (embedded in 4a) — forecast accuracy metrics.

All metrics operate on a tidy frame with columns:
    Store ID, Product ID, Date, y_true, y_pred

MASE is the primary metric for comparing ARIMA vs LSTM across 100
heterogeneous series: it is scale-free (unlike MAE/RMSE) and doesn't blow up
on near-zero volume series (unlike raw MAPE). It scales each series' error
by that same series' in-sample seasonal-naive error, so a MASE < 1 means
"better than seasonal-naive on its own history."
"""

import numpy as np
import pandas as pd

from src import config


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def smape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100)


def seasonal_naive_scale(train_y: np.ndarray, m: int = config.SEASONAL_PERIOD) -> float:
    """In-sample mean absolute seasonal-naive error, used to scale MASE."""
    train_y = np.asarray(train_y, dtype=float)
    if len(train_y) <= m:
        return float(np.mean(np.abs(np.diff(train_y)))) if len(train_y) > 1 else np.nan
    diffs = np.abs(train_y[m:] - train_y[:-m])
    return float(np.mean(diffs))


def mase(y_true, y_pred, scale: float) -> float:
    if scale is None or scale == 0 or np.isnan(scale):
        return np.nan
    return mae(y_true, y_pred) / scale


def compute_scales(train_df: pd.DataFrame, m: int = config.SEASONAL_PERIOD) -> dict:
    """Per-series seasonal-naive scale computed on TRAIN only, keyed by (Store ID, Product ID).

    Shared by every model's MASE calculation so all comparisons use the same
    yardstick regardless of which model produced the forecast.
    """
    scales = {}
    for (store, product), g in train_df.groupby(config.ID_COLS, observed=True):
        y = g.sort_values(config.DATE_COL)[config.TARGET_COL].to_numpy(dtype=float)
        scales[(store, product)] = seasonal_naive_scale(y, m)
    return scales


def evaluate_forecasts(
    df: pd.DataFrame,
    scales: dict,
    by: list | None = None,
) -> pd.DataFrame:
    """Aggregate metrics across series.

    df: columns [Store ID, Product ID, Date, y_true, y_pred]
    scales: dict keyed by (Store ID, Product ID) -> seasonal_naive_scale computed on TRAIN only
    by: optional extra grouping columns (e.g. ["Category"]) present in df, in
        addition to the per-series and overall aggregates.
    """
    rows = []
    for (store, product), g in df.groupby(config.ID_COLS, observed=True):
        scale = scales.get((store, product), np.nan)
        rows.append(
            {
                "Store ID": store,
                "Product ID": product,
                "n_obs": len(g),
                "volume": g["y_true"].sum(),
                "MAE": mae(g["y_true"], g["y_pred"]),
                "RMSE": rmse(g["y_true"], g["y_pred"]),
                "sMAPE": smape(g["y_true"], g["y_pred"]),
                "MASE": mase(g["y_true"], g["y_pred"], scale),
            }
        )
    per_series = pd.DataFrame(rows)

    overall = {
        "MAE": mae(df["y_true"], df["y_pred"]),
        "RMSE": rmse(df["y_true"], df["y_pred"]),
        "sMAPE": smape(df["y_true"], df["y_pred"]),
        "MASE_macro": per_series["MASE"].mean(),
        "MASE_volume_weighted": np.average(
            per_series["MASE"].fillna(per_series["MASE"].mean()),
            weights=per_series["volume"].clip(lower=1),
        ),
    }

    result = {"per_series": per_series, "overall": pd.Series(overall)}

    if by:
        merge_cols = config.ID_COLS + by
        with_group = per_series.merge(
            df[merge_cols].drop_duplicates(), on=config.ID_COLS, how="left"
        )
        result["by_group"] = with_group.groupby(by, observed=True)[
            ["MAE", "RMSE", "sMAPE", "MASE"]
        ].mean()

    return result
