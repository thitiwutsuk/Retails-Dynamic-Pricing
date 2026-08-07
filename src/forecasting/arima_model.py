"""Stage 4a — classical baseline: per-series SARIMA via pmdarima.auto_arima.

Fit independently per (Store ID, Product ID) series (unlike the LSTM, which
is trained as one global model) — this asymmetry is deliberate and is part
of the ARIMA-vs-LSTM comparison narrative: ARIMA can't share statistical
strength across series, LSTM can.

Evaluation protocol: **one-step-ahead with true history**. The model is
fit once on train, then walked forward through the test period: at each
day it forecasts 1 step ahead, then is updated with that day's *true*
observed value (via pmdarima's incremental `.update()`, which does not
refit but does update the model's internal state) before forecasting the
next day. This matches exactly how the LSTM is evaluated (its lag/rolling
features at each test day are computed from true history, not from its own
past predictions) — apples-to-apples, not "ARIMA gets easier true-history
input while LSTM gets a harder recursive rollout" or vice versa.
"""

import mlflow
import numpy as np
import pandas as pd
import pmdarima as pm

from src import config
from src.forecasting.evaluate import mae, rmse, mase, seasonal_naive_scale

EXPERIMENT_NAME = "forecasting_arima"


def fit_predict_series(train_y: np.ndarray, test_y: np.ndarray, m: int = config.SEASONAL_PERIOD):
    """Fit on train_y, then one-step-ahead walk-forward through test_y using true history."""
    model = pm.auto_arima(
        train_y,
        seasonal=True,
        m=m,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=3,
        max_q=3,
        max_P=2,
        max_Q=2,
    )

    preds = np.empty(len(test_y))
    for i, true_val in enumerate(test_y):
        preds[i] = model.predict(n_periods=1)[0]
        model.update([true_val])

    return preds, model


def run_arima_all_series(
    train: pd.DataFrame,
    test: pd.DataFrame,
    log_to_mlflow: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit + one-step-ahead evaluate SARIMA per series in `train`'s scope.

    Returns (forecasts_df, orders_df).
    """
    if log_to_mlflow:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

    forecast_rows = []
    order_rows = []

    series_keys = list(train[config.ID_COLS].drop_duplicates().itertuples(index=False, name=None))

    for store, product in series_keys:
        train_g = train[
            (train["Store ID"] == store) & (train["Product ID"] == product)
        ].sort_values(config.DATE_COL)
        test_g = test[
            (test["Store ID"] == store) & (test["Product ID"] == product)
        ].sort_values(config.DATE_COL)

        train_y = train_g[config.TARGET_COL].to_numpy(dtype=float)
        test_y = test_g[config.TARGET_COL].to_numpy(dtype=float)

        y_pred, model = fit_predict_series(train_y, test_y)
        y_pred = np.clip(y_pred, a_min=0, a_max=None)

        scale = seasonal_naive_scale(train_y)
        series_mae = mae(test_y, y_pred)
        series_rmse = rmse(test_y, y_pred)
        series_mase = mase(test_y, y_pred, scale)

        order = model.order
        seasonal_order = model.seasonal_order

        if log_to_mlflow:
            with mlflow.start_run(run_name=f"{store}_{product}"):
                mlflow.log_params(
                    {
                        "store": store,
                        "product": product,
                        "order": str(order),
                        "seasonal_order": str(seasonal_order),
                    }
                )
                mlflow.log_metrics(
                    {"MAE": series_mae, "RMSE": series_rmse, "MASE": series_mase}
                )

        for date, yt, yp in zip(test_g[config.DATE_COL], test_y, y_pred):
            forecast_rows.append(
                {
                    "Store ID": store,
                    "Product ID": product,
                    "Date": date,
                    "y_true": yt,
                    "y_pred": yp,
                    "model": "SARIMA",
                }
            )

        order_rows.append(
            {
                "Store ID": store,
                "Product ID": product,
                "order": str(order),
                "seasonal_order": str(seasonal_order),
                "MAE": series_mae,
                "RMSE": series_rmse,
                "MASE": series_mase,
            }
        )

    forecasts_df = pd.DataFrame(forecast_rows)
    orders_df = pd.DataFrame(order_rows)
    return forecasts_df, orders_df
