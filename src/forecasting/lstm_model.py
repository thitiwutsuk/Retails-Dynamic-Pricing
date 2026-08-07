"""Stage 4a — deep model: a single global PyTorch LSTM trained across every
series in the current scope, in deliberate contrast with ARIMA's per-series
fitting (src/forecasting/arima_model.py) — the LSTM shares statistical
strength across series instead of learning each one in isolation.

Windowing: for target day `t`, the input window is the `LSTM_WINDOW` days
strictly *before* `t` (never including day `t` itself), so there is no
ambiguity about "known in advance" exogenous variables — only genuinely
past information is ever used to predict a given day.

Series identity is captured implicitly through each series' own windowed
history (lags/rolling stats are already series-specific) plus its
Category/Region one-hot columns, rather than an explicit Store/Product ID
embedding — a deliberate simplification for this project's scope.
"""

import random
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src import config

EXPERIMENT_NAME = "forecasting_lstm"

EXCLUDE_COLS = {config.DATE_COL, "Store ID", "Product ID", "Demand Forecast"}


def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def set_seed(seed: int = config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _assign_split(date: pd.Timestamp) -> str:
    if date <= pd.Timestamp(config.TRAIN_END):
        return "train"
    if date <= pd.Timestamp(config.VAL_END):
        return "val"
    if date <= pd.Timestamp(config.TEST_END):
        return "test"
    return "excluded"


def build_windows(features_df: pd.DataFrame, window: int = config.LSTM_WINDOW):
    """Slice every scoped series into (past-window -> next-day-target) samples.

    Returns a dict of lists keyed by split ("train"/"val"/"test"), each a
    list of dicts: {x: (window, n_features) ndarray, y: float, store, product, date}.
    """
    cols = feature_columns(features_df)
    samples = {"train": [], "val": [], "test": []}

    for (store, product), g in features_df.groupby(config.ID_COLS, observed=True):
        g = g.sort_values(config.DATE_COL).reset_index(drop=True)
        X = g[cols].to_numpy(dtype=float)
        y = g[config.TARGET_COL].to_numpy(dtype=float)
        dates = g[config.DATE_COL].to_numpy()

        for j in range(window, len(g)):
            x_window = X[j - window : j]
            if np.isnan(x_window).any() or np.isnan(y[j]):
                continue
            split = _assign_split(pd.Timestamp(dates[j]))
            if split == "excluded":
                continue
            samples[split].append(
                {
                    "x": x_window,
                    "y": y[j],
                    "store": store,
                    "product": product,
                    "date": pd.Timestamp(dates[j]),
                }
            )

    return samples, cols


def fit_scalers(train_samples: list) -> dict:
    """Fit one (x_scaler, y_scaler) pair per series, using only that series' train windows."""
    by_series = {}
    for s in train_samples:
        by_series.setdefault((s["store"], s["product"]), {"x": [], "y": []})
        by_series[(s["store"], s["product"])]["x"].append(s["x"])
        by_series[(s["store"], s["product"])]["y"].append(s["y"])

    scalers = {}
    for key, data in by_series.items():
        x_all = np.concatenate(data["x"], axis=0)
        y_all = np.array(data["y"]).reshape(-1, 1)
        x_scaler = StandardScaler().fit(x_all)
        y_scaler = StandardScaler().fit(y_all)
        scalers[key] = (x_scaler, y_scaler)
    return scalers


def _scale_samples(samples: list, scalers: dict) -> list:
    scaled = []
    for s in samples:
        key = (s["store"], s["product"])
        if key not in scalers:
            continue
        x_scaler, y_scaler = scalers[key]
        x_shape = s["x"].shape
        x_scaled = x_scaler.transform(s["x"].reshape(-1, x_shape[-1])).reshape(x_shape)
        y_scaled = y_scaler.transform([[s["y"]]])[0, 0]
        scaled.append({**s, "x_scaled": x_scaled.astype("float32"), "y_scaled": np.float32(y_scaled)})
    return scaled


class WindowDataset(Dataset):
    def __init__(self, scaled_samples: list):
        self.samples = scaled_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return torch.from_numpy(s["x_scaled"]), torch.tensor(s["y_scaled"])


class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.fc1 = nn.Linear(hidden2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        last = out[:, -1, :]
        h = self.relu(self.fc1(last))
        return self.fc2(h).squeeze(-1)


def train_lstm(
    train_samples: list,
    val_samples: list,
    scalers: dict,
    n_features: int,
    hidden1: int = 64,
    hidden2: int = 32,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 60,
    patience: int = 8,
    log_to_mlflow: bool = True,
):
    set_seed()

    train_scaled = _scale_samples(train_samples, scalers)
    val_scaled = _scale_samples(val_samples, scalers)

    train_loader = DataLoader(WindowDataset(train_scaled), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WindowDataset(val_scaled), batch_size=batch_size, shuffle=False)

    model = LSTMForecaster(n_features, hidden1, hidden2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.HuberLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    if log_to_mlflow:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        run_ctx = mlflow.start_run(run_name="lstm_global")
        mlflow.log_params(
            {
                "window": config.LSTM_WINDOW,
                "hidden1": hidden1,
                "hidden2": hidden2,
                "lr": lr,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "patience": patience,
                "n_features": n_features,
                "n_series": len(scalers),
            }
        )
    else:
        run_ctx = None

    for epoch in range(max_epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if log_to_mlflow:
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if run_ctx is not None:
        config.LSTM_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_path = config.LSTM_DIR / "lstm.pt"
        torch.save(model.state_dict(), ckpt_path)
        mlflow.log_artifact(str(ckpt_path))
        mlflow.log_metric("best_val_loss", best_val_loss)

    return model, history, run_ctx


def predict(model: nn.Module, samples: list, scalers: dict) -> pd.DataFrame:
    """Run the trained model on `samples`, inverse-scale, return a tidy forecast frame."""
    scaled = _scale_samples(samples, scalers)
    model.eval()

    rows = []
    with torch.no_grad():
        for s in scaled:
            x = torch.from_numpy(s["x_scaled"]).unsqueeze(0)
            pred_scaled = model(x).item()
            _, y_scaler = scalers[(s["store"], s["product"])]
            pred = y_scaler.inverse_transform([[pred_scaled]])[0, 0]
            rows.append(
                {
                    "Store ID": s["store"],
                    "Product ID": s["product"],
                    "Date": s["date"],
                    "y_true": s["y"],
                    "y_pred": max(pred, 0.0),
                    "model": "LSTM",
                }
            )
    return pd.DataFrame(rows).sort_values(config.ID_COLS + [config.DATE_COL]).reset_index(drop=True)


def save_scalers(scalers: dict):
    out_dir = config.LSTM_DIR / "scalers"
    out_dir.mkdir(parents=True, exist_ok=True)
    for (store, product), (x_scaler, y_scaler) in scalers.items():
        joblib.dump(
            {"x_scaler": x_scaler, "y_scaler": y_scaler},
            out_dir / f"{store}_{product}.joblib",
        )
