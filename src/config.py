"""Shared paths, constants, and scope control for the whole project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw" / "retail_store_inventory.csv"
DATA_INTERIM = ROOT / "data" / "interim" / "cleaned_panel.parquet"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_FULL = DATA_PROCESSED_DIR / "features_full.parquet"
FORECASTS_DIR = DATA_PROCESSED_DIR / "forecasts"

MODELS_DIR = ROOT / "models"
ARIMA_DIR = MODELS_DIR / "arima"
LSTM_DIR = MODELS_DIR / "lstm"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MLFLOW_TRACKING_URI = f"file:{ROOT / 'mlruns'}"

RANDOM_SEED = 42

# Target variable — never use the dataset's own "Demand Forecast" column as
# a feature or target: it is an external benchmark to beat, not ground truth.
TARGET_COL = "Units Sold"

ID_COLS = ["Store ID", "Product ID"]
DATE_COL = "Date"

# Chronological, non-overlapping split shared by every downstream stage
# (forecasting, inventory simulation, pricing evaluation).
TRAIN_END = "2023-06-30"
VAL_END = "2023-09-30"
TEST_END = "2024-01-01"

# --- Scope control -------------------------------------------------------
# Start on a subset of (Store ID, Product ID) series for fast iteration,
# then switch to "all" to run the identical pipeline on all 100 series.
# Either a list of (store_id, product_id) tuples, an int N (top-N by total
# historical Units Sold), or the string "all".
N_SERIES_SUBSET = 10

# Weekly seasonality (7-day cycle) used by SARIMA and the LSTM windowing.
SEASONAL_PERIOD = 7
LSTM_WINDOW = 28
FORECAST_HORIZONS = (1, 7)
