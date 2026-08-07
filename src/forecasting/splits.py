"""Stage 3 — chronological, non-overlapping train/val/test split.

Identical cut dates applied to every (Store ID, Product ID) series, no
shuffling. Reused as-is by the Forecasting, Inventory simulation, and
Pricing evaluation stages so all three challenges are judged on the same
held-out window.
"""

import pandas as pd

from src import config


def time_split(
    df: pd.DataFrame,
    train_end: str = config.TRAIN_END,
    val_end: str = config.VAL_END,
    test_end: str = config.TEST_END,
):
    """Split a panel chronologically into (train, val, test).

    train:  Date <= train_end
    val:    train_end < Date <= val_end
    test:   val_end < Date <= test_end
    """
    dates = df[config.DATE_COL]
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    test_end_ts = pd.Timestamp(test_end)

    train = df.loc[dates <= train_end_ts].reset_index(drop=True)
    val = df.loc[(dates > train_end_ts) & (dates <= val_end_ts)].reset_index(drop=True)
    test = df.loc[(dates > val_end_ts) & (dates <= test_end_ts)].reset_index(drop=True)

    return train, val, test
