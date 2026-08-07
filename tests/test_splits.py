import pandas as pd

from src import config
from src.forecasting.splits import time_split


def _toy_panel():
    dates = pd.date_range("2022-01-01", "2022-01-20", freq="D")
    rows = []
    for store in ["S001", "S002"]:
        for i, d in enumerate(dates):
            rows.append({"Store ID": store, "Product ID": "P0001", "Date": d, "Units Sold": i})
    return pd.DataFrame(rows)


def test_split_covers_all_rows_with_no_overlap():
    df = _toy_panel()
    train, val, test = time_split(df, train_end="2022-01-10", val_end="2022-01-15", test_end="2022-01-20")

    assert len(train) + len(val) + len(test) == len(df)

    assert train[config.DATE_COL].max() <= pd.Timestamp("2022-01-10")
    assert val[config.DATE_COL].min() > pd.Timestamp("2022-01-10")
    assert val[config.DATE_COL].max() <= pd.Timestamp("2022-01-15")
    assert test[config.DATE_COL].min() > pd.Timestamp("2022-01-15")
    assert test[config.DATE_COL].max() <= pd.Timestamp("2022-01-20")


def test_split_boundaries_do_not_overlap():
    df = _toy_panel()
    train, val, test = time_split(df, train_end="2022-01-10", val_end="2022-01-15", test_end="2022-01-20")

    train_dates = set(train[config.DATE_COL])
    val_dates = set(val[config.DATE_COL])
    test_dates = set(test[config.DATE_COL])

    assert train_dates.isdisjoint(val_dates)
    assert val_dates.isdisjoint(test_dates)
    assert train_dates.isdisjoint(test_dates)


def test_split_is_chronological_per_series():
    df = _toy_panel()
    train, val, _ = time_split(df, train_end="2022-01-10", val_end="2022-01-15", test_end="2022-01-20")

    for store in df["Store ID"].unique():
        train_max = train.loc[train["Store ID"] == store, config.DATE_COL].max()
        val_min = val.loc[val["Store ID"] == store, config.DATE_COL].min()
        assert train_max < val_min
