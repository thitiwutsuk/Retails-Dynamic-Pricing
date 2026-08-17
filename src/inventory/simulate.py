"""Stage 4b — day-step inventory simulation.

Replays the test window one day at a time against **actual realized demand**
(`Units Sold`, i.e. what really happened), so a policy's stockout/holding
numbers reflect how it would have performed against reality, not against its
own forecast. Two things are compared:

- `simulate_rop_eoq`: an (s, Q) reorder-point / EOQ policy — reorders `eoq`
  units whenever inventory drops to/below `reorder_point`, arriving after
  `lead_time_days`. Only one order is ever outstanding at a time (a second
  low-inventory day while an order is already in transit does not trigger a
  duplicate order).
- `historical_baseline_metrics`: no simulation at all — reads the stockout
  proxy and inventory level the retailer's own historical policy actually
  produced, straight from the data, as the baseline every built policy is
  judged against.

Both report the same metrics (stockout rate, fill rate, avg inventory,
approximate holding cost) so they are directly comparable in one table.
"""

import numpy as np
import pandas as pd

from src import config
from src.inventory.policies import HOLDING_COST_RATE


def simulate_rop_eoq(
    series_df: pd.DataFrame,
    initial_inventory: float,
    reorder_point: float,
    eoq: float,
    lead_time_days: int = 5,
    unit_price: float = 1.0,
) -> dict:
    """Day-step (s, Q) simulation over one series' test-period rows.

    series_df: one series, sorted by Date, with a `Units Sold` column
        (actual realized demand used to drive the simulation).
    """
    dates = series_df[config.DATE_COL].to_numpy()
    demand = series_df["Units Sold"].to_numpy(dtype=float)
    n_days = len(demand)

    inventory = float(initial_inventory)
    pending_arrival_day = None  # index of the day an outstanding order arrives, or None
    pending_qty = 0.0

    stockout_days = 0
    total_demand = 0.0
    total_fulfilled = 0.0
    orders_placed = 0
    inventory_trace = np.empty(n_days)

    for i in range(n_days):
        if pending_arrival_day is not None and i == pending_arrival_day:
            inventory += pending_qty
            pending_arrival_day = None
            pending_qty = 0.0

        d = demand[i]
        fulfilled = min(d, inventory)
        lost = d - fulfilled
        if lost > 0:
            stockout_days += 1
        inventory -= fulfilled

        total_demand += d
        total_fulfilled += fulfilled
        inventory_trace[i] = inventory

        if inventory <= reorder_point and pending_arrival_day is None:
            pending_arrival_day = i + lead_time_days
            pending_qty = eoq
            orders_placed += 1

    avg_inventory = float(inventory_trace.mean())
    holding_cost = avg_inventory * HOLDING_COST_RATE * unit_price * (n_days / 365)

    return {
        "n_days": n_days,
        "stockout_days": stockout_days,
        "stockout_rate": stockout_days / n_days,
        "fill_rate": total_fulfilled / total_demand if total_demand > 0 else np.nan,
        "avg_inventory": avg_inventory,
        "orders_placed": orders_placed,
        "holding_cost": holding_cost,
        "lost_units": total_demand - total_fulfilled,
    }


def historical_baseline_metrics(series_df: pd.DataFrame, unit_price: float = 1.0) -> dict:
    """Metrics read directly off the retailer's own historical behavior — no
    simulation, this is simply what already happened in the data."""
    n_days = len(series_df)
    stockout_days = int(series_df["stockout_flag"].sum()) if "stockout_flag" in series_df else int(
        ((series_df["Units Sold"] >= 0.95 * series_df["Inventory Level"]) & (series_df["Inventory Level"] > 0)).sum()
    )
    avg_inventory = float(series_df["Inventory Level"].mean())
    holding_cost = avg_inventory * HOLDING_COST_RATE * unit_price * (n_days / 365)

    return {
        "n_days": n_days,
        "stockout_days": stockout_days,
        "stockout_rate": stockout_days / n_days,
        "fill_rate": np.nan,  # unobservable: true unconstrained demand isn't in the data
        "avg_inventory": avg_inventory,
        "orders_placed": int((series_df["Units Ordered"] > 0).sum()),
        "holding_cost": holding_cost,
        "lost_units": np.nan,
    }
