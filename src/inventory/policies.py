"""Stage 4b — inventory policy formulas: safety stock, reorder point, EOQ.

Assumptions made explicit here (no lead-time or cost data exists in the raw
dataset, so these are documented, reasonable stand-ins rather than fitted
values):
    LEAD_TIME_DAYS      = 5      days between placing an order and receiving it
    ORDERING_COST       = 50     currency units per order placed (fixed cost)
    HOLDING_COST_RATE   = 0.20   fraction of unit Price held as annual cost
    SERVICE_LEVEL_Z     = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}  z-scores

All three policy numbers are computed from a single-series demand forecast
(its mean and its residual std, i.e. forecast uncertainty) plus that
series' own average Price.
"""

import numpy as np

LEAD_TIME_DAYS = 5
ORDERING_COST = 50.0
HOLDING_COST_RATE = 0.20

SERVICE_LEVEL_Z = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}


def safety_stock(sigma_d: float, lead_time_days: float = LEAD_TIME_DAYS, z: float = 1.65) -> float:
    """SS = z * sigma_d * sqrt(lead_time)."""
    return z * sigma_d * np.sqrt(lead_time_days)


def reorder_point(avg_daily_demand: float, lead_time_days: float, ss: float) -> float:
    """ROP = avg_daily_demand * lead_time + safety_stock."""
    return avg_daily_demand * lead_time_days + ss


def eoq(annual_demand: float, ordering_cost: float, holding_cost_per_unit_per_year: float) -> float:
    """Economic Order Quantity = sqrt(2 * D * S / H)."""
    if holding_cost_per_unit_per_year <= 0:
        return annual_demand / 12  # degenerate fallback, avoid div-by-zero
    return np.sqrt(2 * annual_demand * ordering_cost / holding_cost_per_unit_per_year)


def compute_policy_params(
    avg_daily_demand: float,
    sigma_d: float,
    unit_price: float,
    lead_time_days: float = LEAD_TIME_DAYS,
    ordering_cost: float = ORDERING_COST,
    holding_cost_rate: float = HOLDING_COST_RATE,
    service_level: float = 0.95,
) -> dict:
    """Bundle SS / ROP / EOQ for one series given its demand forecast stats."""
    z = SERVICE_LEVEL_Z[service_level]
    ss = safety_stock(sigma_d, lead_time_days, z)
    rop = reorder_point(avg_daily_demand, lead_time_days, ss)
    annual_demand = avg_daily_demand * 365
    holding_cost_per_unit_per_year = holding_cost_rate * unit_price
    order_qty = eoq(annual_demand, ordering_cost, holding_cost_per_unit_per_year)
    return {
        "avg_daily_demand": avg_daily_demand,
        "sigma_d": sigma_d,
        "unit_price": unit_price,
        "service_level": service_level,
        "z": z,
        "safety_stock": ss,
        "reorder_point": rop,
        "eoq": order_qty,
    }
