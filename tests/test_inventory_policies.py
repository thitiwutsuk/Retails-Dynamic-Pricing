import numpy as np

from src.inventory.policies import safety_stock, reorder_point, eoq, compute_policy_params


def test_eoq_matches_hand_calculation():
    # classic textbook EOQ example: D=1000/yr, S=$50/order, H=$2/unit/yr
    # EOQ = sqrt(2*1000*50/2) = sqrt(50000) = 223.6067...
    result = eoq(annual_demand=1000, ordering_cost=50, holding_cost_per_unit_per_year=2)
    assert np.isclose(result, np.sqrt(50000), rtol=1e-9)


def test_eoq_increases_with_demand_and_decreases_with_holding_cost():
    base = eoq(annual_demand=1000, ordering_cost=50, holding_cost_per_unit_per_year=2)
    higher_demand = eoq(annual_demand=4000, ordering_cost=50, holding_cost_per_unit_per_year=2)
    higher_holding = eoq(annual_demand=1000, ordering_cost=50, holding_cost_per_unit_per_year=8)
    assert higher_demand > base
    assert higher_holding < base


def test_safety_stock_matches_hand_calculation():
    # SS = z * sigma_d * sqrt(lead_time) = 1.65 * 10 * sqrt(4) = 33.0
    result = safety_stock(sigma_d=10, lead_time_days=4, z=1.65)
    assert np.isclose(result, 33.0, rtol=1e-9)


def test_reorder_point_matches_hand_calculation():
    # ROP = avg_daily_demand * lead_time + SS = 20*4 + 33 = 113
    ss = safety_stock(sigma_d=10, lead_time_days=4, z=1.65)
    result = reorder_point(avg_daily_demand=20, lead_time_days=4, ss=ss)
    assert np.isclose(result, 20 * 4 + 33.0, rtol=1e-6)


def test_compute_policy_params_higher_service_level_means_more_safety_stock():
    params_90 = compute_policy_params(avg_daily_demand=50, sigma_d=8, unit_price=20, service_level=0.90)
    params_99 = compute_policy_params(avg_daily_demand=50, sigma_d=8, unit_price=20, service_level=0.99)
    assert params_99["safety_stock"] > params_90["safety_stock"]
    assert params_99["reorder_point"] > params_90["reorder_point"]


def test_compute_policy_params_zero_uncertainty_gives_zero_safety_stock():
    params = compute_policy_params(avg_daily_demand=50, sigma_d=0, unit_price=20, service_level=0.95)
    assert params["safety_stock"] == 0
    assert np.isclose(params["reorder_point"], 50 * params_lead_time_default())


def params_lead_time_default():
    from src.inventory.policies import LEAD_TIME_DAYS
    return LEAD_TIME_DAYS
