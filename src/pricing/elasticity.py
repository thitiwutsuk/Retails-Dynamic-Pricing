"""Stage 4c — price elasticity of demand via log-log OLS regression.

log(Units Sold + 1) ~ log(Price) + discount_pct + log(Competitor Pricing)
                        + Holiday/Promotion + Region + Weather Condition + Seasonality

Fit **per Category** (pooling across products/stores within a category gives
more rows per regression than fitting per product). The coefficient on
log(Price) is the price elasticity of demand: a value of -1.2 means "a 1%
price increase is associated with a 1.2% drop in Units Sold, holding
discount/competitor price/season/etc. fixed."

Caveat carried forward honestly rather than hidden: this is a plain OLS
correlation, not a causal estimate — if retailers already raise prices when
demand is high (reverse causality) the elasticity coefficient is biased.
Flagged explicitly in the notebook rather than presented as certain.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

CONTROL_CATEGORICALS = ["Region", "Weather Condition", "Seasonality"]
NUMERIC_FEATURES = ["log_price", "discount_pct", "log_competitor_price", "Holiday/Promotion"]


def build_design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the (X, y) pair for the log-log elasticity regression from a
    slice of the cleaned panel (raw categoricals, not one-hot features_full)."""
    d = df.copy()
    d["log_units_sold"] = np.log1p(d["Units Sold"])
    d["log_price"] = np.log(d["Price"])
    d["log_competitor_price"] = np.log(d["Competitor Pricing"])
    d["discount_pct"] = d["Discount"] / 100.0

    controls = pd.get_dummies(d[CONTROL_CATEGORICALS], drop_first=True)
    X = pd.concat([d[NUMERIC_FEATURES].astype(float), controls.astype(float)], axis=1)
    X = sm.add_constant(X)
    y = d["log_units_sold"].astype(float)
    return X, y


def fit_elasticity_by_category(df: pd.DataFrame, min_rows: int = 50) -> dict:
    """Fit one log-log OLS model per Category. Returns {category: fitted_model}."""
    models = {}
    for category, g in df.groupby("Category", observed=True):
        if len(g) < min_rows:
            continue
        X, y = build_design_matrix(g)
        models[category] = sm.OLS(y, X).fit()
    return models


def elasticity_summary(models: dict) -> pd.DataFrame:
    """One row per category: elasticity coefficient, its p-value, and R^2."""
    rows = []
    for category, model in models.items():
        rows.append(
            {
                "Category": category,
                "elasticity": model.params.get("log_price", np.nan),
                "p_value": model.pvalues.get("log_price", np.nan),
                "discount_coef": model.params.get("discount_pct", np.nan),
                "r_squared": model.rsquared,
                "n_obs": int(model.nobs),
            }
        )
    return pd.DataFrame(rows).sort_values("elasticity")


def compute_vif(X: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Variance Inflation Factor for the named numeric columns of a fitted
    design matrix, to check for multicollinearity among Price / Discount /
    Competitor Pricing."""
    rows = []
    for col in cols:
        idx = X.columns.get_loc(col)
        vif = variance_inflation_factor(X.to_numpy(dtype=float), idx)
        rows.append({"feature": col, "VIF": vif})
    return pd.DataFrame(rows)
