# extended_models/models_garch_har.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict
from arch import arch_model
import statsmodels.api as sm

def fit_garch_11(returns: pd.Series, scale: float = 100.0):
    """Fit GARCH(1,1) to returns. Returns arch result object."""
    r = returns.dropna().astype(float) * scale
    am = arch_model(r, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    return res

def forecast_garch_rolling(returns: pd.Series, train_size: int, scale: float = 100.0, refit_every: int = 0) -> pd.Series:
    """Rolling one-step-ahead volatility forecast with GARCH(1,1)."""
    r = returns.dropna().astype(float)
    idx = r.index; n = len(r)
    assert 0 < train_size < n
    preds, pred_idx = [], []
    last_refit = -1; res = None
    for t in range(train_size, n):
        if (res is None) or (refit_every == 0) or ((t - last_refit) >= refit_every):
            res = fit_garch_11(r.iloc[:t], scale=scale); last_refit = t
        fcast = res.forecast(horizon=1)
        var_next = fcast.variance.iloc[-1, 0]
        vol_next = float(np.sqrt(var_next)) / scale
        preds.append(vol_next); pred_idx.append(idx[t])
    return pd.Series(preds, index=pred_idx, name="garch11_vol_pred")

def _har_features(rv: pd.Series) -> pd.DataFrame:
    rv = rv.astype(float)
    RV1 = rv.shift(1)
    RV5 = rv.shift(1).rolling(5).mean()
    RV22 = rv.shift(1).rolling(22).mean()
    return pd.DataFrame({"RV1": RV1, "RV5": RV5, "RV22": RV22})

def fit_har_ols(rv_train: pd.Series) -> Dict[str, float]:
    """Fit HAR-RV via OLS; returns dict of coefficients."""
    X = _har_features(rv_train)
    y = rv_train
    df = pd.concat([X, y.rename("y")], axis=1).dropna()
    X_ = sm.add_constant(df[["RV1", "RV5", "RV22"]]); y_ = df["y"]
    model = sm.OLS(y_, X_).fit()
    p = model.params.to_dict()
    return {"const": p.get("const", 0.0), "RV1": p["RV1"], "RV5": p["RV5"], "RV22": p["RV22"]}

def forecast_har(rv: pd.Series, params: Dict[str, float], start_idx: int) -> pd.Series:
    """One-step-ahead HAR predictions from rv[start_idx:]."""
    rv = rv.astype(float); idx = rv.index; n = len(rv); preds = []
    for t in range(start_idx, n):
        rv1 = rv.iloc[t-1] if t-1 >= 0 else np.nan
        rv5 = rv.iloc[max(0, t-5):t].mean()
        rv22 = rv.iloc[max(0, t-22):t].mean()
        x = np.array([1.0, rv1, rv5, rv22])
        b = np.array([params["const"], params["RV1"], params["RV5"], params["RV22"]])
        preds.append(float(np.dot(x, b)))
    return pd.Series(preds, index=idx[start_idx:], name="har_vol_pred")
