# apiwrappers/indicators_talib.py
from __future__ import annotations
import pandas as pd
import talib

def compute_ta_indicators(
    df: pd.DataFrame,
    price_prefix: str = "prices_",
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_windows: tuple[int, ...] = (10, 20, 50),
    ema_windows: tuple[int, ...] = (10, 20, 50),
) -> pd.DataFrame:
    """Compute TA-Lib indicators for each symbol column in `df` with name starting by `price_prefix`.
    Returns a DataFrame with columns like: rsi{p}_{coin}, macd_{coin}, macd_signal_{coin}, macd_hist_{coin}, sma{w}_{coin}, ema{w}_{coin}
    """
    out = pd.DataFrame(index=df.index)
    price_cols = [c for c in df.columns if c.startswith(price_prefix)]
    if not price_cols:
        return out
    coins = [c[len(price_prefix):] for c in price_cols]
    for coin in coins:
        p = pd.to_numeric(df[f"{price_prefix}{coin}"], errors="coerce")
        out[f"rsi{rsi_period}_{coin}"] = talib.RSI(p.values, timeperiod=rsi_period)
        macd, macd_sig, macd_hist = talib.MACD(p.values, fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)
        out[f"macd_{coin}"] = macd
        out[f"macd_signal_{coin}"] = macd_sig
        out[f"macd_hist_{coin}"] = macd_hist
        for w in sma_windows:
            out[f"sma{w}_{coin}"] = talib.SMA(p.values, timeperiod=w)
        for w in ema_windows:
            out[f"ema{w}_{coin}"] = talib.EMA(p.values, timeperiod=w)
    out.index = df.index
    return out
