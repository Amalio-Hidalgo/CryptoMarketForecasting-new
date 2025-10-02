import requests
import pandas as pd
import numpy as np, datetime as dt
from dune_client.client import DuneClient
from dune_client.query import QueryBase
import os
import dotenv
import time
# expects these globals to be defined by the notebook:
# TIMEZONE, DAYS_BACK, TARGET_COIN, CG_TOP_N, CG_HEADERS,
# DUNE_CSV_PATH, FRED_API_KEY (env)

# --- CoinGecko ---
def cg_universe(n, cg_headers):
    url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd"
    js = requests.get(url, headers=cg_headers).json()
    df = pd.DataFrame(js)
    uni = df.head(n)['id'].values
    return uni

def cgpriceactiondaily(coins, days, timezone, cg_headers):
    end   = int(dt.datetime.now(dt.timezone.utc).timestamp()) * 1000
    start = int((dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).timestamp()) * 1000
    count= 0
    for c in coins:
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{c}/market_chart/range?vs_currency=usd&from={start}&to={end}"
            js = requests.get(url, headers=cg_headers).json()
            p = pd.DataFrame(js["prices"],        columns=["t", f"prices_{c}"])
            m = pd.DataFrame(js["market_caps"],   columns=["t", f"marketcaps_{c}"])
            v = pd.DataFrame(js["total_volumes"], columns=["t", f"total_volumes_{c}"])
            df = p.merge(m, on="t").merge(v, on="t")
            df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
            df = df.set_index("t")
            df.columns = [x.lower() for x in df.columns]
            df.index = df.index.tz_convert(timezone).tz_localize(None)
            df = df.resample("1D").last().dropna(how="any")
            df.index.name = "date"
            if count ==0: out = df
            else: out = out.join(df, how='inner')
            count= count+1
        except Exception as e:
            print(f"Error for {c}: {e}")
            continue
        time.sleep(2)  # Add delay to avoid rate limits
    return out
# --- Deribit DVOL ---
def deribit_dvol_daily_multi(currencies, days, timezone, resolution="1D"):
    out = None
    end   = int(dt.datetime.now(dt.timezone.utc).timestamp()) * 1000
    start = int((dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).timestamp()) * 1000
    count=0
    for cur in currencies:
        js = requests.post(
            "https://www.deribit.com/api/v2/",
            json={"method": "public/get_volatility_index_data",
                    "params": {"currency": cur, "resolution": resolution,
                                "end_timestamp": end, "start_timestamp": start}}
        ).json()
        data = js.get("result", {}).get("data", [])
        if not data:
            continue
        d = pd.DataFrame(data, columns=["t","open","high","low","dvol"])
        d["t"] = pd.to_datetime(d["t"], unit="ms", utc=True)
        df = d.set_index("t")[["dvol"]].rename(columns={"dvol": f"dvol_{cur.lower()}"})
        df.index = df.index.tz_convert('Europe/Madrid').tz_localize(None)
        df = df.resample("1D").last().dropna(how="any")
        df.index.name = "date"
        if count ==0: out = df
        else: out = out.join(df, how='inner')
        count= count+1
    return out

# --- Dune (CSV) ---
def dune_metrics_daily(path, timezone):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=None)
    dt_col = None
    for c in df.columns:
        try:
            pd.to_datetime(df[c], utc=True, errors="raise")
            dt_col = c
            break
        except Exception:
            continue
    if dt_col is None and "date" in df.columns:
        dt_col = "date"
    if dt_col is None:
        return pd.DataFrame()
    df = df.rename(columns={dt_col: "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.set_index("date")
    df.index = df.index.tz_convert(timezone).tz_localize(None)
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"
    df = df.resample("1D").last().dropna(how="any")
    return df

# --- Dune (API) ---
def fetch_dune_queries_df(query_ids, timezone, dune_api_key=None):
    dune = DuneClient(api_key=dune_api_key or os.environ.get("DUNE_API_KEY"),
                      request_timeout=300, base_url="https://api.dune.com")
    out = None
    for qid in query_ids:
        try:
            q = QueryBase(query_id=qid)
            df = dune.run_query_dataframe(query=q, ping_frequency=2, batch_size=365)
            ok = False
            for col in list(df.columns):
                try:
                    pd.to_datetime(df[col], utc=True, errors="raise")
                    df = df.rename(columns={col: "date"}).set_index("date")
                    ok = True
                    break
                except Exception:
                    continue
            if not ok and not isinstance(df.index, pd.DatetimeIndex):
                continue
            if isinstance(df.index, pd.DatetimeIndex):
                df.index = df.index.tz_convert(timezone).tz_localize(None)
            df.columns = [c.lower() for c in df.columns]
            df.index.name = "date"
            df = df.resample("1D").last().dropna(how="any")
            out = df if out is None else out.join(df, how="inner")
        except Exception:
            continue
    return out if out is not None else pd.DataFrame()

# --- FRED ---
def fetch_fred_series_df(series_ids, start, timezone, fred_api_key=None):
    key = fred_api_key or os.getenv("FRED_API_KEY")
    if not key:
        return pd.DataFrame()
    base = "https://api.stlouisfed.org/fred/series/observations"
    out = None
    for sid in series_ids:
        try:
            js = requests.get(base, params={
                "series_id": sid, "api_key": key, "file_type": "json",
                "observation_start": start
            }).json()
            obs = js.get("observations", [])
            if not obs:
                continue
            df = pd.DataFrame(obs)[["date","value"]]
            df["date"]  = pd.to_datetime(df["date"], utc=True, errors="coerce")
            df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")
            df = df.set_index("date").rename(columns={"value": sid.lower()})
            df.index = df.index.tz_convert(timezone).tz_localize(None)
            df = df.resample("1D").last().dropna(how="any")
            df.index.name = "date"
            out = df if out is None else out.join(df, how="inner")
        except Exception:
            continue
    if out is not None and {"dgs10","dgs2"}.issubset(out.columns):
        out["term_spread_10y_2y"] = out["dgs10"] - out["dgs2"]
    return out if out is not None else pd.DataFrame()
