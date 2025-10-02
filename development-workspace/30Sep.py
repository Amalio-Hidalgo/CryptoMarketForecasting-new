"""
Crypto Volatility Forecasting Toolkit
Complete API integration and modeling framework for cryptocurrency volatility prediction
"""

import random, os, pandas as pd, numpy as np
import matplotlib.pyplot as plt, datetime as dt
import dotenv, requests, time
import talib
import xgboost as xgb
from sklearn.metrics import r2_score
from tsfresh import extract_features, select_features
from tsfresh.utilities import roll_time_series
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.utilities.distribution import ClusterDaskDistributor
import optuna

# Environment setup
dotenv.load_dotenv(dotenv.find_dotenv(filename=".env"))

class CryptoVolatilityAPI:
    """Main class for crypto volatility data collection and forecasting"""
    
    def __init__(self, target_coin="ethereum", base_fiat="usd", top_n=10, 
                 lookback_days=365*5, timezone="Europe/Madrid"):
        self.TARGET_COIN = target_coin
        self.BASE_FIAT = base_fiat
        self.TOP_N = top_n
        self.LOOKBACK_DAYS = lookback_days
        self.START_DATE = (dt.datetime.now() - dt.timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        self.TODAY = dt.date.today().strftime('%Y-%m-%d')
        self.TIMEZONE = timezone
        self.FREQUENCY = "1D"
        
        # API Keys
        self.DUNE_API_KEY = os.getenv("DUNE_API_KEY")
        self.FRED_API_KEY = os.getenv("FRED_API_KEY")
        self.COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
        
        # Configuration dictionaries
        self.DUNE_QUERIES = {
            "economic_security": 1933076,
            "daily_dex_volume": 4388,
            "btc_etf_flows": 5795477,
            "eth_etf_flows": 5795645,
            "total_defi_users": 2972,
            "median_gas": 2981260,
        }
        
        self.FRED_KNOWN = {
            "VIXCLS": "vix_equity_vol",
            "MOVE": "move_bond_vol",
            "OVXCLS": "ovx_oil_vol",
            "GVZCLS": "gvz_gold_vol",
            "DTWEXBGS": "usd_trade_weighted_index",
            "DGS2": "us_2y_treasury_yield",
            "DGS10": "us_10y_treasury_yield",
        }
        
        self.DUNE_CSV_PATH = "OutputData/Dune_Metrics.csv"
        
        # Ensure output directory exists
        os.makedirs("OutputData", exist_ok=True)

    # --- CoinGecko Methods ---
    def coingecko_get_universe(self, n=None):
        """Top n cryptocurrency IDs from CoinGecko API sorted by market cap."""
        if n is None:
            n = self.TOP_N
        cg_headers = {
            "accept": "application/json",
            "x_cg_demo_api_key": self.COINGECKO_API_KEY
        }
        url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd"
        js = requests.get(url, headers=cg_headers).json()
        df = pd.DataFrame(js)
        time.sleep(3)
        try:
            return df.head(n)['id'].values
        except:
            print("Error Getting Coin Id's: ", df.get('error_message', 'Unknown error'))
            return np.array([])

    def coingecko_get_universe_v2(self, n=None, output_format="ids"):
        """Enhanced universe getter with multiple output formats"""
        if n is None:
            n = self.TOP_N
        cg_headers = {
            "accept": "application/json",
            "x_cg_demo_api_key": self.COINGECKO_API_KEY
        }
        url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc"
        js = requests.get(url, headers=cg_headers).json()
        df = pd.DataFrame(js)
        time.sleep(3)
        
        try:
            if output_format == "ids":
                result = df.head(n)['id'].values
                print(f"Retrieved {len(result)} coin IDs by market cap from CoinGecko")
                return result
            elif output_format == "symbols":
                result = df.head(n)['symbol'].str.upper().values
                print(f"Retrieved {len(result)} coin symbols by market cap from CoinGecko")
                return result
            elif output_format == "both":
                ids = df.head(n)['id'].values
                symbols = df.head(n)['symbol'].str.upper().values
                print(f"Retrieved {len(ids)} coins by market cap from CoinGecko")
                return {"ids": ids, "ticker": symbols}
            else:
                raise ValueError("output_format must be 'ids', 'symbols', or 'both'")
        except:
            print("Error Getting Coin Data")
            return np.array([])

    def coingecko_get_price_action(self, coins=None, start=None):
        """Get price action data from CoinGecko"""
        if coins is None:
            coins = self.coingecko_get_universe()
        if start is None:
            start = self.START_DATE
            
        coins = coins.tolist() if hasattr(coins, 'tolist') else coins
        end_timestamp = int(dt.datetime.now().timestamp()) * 1000
        start_timestamp = int(pd.to_datetime(start).timestamp()) * 1000
        
        outbig = None
        for c in coins:
            try:
                url = f"https://api.coingecko.com/api/v3/coins/{c}/market_chart/range?vs_currency=usd&from={start_timestamp}&to={end_timestamp}"
                cg_headers = {
                    "accept": "application/json",
                    "x_cg_demo_api_key": self.COINGECKO_API_KEY
                }
                js = requests.get(url, headers=cg_headers).json()
                outsmall = None
                for column in js:
                    timestamps = pd.to_datetime([x[0] for x in js[column]], unit='ms', utc=True).tz_convert(self.TIMEZONE)
                    values = [x[1] for x in js[column]]
                    if outsmall is None:
                        outsmall = pd.DataFrame(data=values, columns=[(column+'_'+c)], index=timestamps)
                    else:
                        outsmall[(column+'_'+c)] = values
                if outbig is None:
                    outbig = outsmall
                else:
                    outbig = outbig.join(outsmall, how='outer')
                time.sleep(3)
                continue
            except Exception as e:
                print(f'Error compiling data for {c}: {str(e)}')
                time.sleep(3)
                continue
        return outbig

    def coingecko_get_historical_paginated(self, coin_id, vs_currency="usd", max_days=365, step_days=90):
        """Get extended historical data using pagination"""
        full_prices = []
        full_market_caps = []
        full_volumes = []
        cg_headers = {"accept": "application/json", "x_cg_demo_api_key": self.COINGECKO_API_KEY}
        
        end_date = dt.datetime.now()
        current_end = int(end_date.timestamp())
        target_start_date = end_date - dt.timedelta(days=max_days)
        print(f"Fetching data for {coin_id} from {end_date.date()} back to {target_start_date.date()}")
        api_requests = 0
        
        while True:
            current_start = int((end_date - dt.timedelta(days=step_days)).timestamp())
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
            params = {
                "vs_currency": vs_currency,
                "from": current_start,
                "to": current_end
            }
            
            response = requests.get(url, headers=cg_headers, params=params)
            data = response.json()
            api_requests += 1
            
            if 'prices' not in data:
                print(f"No more data available or error after {api_requests} requests")
                if 'error' in data:
                    print(f"Error: {data['error']}")
                break
                
            prices = data.get('prices', [])
            market_caps = data.get('market_caps', [])
            volumes = data.get('total_volumes', [])
            
            if not prices:
                break
                
            full_prices = prices + full_prices
            full_market_caps = market_caps + full_market_caps
            full_volumes = volumes + full_volumes
            
            print(f"Request #{api_requests}: Got {len(prices)} price points")
            
            end_date = dt.datetime.fromtimestamp(current_start)
            current_end = current_start - 1
            
            if end_date <= target_start_date:
                print(f"Reached target date")
                break
                
            time.sleep(3)
        
        if not full_prices:
            print("No data collected")
            return pd.DataFrame()
        
        # Create DataFrames
        df_prices = pd.DataFrame(full_prices, columns=['timestamp', f'prices_{coin_id}'])
        df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp'], unit='ms')
        
        df_mcaps = pd.DataFrame(full_market_caps, columns=['timestamp', f'market_caps_{coin_id}'])
        df_mcaps['timestamp'] = pd.to_datetime(df_mcaps['timestamp'], unit='ms')
        
        df_volumes = pd.DataFrame(full_volumes, columns=['timestamp', f'total_volumes_{coin_id}'])
        df_volumes['timestamp'] = pd.to_datetime(df_volumes['timestamp'], unit='ms')
        
        df = df_prices.merge(df_mcaps, on='timestamp', how='outer')
        df = df.merge(df_volumes, on='timestamp', how='outer')
        df = df.set_index('timestamp')
        
        if self.TIMEZONE:
            df.index = df.index.tz_localize('UTC').tz_convert(self.TIMEZONE)
        df.index.name = 'date'
        
        print(f"Total data points: {len(df)}")
        print(f"Data ranges from {df.index.min().date()} to {df.index.max().date()}")
        return df

    # --- Binance Methods ---
    def binance_get_ohlc_paginated(self, ids=None, tickers=None, interval="1d", max_days=None):
        """Get extended OHLCV data from Binance using pagination"""
        if max_days is None:
            max_days = self.LOOKBACK_DAYS
            
        outbig = None
        if ids is None or tickers is None:
            universe = self.coingecko_get_universe_v2(n=self.TOP_N, output_format="both")
            ids, tickers = universe["ids"], universe["ticker"]
            
        for id, ticker in zip(ids, tickers):
            ticker = ticker.upper()
            print(f"Fetching {interval} candles for {id} going back {max_days} days...")
            
            full_data = []
            end_time = int(dt.datetime.now().timestamp() * 1000)
            start_date_target = dt.datetime.now() - dt.timedelta(days=max_days)
            api_requests = 0
            
            while True:
                url = "https://api.binance.com/api/v3/klines"
                params = {
                    "symbol": ticker + "USDT",
                    "interval": interval,
                    "endTime": end_time,
                    "limit": 1000
                }
                
                response = requests.get(url, params=params)
                data = response.json()
                api_requests += 1
                
                if not data or len(data) == 0 or (isinstance(data, dict) and 'code' in data):
                    print(f"No more data available for {id} after {api_requests} requests")
                    break
                    
                print(f"Request #{api_requests}: Got {len(data)} candles for {id}")
                full_data = data + full_data
                
                oldest_timestamp = int(data[0][0])
                oldest_date = dt.datetime.fromtimestamp(oldest_timestamp/1000)
                
                if oldest_date <= start_date_target:
                    print(f"Reached target date ({start_date_target.date()}) for {id}")
                    break
                    
                end_time = oldest_timestamp - 1
                time.sleep(1)
            
            if not full_data:
                print(f"No data collected for {id}")
                continue
                
            df = pd.DataFrame(full_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
                df[col + '_' + id.lower()] = df[col]
                
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            if self.TIMEZONE:
                df['date'] = df['date'].dt.tz_localize('UTC').dt.tz_convert(self.TIMEZONE)
            df = df.set_index('date')
            
            symbol_cols = [f"{col}_{id}" for col in ['open', 'high', 'low', 'close', 'volume']]
            df = df[symbol_cols]
            
            print(f"Total candles collected for {id}: {len(df)}")
            print(f"Data ranges from {df.index.min().date()} to {df.index.max().date()}")
            
            if outbig is None:
                outbig = df
            else:
                outbig = outbig.join(df, how='outer')
        
        if outbig is None:
            print("No data collected for any symbols.")
            return pd.DataFrame()
            
        outbig = outbig.sort_index()
        outbig.index.name = 'date'
        print(f"Combined data has {len(outbig)} rows")
        return outbig

    # --- Deribit Methods ---
    def deribit_get_dvol(self, currencies=['BTC', 'ETH'], days=None, resolution="1D"):
        """Get DVOL data from Deribit"""
        if days is None:
            days = self.LOOKBACK_DAYS
            
        out = None
        end = int(dt.datetime.now(dt.timezone.utc).timestamp()) * 1000
        start = int((dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).timestamp()) * 1000
        count = 0
        
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
                
            d = pd.DataFrame(data, columns=["t", "open", "high", "low", "dvol"])
            d["t"] = pd.to_datetime(d["t"], unit="ms", utc=True)
            df = d.set_index("t")[["dvol"]].rename(columns={"dvol": f"dvol_{cur.lower()}"})
            df.index = df.index.tz_convert(self.TIMEZONE)
            df = df.resample("1D").last().dropna(how="any")
            df.index.name = "date"
            
            if count == 0:
                out = df
            else:
                out = out.join(df, how='inner')
            count += 1
            
        return out

    # --- FRED Methods ---
    def fred_get_series(self, series_ids=None, start=None):
        """Get macroeconomic data from FRED"""
        if series_ids is None:
            series_ids = self.FRED_KNOWN
        if start is None:
            start = self.START_DATE
            
        key = self.FRED_API_KEY
        if not key:
            print("No FRED API Key Available")
            return pd.DataFrame()
            
        base = "https://api.stlouisfed.org/fred/series/observations"
        df = None
        
        for sid in series_ids:
            try:
                js = requests.get(base, params={
                    "series_id": sid, "api_key": key, "file_type": "json",
                    "observation_start": start
                }).json()
                
                obs = pd.DataFrame(js['observations'])
                index = pd.DatetimeIndex(obs['date'], freq='infer', tz=self.TIMEZONE)
                obs = obs.set_index(index)['value'].rename(self.FRED_KNOWN[sid])
                obs = pd.to_numeric(obs, errors='coerce')
                
                if df is not None:
                    df = pd.merge(left=df, right=obs, left_index=True, right_index=True)
                else:
                    df = obs
            except Exception as e:
                print(f"Error fetching {sid}: {e}")
                continue
            time.sleep(2)
            
        if df is not None:
            return df.asfreq('D', method='ffill')
        else:
            print('Error Compiling FRED Data')
            return pd.DataFrame()

    # --- Dune Methods ---
    def dune_from_csv(self, path=None):
        """Load Dune data from CSV"""
        if path is None:
            path = self.DUNE_CSV_PATH
            
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
        df.index = df.index.tz_convert(self.TIMEZONE)
        df.columns = [c.lower() for c in df.columns]
        df.index.name = "date"
        df = df.resample("1D").last().dropna(how="any")
        return df

    # --- Feature Engineering ---
    def compute_ta_indicators(self, df, price_prefix="prices_", rsi_period=14,
                            macd_fast=12, macd_slow=26, macd_signal=9,
                            sma_windows=(10,20,50), ema_windows=(10,20,50)):
        """Compute technical analysis indicators"""
        out = pd.DataFrame(index=df.index)
        price_cols = [c for c in df.columns if c.startswith(price_prefix)]
        coins = [c[len(price_prefix):] for c in price_cols]
        
        for coin in coins:
            p = df[f"{price_prefix}{coin}"]
            
            # RSI
            out[f"rsi{rsi_period}_{coin}"] = talib.RSI(p.values, timeperiod=rsi_period)
            
            # MACD
            macd, macd_sig, macd_hist = talib.MACD(p.values, fastperiod=macd_fast, 
                                                 slowperiod=macd_slow, signalperiod=macd_signal)
            out[f"macd_{coin}"] = macd
            out[f"macd_signal_{coin}"] = macd_sig
            out[f"macd_hist_{coin}"] = macd_hist
            
            # Moving averages
            for w in sma_windows:
                out[f"sma{w}_{coin}"] = talib.SMA(p.values, timeperiod=w)
            for w in ema_windows:
                out[f"ema{w}_{coin}"] = talib.EMA(p.values, timeperiod=w)
            
            # Bollinger Bands
            out[f"bb_upper_{coin}"], out[f"bb_middle_{coin}"], out[f"bb_lower_{coin}"] = talib.BBANDS(p.values)
            
            # Other indicators (if OHLC data available)
            if f"high_{coin}" in df.columns and f"low_{coin}" in df.columns:
                out[f"atr_{coin}"] = talib.ATR(df[f"high_{coin}"], df[f"low_{coin}"], p.values)
                out[f"adx_{coin}"] = talib.ADX(df[f"high_{coin}"], df[f"low_{coin}"], p.values)
                out[f"stoch_k_{coin}"], out[f"stoch_d_{coin}"] = talib.STOCH(df[f"high_{coin}"], df[f"low_{coin}"], p.values)
                out[f"cci_{coin}"] = talib.CCI(df[f"high_{coin}"], df[f"low_{coin}"], p.values)
                out[f"willr_{coin}"] = talib.WILLR(df[f"high_{coin}"], df[f"low_{coin}"], p.values)
                
            # Momentum indicators
            out[f"mom_{coin}"] = talib.MOM(p.values)
            out[f"roc_{coin}"] = talib.ROC(p.values)
            
            # Volume indicators (if volume data available)
            if f"volume_{coin}" in df.columns:
                out[f"obv_{coin}"] = talib.OBV(p.values, df[f"volume_{coin}"])
                if f"high_{coin}" in df.columns and f"low_{coin}" in df.columns:
                    out[f"mfi_{coin}"] = talib.MFI(df[f"high_{coin}"], df[f"low_{coin}"], 
                                                 p.values, df[f"volume_{coin}"])
        
        out.index = df.index
        return out

    # --- Unified Data Collection ---
    def collect_all_data(self):
        """Collect data from all sources and combine"""
        print("Collecting cryptocurrency universe...")
        universe = self.coingecko_get_universe_v2(self.TOP_N, output_format="both")
        ids, tickers = universe["ids"], universe["ticker"]
        
        print("Collecting CoinGecko price data...")
        price_action1 = self.coingecko_get_price_action(ids, start=self.START_DATE)
        
        print("Collecting Binance OHLC data...")
        price_action2 = self.binance_get_ohlc_paginated(ids=ids, tickers=tickers, 
                                                      interval="1d", max_days=self.LOOKBACK_DAYS)
        
        print("Collecting DVOL data...")
        dvol = self.deribit_get_dvol(['BTC','ETH'], days=self.LOOKBACK_DAYS)
        
        print("Loading on-chain analytics...")
        onchainanalytics = self.dune_from_csv()
        
        print("Collecting macroeconomic data...")
        macrodata = self.fred_get_series(series_ids=self.FRED_KNOWN, start=self.START_DATE)
        
        print("Combining all datasets...")
        unified = None
        for df in [price_action1, dvol, onchainanalytics, macrodata, price_action2]:
            if df is not None and not df.empty:
                df.index = df.index.tz_convert(self.TIMEZONE).date
                if unified is None:
                    unified = df
                else:
                    unified = unified.join(df, how='outer')
        
        if unified is not None:
            unified = unified.sort_index().dropna()
            print(f"Combined dataset shape: {unified.shape}")
            return unified
        else:
            print("No data collected")
            return pd.DataFrame()

    # --- TSFresh + XGBoost Pipeline ---
    def tsxg_multiprocessing(self, X, y, id='variable', sort='date', maxtimeshift=7,
                           fcparameters=None, fdrlvl=0.10, split_ratio=0.10, 
                           njobs=18, plot=True, xgb_params=None):
        """TSFresh feature extraction + XGBoost modeling with multiprocessing"""
        if fcparameters is None:
            fcparameters = EfficientFCParameters()
            
        # Stack long
        stacked = X.reset_index().melt(id_vars=sort, var_name='variable', value_name='value')
        
        # Roll windows
        rolled = roll_time_series(
            stacked,
            column_id=id,
            column_sort=sort,
            max_timeshift=maxtimeshift,
            n_jobs=njobs
        ).dropna()
        
        # Extract features
        features_raw = extract_features(
            rolled,
            column_id='id',
            column_sort=sort,
            column_kind=id,
            column_value='value',
            default_fc_parameters=fcparameters,
            n_jobs=njobs
        )
        
        # Merge by kind key
        count = 0
        for key in features_raw.index.levels[0]:
            if count == 0:
                feats = features_raw.loc[key].dropna(axis=1)
            else:
                feats = feats.merge(features_raw.loc[key].dropna(axis=1),
                                  left_index=True, right_index=True, how='inner')
            count += 1
        
        # Align + split
        Xf = feats.loc[y.index]
        test_n = max(1, int(len(Xf) * split_ratio))
        split = len(Xf) - test_n
        
        X_train = Xf.iloc[:split]
        y_train = y.iloc[:split]
        X_test = Xf.iloc[split:]
        y_test = y.iloc[split:]
        
        # Feature selection
        X_train_filtered = select_features(
            X_train, y_train,
            hypotheses_independent=False,
            ml_task='regression',
            n_jobs=njobs,
            fdr_level=fdrlvl
        )
        
        # Model params
        if xgb_params is None:
            xgb_params = dict(
                max_depth=5,
                learning_rate=0.1,
                min_child_weight=3.0,
                subsample=0.9,
                colsample_bytree=0.9,
                n_estimators=600,
                objective='reg:squarederror',
                tree_method='hist',
                eval_metric='mae',
                random_state=42,
                verbosity=0,
            )
        
        model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=10)
        
        eval_model = model.fit(
            X_train_filtered, y_train,
            eval_set=[(X_train_filtered, y_train)],
            verbose=False
        )
        
        preds = model.predict(X_test[X_train_filtered.columns])
        realized = y_test
        
        # Metrics
        mae_model = np.mean(np.abs(realized - preds))
        naive = realized.shift(1)
        mae_naive = np.mean(np.abs(realized[1:] - naive[1:]))
        mase = mae_model / mae_naive if (mae_naive is not None and not np.isnan(mae_naive)) else np.nan
        
        metrics = {
            "Best Score": getattr(eval_model, "best_score", np.nan),
            "Best Score / Median": (getattr(eval_model, "best_score", np.nan) /
                                  np.median(np.abs(y_train)) if len(y_train) else np.nan),
            "MASE": mase,
            "R^2": r2_score(realized, preds) if len(realized) else np.nan
        }
        
        if plot:
            plt.figure(figsize=(14, 4))
            plt.plot(realized.index, realized, label='Realized')
            plt.plot(realized.index, preds, label='Predicted')
            plt.title('Realized vs Predicted')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.show()
        
        return {
            'final_features': X_train_filtered,
            'xgb_model': model,
            'evaluation_metrics': metrics,
            'test_pred': pd.Series(preds, index=realized.index, name='y_hat')
        }

    def make_optuna_objective(self, X, y, scheduler_address=None, split_ratio=0.10, njobs=18, plot=False):
        """Create Optuna objective function for hyperparameter tuning"""
        def objective(trial):
            maxtimeshift = trial.suggest_int("maxtimeshift", 1, 62)
            fdrlvl = trial.suggest_float("fdrlvl", 0.02, 0.50)
            
            xgb_params = dict(
                max_depth=trial.suggest_int("max_depth", 3, 20),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.9, log=True),
                min_child_weight=trial.suggest_float("min_child_weight", 1.0, 10.0),
                subsample=trial.suggest_float("subsample", 0.1, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 1.0),
                n_estimators=trial.suggest_int("n_estimators", 10, 3000),
                objective="reg:squarederror",
                tree_method="hist",
                eval_metric="mae",
                verbosity=0,
            )
            
            res = self.tsxg_multiprocessing(
                X, y,
                maxtimeshift=maxtimeshift,
                njobs=njobs,
                fdrlvl=fdrlvl,
                split_ratio=split_ratio,
                plot=plot,
                xgb_params=xgb_params
            )
            
            mase = res["evaluation_metrics"]["MASE"]
            return mase if mase == mase else 1e9  # handle NaN -> large penalty
        
        return objective

    # --- Complete Pipeline ---
    def run_volatility_forecast_pipeline(self, optimize_hyperparams=False, n_trials=100):
        """Complete end-to-end volatility forecasting pipeline"""
        print("=== Crypto Volatility Forecasting Pipeline ===")
        
        # 1. Data Collection
        print("\n1. Collecting all data sources...")
        unified = self.collect_all_data()
        
        if unified.empty:
            print("No data collected. Exiting.")
            return None
        
        # 2. Feature Engineering
        print("\n2. Computing technical indicators...")
        ta_features = self.compute_ta_indicators(unified, price_prefix="prices_")
        unified = unified.join(ta_features, how='left').dropna()
        
        # 3. Target Variable Construction
        print("\n3. Creating target variable (realized volatility)...")
        unified[f'log_returns_{self.TARGET_COIN}'] = (np.log(unified[f'prices_{self.TARGET_COIN}']) - 
                                                    np.log(unified[f'prices_{self.TARGET_COIN}'].shift(1)))
        unified[f'realized_vol_{self.TARGET_COIN}'] = abs(unified[f'log_returns_{self.TARGET_COIN}'])
        
        X = unified.diff().dropna()
        y = X[f'realized_vol_{self.TARGET_COIN}'].shift(-1).dropna()
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        # 4. Model Training
        if optimize_hyperparams:
            print(f"\n4. Hyperparameter optimization with {n_trials} trials...")
            study = optuna.create_study(direction='minimize')
            objective = self.make_optuna_objective(X, y, plot=False)
            study.optimize(objective, n_trials=n_trials)
            
            print("Best parameters:", study.best_params)
            print("Best MASE:", study.best_value)
            
            # Train final model with best params
            best_params = study.best_params
            xgb_params = {k: v for k, v in best_params.items() 
                         if k not in ['maxtimeshift', 'fdrlvl']}
            xgb_params.update({
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "eval_metric": "mae",
                "verbosity": 0,
            })
            
            results = self.tsxg_multiprocessing(
                X, y,
                maxtimeshift=best_params['maxtimeshift'],
                fdrlvl=best_params['fdrlvl'],
                xgb_params=xgb_params,
                plot=True
            )
        else:
            print("\n4. Training model with default parameters...")
            results = self.tsxg_multiprocessing(X, y, plot=True)
        
        print("\n=== Final Results ===")
        for metric, value in results['evaluation_metrics'].items():
            print(f"{metric}: {value:.4f}")
        
        return {
            'unified_data': unified,
            'features': X,
            'target': y,
            'model_results': results
        }


# Example usage function
def main():
    """Example usage of the CryptoVolatilityAPI"""
    
    # Initialize the API
    api = CryptoVolatilityAPI(
        target_coin="ethereum",
        top_n=10,
        lookback_days=365*2,  # 2 years
        timezone="Europe/Madrid"
    )
    
    # Run the complete pipeline
    results = api.run_volatility_forecast_pipeline(
        optimize_hyperparams=True,
        n_trials=50
    )
    
    if results:
        print("\nPipeline completed successfully!")
        print(f"Final model MASE: {results['model_results']['evaluation_metrics']['MASE']:.4f}")
        print(f"Final model R²: {results['model_results']['evaluation_metrics']['R^2']:.4f}")
    
    return results


if __name__ == "__main__":
    results = main()