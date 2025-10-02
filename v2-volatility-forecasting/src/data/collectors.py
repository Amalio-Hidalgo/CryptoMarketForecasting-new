"""
Data Collection Module for Cryptocurrency Volatility Forecasting

This module contains all API collection functions from the working LatestNotebook.ipynb
Handles data from CoinGecko, Binance, Deribit, FRED, and Dune Analytics.
"""

import os
import time
import requests
import pandas as pd
import datetime as dt
from typing import List, Dict, Optional, Union, Tuple
from dune_client.client import DuneClient
from dune_client.query import QueryBase


class CryptoDataCollector:
    """Main class for collecting cryptocurrency and macro data from multiple sources."""
    
    def __init__(self, 
                 timezone: str = "Europe/Madrid",
                 top_n: int = 10,
                 lookback_days: int = 365,
                 frequency: str = "1D",
                 dune_strategy: str = "csv_cached_execute",
                 allow_dune_execution: bool = False):
        """
        Initialize the data collector.
        
        Args:
            timezone: Timezone for data alignment
            top_n: Number of top cryptocurrencies to collect
            lookback_days: Days of historical data to collect
            frequency: Data frequency ("1D", "1H", etc.)
            dune_strategy: Dune data collection strategy
            allow_dune_execution: Whether to allow API credit usage
        """
        self.TIMEZONE = timezone
        self.TOP_N = top_n
        self.LOOKBACK_DAYS = lookback_days
        self.START_DATE = (dt.datetime.now() - dt.timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        self.TODAY = dt.date.today().strftime('%Y-%m-%d')
        
        # API Keys from environment
        self.COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
        self.DUNE_API_KEY = os.getenv("DUNE_API_KEY_2")  # Using DUNE_API_KEY_2
        self.FRED_API_KEY = os.getenv("FRED_API_KEY")
        
        # Frequency configuration for batch sizing and resampling
        self.FREQUENCY = frequency if frequency else "1D"
        
        # Dune configuration
        self.DUNE_STRATEGY = dune_strategy
        self.ALLOW_DUNE_EXECUTION = allow_dune_execution
        
        # Configuration dictionaries - Updated daily queries (26 total)
        self.DUNE_QUERIES = {
            "query_01": 5893929,
            "query_02": 5893461,
            "query_03": 5893952,
            "query_04": 5893947,
            "query_05": 5894076,
            "query_06": 5893557,
            "query_07": 5893307,
            "query_08": 5894092,
            "query_09": 5894035,
            "query_10": 5893555,
            "query_11": 5893552,
            "query_12": 5893566,
            "query_13": 5893781,
            "query_14": 5893821,
            "query_15": 5892650,
            "query_16": 5893009,
            "query_17": 5892998,
            "query_18": 5893911,
            "query_19": 5892742,
            "query_20": 5892720,
            "query_21": 5891651,
            "query_22": 5892696,
            "query_23": 5892424,
            "query_24": 5892227,
            "query_25": 5891691,
            "query_26": 5795645
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

    def get_pandas_freq(self) -> str:
        """Convert internal frequency format to pandas resample format."""
        if self.FREQUENCY in ["1H", "1h", "hourly"]:
            return "H"
        else:
            return "D"
    
    def get_binance_interval(self) -> str:
        """Convert internal frequency format to Binance interval format."""
        if self.FREQUENCY in ["1H", "1h", "hourly"]:
            return "1h"
        else:
            return "1d"
    
    def get_deribit_resolution(self) -> str:
        """Convert internal frequency format to Deribit resolution format."""
        if self.FREQUENCY in ["1H", "1h", "hourly"]:
            return "1H"
        else:
            return "1D"
    
    def get_fred_frequency(self) -> str:
        """Convert internal frequency format to FRED frequency format."""
        if self.FREQUENCY in ["1H", "1h", "hourly"]:
            return "Daily (resampled to hourly)"
        else:
            return "Daily"
    
    def get_dune_resolution(self) -> str:
        """Convert internal frequency format to Dune resolution format."""
        if self.FREQUENCY in ["1H", "1h", "hourly"]:
            return "Hourly (when available)"
        else:
            return "Daily"
    
    # Keep private methods for backward compatibility
    def _get_pandas_freq(self) -> str:
        """DEPRECATED: Use get_pandas_freq() instead."""
        return self.get_pandas_freq()
    
    def _get_binance_interval(self) -> str:
        """DEPRECATED: Use get_binance_interval() instead."""
        return self.get_binance_interval()
    
    def _get_deribit_resolution(self) -> str:
        """DEPRECATED: Use get_deribit_resolution() instead."""
        return self.get_deribit_resolution()
    
    def _get_fred_frequency(self) -> str:
        """DEPRECATED: Use get_fred_frequency() instead."""
        return self.get_fred_frequency()
    
    def _get_dune_resolution(self) -> str:
        """DEPRECATED: Use get_dune_resolution() instead."""
        return self.get_dune_resolution()

    # --- CoinGecko Methods ---
    def coingecko_get_universe_v2(self, 
                                  n: Optional[int] = None, 
                                  output_format: str = "ids", 
                                  sleep_time: int = 6) -> Union[List[str], Dict[str, List[str]]]:
        """
        Top n cryptocurrency tickers and/or ids from CoinGecko API by market cap.
        
        Args:
            n: Number of top coins to retrieve
            output_format: "ids", "symbols", or "both"
            sleep_time: Sleep time between requests
            
        Returns:
            Array of identifiers or dictionary containing both formats
        """
        if n is None:
            n = self.TOP_N
        if self.COINGECKO_API_KEY is None:
            print("No CoinGecko API Key Available")
            if output_format == "both":
                return {"ids": [], "ticker": []}
            else:
                return []
            
        cg_headers = {
            "accept": "application/json",
            "x_cg_demo_api_key": self.COINGECKO_API_KEY
        }
        url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc"
        
        try:
            js = requests.get(url, headers=cg_headers).json()
            df = pd.DataFrame(js)
            time.sleep(sleep_time)
            
            if output_format == "ids":
                result = df.head(n)['id'].values
                return result
            elif output_format == "symbols":
                result = df.head(n)['symbol'].str.upper().values
                return result
            elif output_format == "both":
                ids = df.head(n)['id'].values
                symbols = df.head(n)['symbol'].str.upper().values
                return {"ids": ids, "ticker": symbols}
            else:
                raise ValueError("output_format must be 'ids', 'symbols', or 'both'")
        except Exception as e:
            print(f"Error getting coin IDs: {e}")
            return []

    def coingecko_get_price_action(self, 
                                   coins: List[str], 
                                   start: Optional[str] = None,
                                   freq: Optional[str] = None,
                                   sleep_time: int = 6) -> pd.DataFrame:
        """
        Get price action data from CoinGecko.
        Only works up to past 365 days, loses intraday data if > 90 days due to API limits.
        """
        if start is None:
            start = self.START_DATE
        if freq is None:
            freq = self.get_pandas_freq()
            
        end_timestamp = int(dt.datetime.now().timestamp()) * 1000
        start_timestamp = int(pd.to_datetime(start).timestamp()) * 1000
        
        cg_headers = {
            "accept": "application/json",
            "x_cg_demo_api_key": self.COINGECKO_API_KEY
        }
        
        outbig = None
        successful_coins = []
        failed_coins = []
        
        for c in coins:
            try:
                url = f"https://api.coingecko.com/api/v3/coins/{c}/market_chart/range?vs_currency=usd&from={start_timestamp}&to={end_timestamp}"
                js = requests.get(url, headers=cg_headers).json()
                
                outsmall = None
                for column in js:
                    timestamps = pd.to_datetime([x[0] for x in js[column]], unit='ms').tz_localize(self.TIMEZONE)
                    values = [x[1] for x in js[column]]
                    if outsmall is None:
                        outsmall = pd.DataFrame(data=values, columns=[(column+'_'+c)], index=timestamps)
                    else:
                        outsmall[(column+'_'+c)] = values
                
                outsmall[['prices_'+c, 'market_caps_'+c, 'total_volumes_'+c]] = outsmall[['prices_'+c, 'market_caps_'+c, 'total_volumes_'+c]].apply(pd.to_numeric, errors='coerce')
                outsmall.index.name = 'date'
                
                pricesandmc = outsmall[['prices_'+c, 'market_caps_'+c]].resample(freq).last().dropna()
                volumes = outsmall[['total_volumes_'+c]].resample(freq).sum().dropna()
                outsmall = pricesandmc.join(volumes, how='inner')
                
                successful_coins.append(f"{c} ({len(outsmall)} rows)")
                time.sleep(sleep_time)
                
                if outbig is None:
                    outbig = outsmall
                else:
                    outbig = outbig.join(outsmall, how='inner')
                    
            except Exception as e:
                failed_coins.append(f"{c} ({str(e)[:30]}...)")
                continue
        
        # Brief status output only if there are failures
        if failed_coins:
            print(f"⚠️  CoinGecko: {len(failed_coins)} failures")
                
        return outbig if outbig is not None else pd.DataFrame()

    # --- Binance Methods ---
    def binance_get_price_action(self, 
                                 ids: Optional[List[str]] = None,
                                 tickers: Optional[List[str]] = None,
                                 interval: Optional[str] = None,
                                 max_days: Optional[int] = None) -> pd.DataFrame:
        """
        Gets extended OHLCV data from Binance using pagination to overcome the 1000 candle limit.
        """
        if max_days is None:
            max_days = self.LOOKBACK_DAYS
        if interval is None:
            interval = self.get_binance_interval()
            
        outbig = None
        if ids is None or tickers is None:
            data = self.coingecko_get_universe_v2(n=self.TOP_N, output_format="both")
            ids, tickers = data["ids"], data["ticker"]
            
        successful_coins = []
        failed_coins = []
        
        for id, ticker in zip(ids, tickers):
            ticker = ticker.upper()
            
            # Pagination variables
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
                
                try:
                    response = requests.get(url, params=params)
                    data = response.json()
                    api_requests += 1
                    
                    if not data or len(data) == 0 or (isinstance(data, dict) and 'code' in data):
                        break
                        
                    full_data = data + full_data
                    
                    oldest_timestamp = int(data[0][0])
                    oldest_date = dt.datetime.fromtimestamp(oldest_timestamp/1000)
                    
                    if oldest_date <= start_date_target:
                        break
                        
                    end_time = oldest_timestamp - 1
                    time.sleep(1)
                    
                except Exception as e:
                    failed_coins.append(f"{id} ({str(e)[:30]}...)")
                    break
            
            if not full_data:
                failed_coins.append(f"{id} (no data)")
                continue
                
            df = pd.DataFrame(full_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
                df[col + '_' + id.lower()] = df[col]
                
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce', utc=True)
            df = df.set_index('date').tz_convert(self.TIMEZONE)
            
            symbol_cols = [f"{col}_{id}" for col in ['open', 'high', 'low', 'close', 'volume']]
            df = df[symbol_cols]
            
            successful_coins.append(f"{id} ({len(full_data)} candles)")
            
            if outbig is None:
                outbig = df
            else:
                outbig = outbig.join(df, how='outer')
        
        # Brief status output only if there are failures
        if failed_coins:
            print(f"⚠️  Binance: {len(failed_coins)} failures")
        
        if outbig is not None:
            outbig = outbig.sort_index()
            outbig.index.name = 'date'
            
        return outbig if outbig is not None else pd.DataFrame()

    # --- Deribit DVOL Methods ---
    def deribit_get_dvol(self, 
                         currencies: List[str] = ['BTC', 'ETH'],
                         days: Optional[int] = None,
                         resolution: Optional[str] = None) -> pd.DataFrame:
        """Get DVOL data from Deribit."""
        if days is None:
            days = self.LOOKBACK_DAYS
        if resolution is None:
            resolution = self.get_deribit_resolution()
            
        out = None
        end = int(dt.datetime.now().timestamp()) * 1000
        start = int((dt.datetime.now() - dt.timedelta(days=days)).timestamp()) * 1000
        count = 0
        
        for cur in currencies:
            try:
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
                d["t"] = pd.to_datetime(d["t"], unit="ms")
                df = d.set_index("t")[["dvol"]].rename(columns={"dvol": f"dvol_{cur.lower()}"})
                df.index = df.index.tz_localize(self.TIMEZONE)
                # Use frequency-aware resampling
                df = df.resample(self.get_pandas_freq()).last().dropna(how="any")
                df.index.name = "date"
                
                if count == 0:
                    out = df
                else:
                    out = out.join(df, how='inner')
                count += 1
                
            except Exception as e:
                print(f"Error fetching DVOL for {cur}: {e}")
                continue
                
        return out if out is not None else pd.DataFrame()

    # --- FRED Methods ---
    def fred_get_series(self, 
                        series_ids: Optional[Dict[str, str]] = None,
                        start: Optional[str] = None) -> pd.DataFrame:
        """Get macroeconomic data from FRED."""
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
                    "series_id": sid, 
                    "api_key": key, 
                    "file_type": "json",
                    "observation_start": start
                }).json()
                
                obs = pd.DataFrame(js['observations'])
                index = pd.DatetimeIndex(obs['date'], freq='infer', tz=self.TIMEZONE)
                obs = obs.set_index(index)['value'].rename(series_ids[sid])
                obs = pd.to_numeric(obs, errors='coerce')
                
                if df is not None:
                    df = pd.merge(left=df, right=obs, left_index=True, right_index=True)
                else:
                    df = obs
                    
                time.sleep(2)
                
            except Exception as e:
                print(f"Error fetching {series_ids[sid]}: {e}")
                continue
                
        # Use frequency-aware resampling
        if df is not None:
            return df.asfreq(self.get_pandas_freq(), method='ffill')
        return pd.DataFrame()

    # --- Dune Methods ---
    def dune_from_csv(self, path: str = "OutputData/Dune_Metrics.csv") -> pd.DataFrame:
        """Load Dune data from CSV."""
        if not os.path.exists(path):
            print(f"Dune CSV file not found at {path}")
            return pd.DataFrame()
            
        try:
            df = pd.read_csv(path, index_col=None)
            dt_col = None
            
            # Find datetime column
            for c in df.columns:
                try:
                    pd.to_datetime(df[c], errors="raise")
                    dt_col = c
                    break
                except Exception:
                    continue
                    
            if dt_col is None and "date" in df.columns:
                dt_col = "date"
            if dt_col is None:
                return pd.DataFrame()
                
            df = df.rename(columns={dt_col: "date"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.set_index("date")
            df.index = df.index.tz_localize(self.TIMEZONE)
            df.columns = [c.lower() for c in df.columns]
            df.index.name = "date"
            df = df.resample("1D").last().dropna(how="any")
            
            return df
            
        except Exception as e:
            print(f"Error loading Dune CSV: {e}")
            return pd.DataFrame()

    def check_query_freshness(self, query_id: int) -> bool:
        """Check if query was executed in the last 24 hours."""
        try:
            from dune_client.client import DuneClient
            from dune_client.query import QueryBase
            
            dune = DuneClient(api_key=self.DUNE_API_KEY, base_url="https://api.dune.com")
            
            # Try to get latest result metadata
            query = QueryBase(query_id=query_id)
            result = dune.get_latest_result(query)
            if result and hasattr(result, 'execution_ended_at') and result.execution_ended_at:
                # Check if execution was in last 24 hours
                time_diff = dt.datetime.now(dt.timezone.utc) - result.execution_ended_at
                return time_diff.total_seconds() < 86400  # 24 hours
            return False
        except Exception as e:
            print(f"Could not check freshness for query {query_id}: {e}")
            return False

    def get_batch_size_for_frequency(self) -> int:
        """Get appropriate batch size based on lookback period and frequency."""
        if self.FREQUENCY in ["1H", "1h", "hourly"]:
            # For hourly data: lookback_days * 24 hours per day
            return self.LOOKBACK_DAYS * 24
        else:
            # For daily data: just the lookback_days
            return self.LOOKBACK_DAYS

    def get_dune_data(self, query_ids: Optional[List[int]] = None, 
                      strategy: Optional[str] = None, 
                      csv_path: str = "OutputData/dune_results.csv") -> pd.DataFrame:
        """
        Unified method to get Dune data based on configured strategy.
        
        Strategies:
        - "csv_only": Only load from CSV file
        - "cached_only": Only use cached API results  
        - "execute_only": Force fresh execution (costs credits!)
        - "csv_cached_execute": Try CSV -> cached -> execute (default)
        
        Args:
            query_ids: List of query IDs. If None, uses all configured queries.
            strategy: Data collection strategy. If None, uses config setting.
            csv_path: Path to save/load CSV file
            
        Returns:
            DataFrame with Dune analytics data
        """
        if query_ids is None:
            query_ids = list(self.DUNE_QUERIES.values())
            
        if not self.DUNE_API_KEY:
            print("❌ No Dune API key available")
            return pd.DataFrame()
        
        # Use configured strategy if not provided
        if strategy is None:
            strategy = getattr(self, 'DUNE_STRATEGY', 'csv_cached_execute')
        
        # Strategy: CSV only
        if strategy == "csv_only":
            return self._load_from_csv(csv_path)
            
        # Strategy: Cached only  
        elif strategy == "cached_only":
            return self._get_cached_dune_results(query_ids)
            
        # Strategy: Execute only
        elif strategy == "execute_only":
            if not getattr(self, 'ALLOW_DUNE_EXECUTION', False):
                print("🚫 Execution blocked by ALLOW_DUNE_EXECUTION=False")
                return pd.DataFrame()
            return self._execute_dune_queries_with_date_filter(query_ids)
            
        # Strategy: CSV -> Cached -> Execute (default)
        else:  # csv_cached_execute
            # Step 1: Try CSV first
            df = self._load_from_csv(csv_path)
            if not df.empty:
                return df
                
            # Step 2: Try cached results
            df = self._get_cached_dune_results(query_ids)
            if not df.empty:
                self._save_dataframe_to_csv(df, csv_path)
                return df
                
            # Step 3: Execute if allowed
            if getattr(self, 'ALLOW_DUNE_EXECUTION', False):
                print("⚠️  Executing fresh Dune queries (consumes credits)")
                df = self._execute_dune_queries_with_date_filter(query_ids)
                if not df.empty:
                    self._save_dataframe_to_csv(df, csv_path)
                    return df
            else:
                print("🚫 Dune execution blocked (ALLOW_DUNE_EXECUTION=False)")
                
        return pd.DataFrame()
    
    def _load_from_csv(self, csv_path: str) -> pd.DataFrame:
        """Load Dune data from CSV file."""
        if not os.path.exists(csv_path):
            return pd.DataFrame()
            
        try:
            df = pd.read_csv(csv_path, index_col=0)
            df.index = pd.to_datetime(df.index)
            if self.TIMEZONE:
                df.index = df.index.tz_localize(self.TIMEZONE)
            df.index.name = "date"
            # CSV loaded silently
            return df
        except Exception as e:
            print(f"⚠️  CSV loading failed: {str(e)[:50]}...")
            return pd.DataFrame()
    
    def _get_cached_dune_results(self, query_ids: List[int]) -> pd.DataFrame:
        """Get cached results from Dune queries."""
        from dune_client.client import DuneClient
        from dune_client.query import QueryBase
        
        dune = DuneClient(api_key=self.DUNE_API_KEY, base_url="https://api.dune.com")
        
        out = None
        successful_queries = []
        failed_queries = []
        
        for qid in query_ids:
            try:
                # Try to get latest results without executing the query
                query = QueryBase(query_id=qid)
                result = dune.get_latest_result(query)
                
                if not result or not result.result or not result.result.rows:
                    failed_queries.append(f"Query {qid} (no cached data)")
                    continue
                    
                df = pd.DataFrame(result.result.rows)
                df = self._process_dune_dataframe(df, qid)
                
                if not df.empty:
                    successful_queries.append(f"Query {qid}")
                    if out is None:
                        out = df
                    else:
                        out = out.join(df, how='outer')
                else:
                    failed_queries.append(f"Query {qid} (processing failed)")
                    
            except Exception as e:
                failed_queries.append(f"Query {qid} ({str(e)[:50]}...)")
                continue
        
        # Clean summary output - only show failures
        if failed_queries:
            print(f"⚠️  Dune cached: {len(failed_queries)} failures")
                
        return out if out is not None else pd.DataFrame()
    
    def _execute_dune_queries(self, query_ids: List[int]) -> pd.DataFrame:
        """Execute fresh Dune queries (legacy method)."""
        return self._execute_dune_queries_with_date_filter(query_ids)
    
    def _execute_dune_queries_with_date_filter(self, query_ids: List[int]) -> pd.DataFrame:
        """
        Execute fresh Dune queries with date filtering to minimize credit usage.
        Only requests data for the configured lookback period.
        """
        from dune_client.client import DuneClient
        from dune_client.query import QueryBase
        import datetime as dt
        
        dune = DuneClient(api_key=self.DUNE_API_KEY, base_url="https://api.dune.com")
        batch_size = self.get_batch_size_for_frequency()
        
        # Calculate date range based on lookback days
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=self.LOOKBACK_DAYS)
        
        # Executing Dune queries with date filtering
        
        successful_queries = []
        failed_queries = []
        out = None
        
        for qid in query_ids:
            try:
                q = QueryBase(query_id=qid)
                
                # Execute with parameters if the query supports date filtering
                # Most Dune queries can be parameterized with start_date and end_date
                try:
                    # Try executing with date parameters
                    df = dune.run_query_dataframe(
                        query=q, 
                        ping_frequency=2, 
                        batch_size=batch_size,
                        parameters={
                            'start_date': start_date.strftime('%Y-%m-%d'),
                            'end_date': end_date.strftime('%Y-%m-%d')
                        }
                    )
                except:
                    # Fallback to standard execution if parameterization fails
                    df = dune.run_query_dataframe(query=q, ping_frequency=2, batch_size=batch_size)
                
                df = self._process_dune_dataframe(df, qid)
                
                if not df.empty:
                    # Additional date filtering after processing
                    df = df[df.index >= start_date.replace(tzinfo=df.index.tz)]
                    
                    successful_queries.append(f"Query {qid}")
                    if out is None:
                        out = df
                    else:
                        out = out.join(df, how='outer')
                else:
                    failed_queries.append(f"Query {qid} (no data)")
                    
            except Exception as e:
                failed_queries.append(f"Query {qid} ({str(e)[:50]}...)")
                continue
        
        # Summary output
        if successful_queries:
            print(f"✅ Successfully executed {len(successful_queries)} queries")
        if failed_queries:
            print(f"⚠️  Failed executions: {', '.join(failed_queries[:3])}" + 
                  (f" +{len(failed_queries)-3} more" if len(failed_queries) > 3 else ""))
                
        return out if out is not None else pd.DataFrame()
    
    def _process_dune_dataframe(self, df: pd.DataFrame, query_id: int) -> pd.DataFrame:
        """Process and standardize Dune dataframe."""
        if df.empty:
            return df
            
        # Find date column
        date_col = None
        for col in df.columns:
            try:
                sample_data = df[col].dropna()
                if len(sample_data) > 0:
                    pd.to_datetime(sample_data.iloc[0], errors='raise')
                    date_col = col
                    break
            except (ValueError, TypeError, AttributeError):
                continue
        
        if date_col is None:
            print(f"No date column found in query {query_id}")
            return pd.DataFrame()
        
        # Process the dataframe
        df = df.rename(columns={date_col: "date"}).set_index("date")
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.dropna(subset=[df.index.name])
        
        if self.TIMEZONE:
            df.index = df.index.tz_localize(self.TIMEZONE)
        
        df.columns = [c.lower() for c in df.columns]
        df.index.name = "date"
        
        # Resample to frequency
        df = df.resample(self.get_pandas_freq()).last().dropna(how="all")
        return df
    
    def _save_dataframe_to_csv(self, df: pd.DataFrame, path: str) -> None:
        """Save dataframe to CSV with directory creation."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path)
            print(f"💾 Saved {len(df)} rows to {path}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")

    def dune_get_queries(self, query_ids: List[int], force_refresh: bool = False, allow_execution: bool = False) -> pd.DataFrame:
        """
        DEPRECATED: Use get_dune_data() instead.
        
        This method is kept for backward compatibility but now uses the unified approach.
        """
        print("⚠️  dune_get_queries() is deprecated. Use get_dune_data() instead.")
        
        # Determine strategy based on parameters
        if force_refresh and allow_execution:
            strategy = "execute_only"
        elif allow_execution:
            strategy = "csv_cached_execute"
        else:
            strategy = "cached_only"
            
        # Temporarily override execution permission if explicitly allowed
        old_execution = self.ALLOW_DUNE_EXECUTION
        if allow_execution:
            self.ALLOW_DUNE_EXECUTION = True
            
        try:
            result = self.get_dune_data(query_ids=query_ids, strategy=strategy)
        finally:
            # Restore original setting
            self.ALLOW_DUNE_EXECUTION = old_execution
            
        return result

    def dune_get_cached_results_only(self, query_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Fetch only cached results from Dune queries without executing any queries.
        Perfect for collecting recent results without hitting API execution limits.
        
        Args:
            query_ids: List of query IDs to fetch cached results for. If None, uses all configured queries.
        """
        if query_ids is None:
            query_ids = list(self.DUNE_QUERIES.values())
            
        if not self.DUNE_API_KEY:
            print("No Dune API key available")
            return pd.DataFrame()
            
        try:
            from dune_client.client import DuneClient
            dune = DuneClient(api_key=self.DUNE_API_KEY, base_url="https://api.dune.com")
            
            all_dataframes = []
            successful_queries = 0
            
            print(f"🔍 Fetching cached results for {len(query_ids)} queries...")
            
            for i, qid in enumerate(query_ids, 1):
                try:
                    print(f"  [{i:2d}/{len(query_ids)}] Query {qid}...", end=" ")
                    
                    # Try to get the most recent execution results
                    try:
                        # Get latest execution
                        executions = dune.get_latest_result(qid)
                        if executions and hasattr(executions, 'get_rows'):
                            # Convert to DataFrame
                            df = pd.DataFrame(executions.get_rows())
                            
                            if len(df) > 0:
                                # Process the dataframe similar to regular dune_get_queries
                                date_col_found = False
                                for col in list(df.columns):
                                    try:
                                        pd.to_datetime(df[col], errors="raise")
                                        df = df.rename(columns={col: "date"}).set_index("date")
                                        date_col_found = True
                                        break
                                    except:
                                        continue
                                
                                if date_col_found:
                                    # Ensure datetime index
                                    if not isinstance(df.index, pd.DatetimeIndex):
                                        df.index = pd.to_datetime(df.index)
                                    
                                    # Localize timezone
                                    if df.index.tz is None:
                                        df.index = df.index.tz_localize(self.TIMEZONE)
                                    
                                    # Clean column names
                                    df.columns = [f"{c.lower()}_{qid}" for c in df.columns]
                                    df.index.name = "date"
                                    
                                    # Apply frequency resampling
                                    resample_freq = self.get_pandas_freq()
                                    df = df.resample(resample_freq).last().dropna(how="any")
                                    
                                    all_dataframes.append(df)
                                    successful_queries += 1
                                    print(f"✓ {df.shape}")
                                else:
                                    print("✗ No date column")
                            else:
                                print("✗ Empty results")
                        else:
                            print("✗ No cached results")
                    
                    except Exception as inner_e:
                        print(f"✗ Error: {str(inner_e)[:50]}...")
                        continue
                        
                except Exception as e:
                    print(f"✗ Failed: {str(e)[:50]}...")
                    continue
            
            # Combine all successful dataframes
            if all_dataframes:
                print(f"\n🔗 Combining {len(all_dataframes)} successful queries...")
                
                # Find common date range
                combined_df = all_dataframes[0]
                for df in all_dataframes[1:]:
                    combined_df = combined_df.join(df, how='outer')
                
                print(f"✅ Combined dataset: {combined_df.shape}")
                print(f"📅 Date range: {combined_df.index.min()} to {combined_df.index.max()}")
                
                return combined_df
            else:
                print("❌ No successful queries found")
                return pd.DataFrame()
                
        except ImportError:
            print("❌ dune_client not installed")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Error collecting cached results: {e}")
            return pd.DataFrame()

    def save_dune_cached_to_csv(self, output_path: str = "OutputData/dune_cached_results.csv") -> bool:
        """
        Collect cached Dune results and save to CSV file.
        
        Args:
            output_path: Path to save the CSV file
        """
        import os
        
        print("🚀 COLLECTING CACHED DUNE RESULTS")
        print("=" * 50)
        
        # Collect cached results
        df = self.dune_get_cached_results_only()
        
        if len(df) > 0:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Sort by date for better readability
            df = df.sort_index()
            
            # Save to CSV
            df.to_csv(output_path)
            
            print(f"💾 SAVED TO: {output_path}")
            print(f"📊 Dataset: {df.shape}")
            print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
            print(f"🔢 Columns: {len(df.columns)} indicators")
            print(f"📈 Frequency: {self.FREQUENCY} ({self.get_pandas_freq()})")
            
            # Show sample of column names
            print(f"\n📋 Sample columns:")
            for i, col in enumerate(df.columns[:10]):
                print(f"   {i+1:2d}. {col}")
            if len(df.columns) > 10:
                print(f"   ... and {len(df.columns) - 10} more")
                
            return True
        else:
            print("❌ No data to save")
            return False

    # --- Main Collection Method ---
    def collect_all_data(self) -> Dict[str, pd.DataFrame]:
        """Collect data from all sources and return as dictionary."""
        print("🔄 Starting data collection...")
        
        # Get universe
        universe = self.coingecko_get_universe_v2(self.TOP_N, output_format="both")
        if isinstance(universe, dict):
            ids, tickers = universe["ids"], universe["ticker"]
        else:
            print("❌ Failed to get universe data")
            return {}
        
        data = {}
        
        # Collect price data silently
        data['binance_price'] = self.binance_get_price_action(ids=ids, tickers=tickers, 
                                                              max_days=self.LOOKBACK_DAYS)
        
        data['coingecko_price'] = self.coingecko_get_price_action(ids, start=self.START_DATE)
        
        data['dvol'] = self.deribit_get_dvol(['BTC', 'ETH'], days=self.LOOKBACK_DAYS)
        
        data['onchain'] = self.get_dune_data()
        
        data['macro'] = self.fred_get_series(series_ids=self.FRED_KNOWN, start=self.START_DATE)
        
        print("✅ Data collection completed")
        return data

    def collect_all_data_with_cached_dune(self) -> Dict[str, pd.DataFrame]:
        """Collect data from all sources using only cached Dune results (no credits consumed)."""
        print("🔄 Starting data collection (cached Dune only)...")
        
        # Get universe
        universe = self.coingecko_get_universe_v2(self.TOP_N, output_format="both")
        if isinstance(universe, dict):
            ids, tickers = universe["ids"], universe["ticker"]
        else:
            print("❌ Failed to get universe data")
            return {}
        
        data = {}
        
        # Collect price data silently
        data['binance_price'] = self.binance_get_price_action(ids=ids, tickers=tickers, 
                                                              max_days=self.LOOKBACK_DAYS)
        
        data['coingecko_price'] = self.coingecko_get_price_action(ids, start=self.START_DATE)
        
        data['dvol'] = self.deribit_get_dvol(['BTC', 'ETH'], days=self.LOOKBACK_DAYS)
        
        # Use cached-only strategy for Dune data
        data['onchain'] = self.get_dune_data(strategy="cached_only")
        
        data['macro'] = self.fred_get_series(series_ids=self.FRED_KNOWN, start=self.START_DATE)
        
        print("✅ Data collection completed (cached)")
        return data

    def combine_data_sources(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine all data sources into unified DataFrame."""
        
        unified = None
        for name, df in data.items():
            if df.empty:
                continue
                
            try:
                # Standardize timezone
                if df.index.tz is None:
                    df.index = pd.DatetimeIndex(df.index).tz_localize(self.TIMEZONE).date
                else:
                    df.index = pd.DatetimeIndex(df.index).tz_convert(self.TIMEZONE).date
                    
                if unified is None:
                    unified = df
                else:
                    unified = unified.join(df, how='outer')
                    
            except Exception as e:
                print(f"❌ Error combining {name}: {e}")
                continue
        
        return unified if unified is not None else pd.DataFrame()


# Convenience functions for quick access
def collect_crypto_data(top_n: int = 10, 
                       lookback_days: int = 365,
                       timezone: str = "Europe/Madrid",
                       frequency: str = "1D") -> pd.DataFrame:
    """
    Quick function to collect and combine all crypto data.
    🔒 SAFE MODE: Uses cached data only - NO API credits consumed.
    """
    collector = CryptoDataCollector(timezone=timezone, top_n=top_n, 
                                  lookback_days=lookback_days, frequency=frequency,
                                  dune_strategy="cached_only", 
                                  allow_dune_execution=False)
    data = collector.collect_all_data()
    return collector.combine_data_sources(data)

def collect_crypto_data_with_cached_dune(top_n: int = 10, 
                                         lookback_days: int = 365,
                                         timezone: str = "Europe/Madrid",
                                         frequency: str = "1D") -> pd.DataFrame:
    """
    SAFE function to collect all crypto data including cached Dune results.
    🔒 NO QUERY EXECUTION - uses only cached results, no API credits consumed.
    """
    collector = CryptoDataCollector(timezone=timezone, top_n=top_n, 
                                  lookback_days=lookback_days, frequency=frequency)
    data = collector.collect_all_data_with_cached_dune()
    return collector.combine_data_sources(data)

def collect_crypto_data_with_fresh_dune(top_n: int = 10, 
                                        lookback_days: int = 365,
                                        timezone: str = "Europe/Madrid",
                                        frequency: str = "1D") -> pd.DataFrame:
    """
    ⚠️  CAUTION: Executes fresh Dune queries - CONSUMES API CREDITS!
    Only use when you need the most recent onchain analytics data.
    """
    print("🚨 WARNING: This function consumes Dune API credits!")
    collector = CryptoDataCollector(timezone=timezone, top_n=top_n, 
                                  lookback_days=lookback_days, frequency=frequency,
                                  dune_strategy="execute_only", 
                                  allow_dune_execution=True)
    data = collector.collect_all_data()
    return collector.combine_data_sources(data)


if __name__ == "__main__":
    # Test the collector
    collector = CryptoDataCollector(top_n=5, lookback_days=30, frequency="1D")
    print(f"Testing collector with frequency: {collector.FREQUENCY}")
    data = collector.collect_all_data()
    unified = collector.combine_data_sources(data)
    print(f"Final dataset shape: {unified.shape}")