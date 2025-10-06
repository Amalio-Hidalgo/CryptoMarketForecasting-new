"""
Data Collection Module for Cryptocurrency Volatility Forecasting

This module contains all API collection functions with enhanced credit protection.
Handles data from CoinGecko, Binance, Deribit, FRED, and Dune Analytics.

🛡️ DUNE API CREDIT PROTECTION:
This module implements multiple layers of protection to prevent accidental 
credit consumption from Dune Analytics API:

SAFE METHODS (NO CREDITS):
- get_dune_data_safe(): Only loads CSV files
- get_dune_data_direct(): Uses cached results via get_latest_result()
- get_dune_data(strategy="cached_only"): Safe cache access
- get_dune_data(strategy="csv_only"): File-based only

CREDIT-CONSUMING METHODS:
- get_dune_data(strategy="execute_only"): Requires allow_dune_execution=True
- _execute_queries(): Direct execution (internal use)

RECOMMENDED APPROACH:
1. Always try get_dune_data_direct() first (free if cached)
2. Use CSV files for offline analysis  
3. Only execute fresh queries when explicitly needed

DEFAULT CONFIGURATION:
- allow_dune_execution=False (prevents accidental execution)
- dune_strategy="cached_only" (safe default)
- Key forecasting queries: [5893929, 5893461, 5893952, 5893947, 5894076, 5893557, 5893307, 5894092, 5894035, 5893555, 5893552, 5893566, 5893781, 5893821, 5892650, 5893009, 5892998, 5893911, 5892742, 5892720, 5891651, 5892696, 5892424, 5892227, 5891691]
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
        self.DUNE_API_KEY = os.getenv("DUNE_API_KEY")  # Using DUNE_API_KEY_2
        self.FRED_API_KEY = os.getenv("FRED_API_KEY")
        
        # Frequency configuration for batch sizing and resampling
        self.FREQUENCY = frequency if frequency else "1D"
        
        # Dune configuration
        self.DUNE_STRATEGY = dune_strategy
        self.ALLOW_DUNE_EXECUTION = allow_dune_execution
        
        # Configuration dictionaries - Updated daily queries (25 total)
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
            "query_25": 5891691
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

    # =============================================================================
    # COINGECKO API DATA COLLECTION
    # =============================================================================

    def coingecko_get_universe(self, 
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

    # =============================================================================
    # BINANCE API DATA COLLECTION
    # =============================================================================

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
            data = self.coingecko_get_universe(n=self.TOP_N, output_format="both")
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

    # =============================================================================
    # DERIBIT DVOL DATA COLLECTION
    # =============================================================================

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

    # =============================================================================
    # FRED MACROECONOMIC DATA COLLECTION
    # =============================================================================

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

    # =============================================================================
    # DUNE ANALYTICS DATA COLLECTION
    # =============================================================================

    def get_dune_data_safe(self, csv_path: str = "OutputData/dune_results.csv") -> pd.DataFrame:
        """
        🛡️  SAFE METHOD: Load Dune data without API calls.
        This method GUARANTEES no credits will be consumed.
        """
        print("� SAFE MODE: Loading CSV only (NO API CALLS)")
        return self._load_dune_csv(csv_path)
    
    def get_dune_data_direct(self, query_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        🎯 RECOMMENDED METHOD: Direct cache access using simple Dune client.
        Uses get_latest_result() - FREE if cached, minimal credits if not.
        
        This is the cleanest approach we learned from testing.
        """
        try:
            from dune_client.client import DuneClient
            from dune_client.query import QueryBase
            
            if not self.DUNE_API_KEY:
                print("❌ No Dune API key available")
                return pd.DataFrame()
            
            # Use key queries for volatility forecasting if none specified    
            if query_ids is None:
                key_queries = [5893929, 5893461, 5893952, 5893947, 5894076, 5893557, 5893307, 5894092, 5894035, 5893555, 5893552, 5893566, 5893781, 5893821, 5892650, 5893009, 5892998, 5893911, 5892742, 5892720, 5891651, 5892696, 5892424, 5892227, 5891691]  # All 25 queries
                query_ids = key_queries
            
            # Map query IDs to your custom titles
            query_titles = {
                5893929: "cum_deposited_eth",
                5893461: "economic_security", 
                5893952: "cum_validators",
                5893947: "staked_validators",
                5894076: "daily_dex_volume",
                5893557: "btc_etf_flows",
                5893307: "eth_etf_flows", 
                5894092: "total_defi_users",
                5894035: "median_gas",
                5893555: "staked_eth_category",
                5893552: "lsd_share",
                5893566: "lsd_tvl",
                5893781: "staking_rewards",
                5893821: "validator_performance",
                5892650: "ethereum_supply",
                5893009: "network_activity",
                5892998: "defi_tvl",
                5893911: "nft_volume",
                5892742: "bridge_activity",
                5892720: "mev_activity",
                5891651: "lending_metrics",
                5892696: "derivative_volume",
                5892424: "governance_activity",
                5892227: "yield_farming",
                5891691: "perpetual_volume"
            }
            
            dune = DuneClient(api_key=self.DUNE_API_KEY)
            results = {}
            cached_count = 0
            error_count = 0
            
            for qid in query_ids:
                try:
                    query = QueryBase(query_id=qid)
                    result = dune.get_latest_result(query)
                    
                    if result and result.result and result.result.rows:
                        df = pd.DataFrame([dict(row) for row in result.result.rows])
                        # Use your custom title instead of generic query number
                        title = query_titles.get(qid, f"query_{qid}")
                        results[title] = df
                        cached_count += 1
                    else:
                        error_count += 1
                        
                except Exception as e:
                    error_count += 1
            
            if cached_count > 0:
                print(f"🎊 Successfully retrieved {cached_count}/{len(query_ids)} cached datasets")
                print(f"💰 Credits used: 0 (cached results)")
                # Simple combination - preserve original column names
                combined_df = pd.DataFrame()
                for name, df in results.items():
                    if not df.empty:
                        print(f"   🔍 Processing {name}: columns = {list(df.columns)[:3]}...")
                        # Keep original column names - no prefixes
                        if combined_df.empty:
                            combined_df = df
                        else:
                            combined_df = pd.concat([combined_df, df], axis=1, sort=False)
                
                return combined_df
            else:
                return pd.DataFrame()
                
        except ImportError:
            print("❌ dune_client not available")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Direct access failed: {e}")
            return pd.DataFrame()
    
    def get_dune_data_recover(self, num_queries: int = 5) -> pd.DataFrame:
        """
        💰 RECOVERY METHOD: Execute a small number of queries to get actual data.
        Since credits were already spent, let's get some data back!
        """
        if not self.ALLOW_DUNE_EXECUTION:
            print("❌ Recovery mode requires allow_dune_execution=True")
            return pd.DataFrame()
            
        # Use only the first few queries to minimize additional credit usage
        query_subset = list(self.DUNE_QUERIES.values())[:num_queries]
        print(f"💡 RECOVERY MODE: Executing {len(query_subset)} queries to get data")
        print(f"💰 This will cost ~{len(query_subset) * 6} additional credits")
        
        return self._execute_queries(query_subset)

    def get_dune_data(self, query_ids: Optional[List[int]] = None, 
                      strategy: Optional[str] = None, 
                      csv_path: str = "OutputData/dune_results.csv") -> pd.DataFrame:
        """
        Get Dune Analytics data using safe, credit-aware strategies.
        
        🛡️  CREDIT PROTECTION STRATEGIES:
        - "csv_only": Load from CSV file only (FREE)
        - "cached_only": Use cached API results only (FREE if cached exists)
        - "direct_cache": Direct client.get_latest_result() (recommended)
        - "execute_only": Force fresh execution (COSTS ~6 credits per query!)
        
        Args:
            query_ids: List of query IDs. If None, uses key forecasting queries.
            strategy: Data collection strategy. Default: "cached_only"
            csv_path: Path to save/load CSV file
            
        Returns:
            DataFrame with Dune analytics data
            
        Note:
            Always try cached strategies first to avoid credit consumption.
            Use execute_only only when you explicitly need fresh data.
        """
        if query_ids is None:
            query_ids = list(self.DUNE_QUERIES.values())
            
        if not self.DUNE_API_KEY:
            print("❌ No Dune API key available")
            return pd.DataFrame()
        
        strategy = strategy or getattr(self, 'DUNE_STRATEGY', 'csv_cached_execute')
        
        # Strategy: CSV only
        if strategy == "csv_only":
            return self._load_dune_csv(csv_path)
            
        # Strategy: Cached only  
        elif strategy == "cached_only":
            print("🛡️  CACHED ONLY MODE: NO API CALLS WILL BE MADE")
            return self._get_cached_results(query_ids)
            
        # Strategy: Execute only
        elif strategy == "execute_only":
            if not getattr(self, 'ALLOW_DUNE_EXECUTION', False):
                print("🚫 Execution blocked by ALLOW_DUNE_EXECUTION=False")
                return pd.DataFrame()
            return self._execute_queries(query_ids)
            
        # Strategy: CSV -> Cached -> Execute (default)
        else:  # csv_cached_execute
            print("🛡️  SAFE MODE: Trying CSV and cache only (NO API CALLS)")
            
            # Try CSV first
            df = self._load_dune_csv(csv_path)
            if not df.empty:
                print("✅ Data loaded from CSV file")
                return df
                
            # Try cached files (local cache only, no API)
            df = self._get_cached_results(query_ids)
            if not df.empty:
                self._save_dune_csv(df, csv_path)
                return df
                
            # NEVER execute automatically - require explicit permission
            if getattr(self, 'ALLOW_DUNE_EXECUTION', False):
                print("⚠️  EXECUTION DISABLED TO PROTECT CREDITS")
                print("   💰 To execute queries (costs credits), use strategy='execute_only'")
            else:
                print("🚫 Dune execution blocked (ALLOW_DUNE_EXECUTION=False)")
                
        return pd.DataFrame()

    def _load_dune_csv(self, csv_path: str) -> pd.DataFrame:
        """Load Dune data from CSV file."""
        if not os.path.exists(csv_path):
            return pd.DataFrame()
            
        try:
            df = pd.read_csv(csv_path, index_col=0)
            df.index = pd.to_datetime(df.index)
            if self.TIMEZONE:
                df.index = df.index.tz_localize(self.TIMEZONE)
            df.index.name = "date"
            return df
        except Exception as e:
            print(f"⚠️  CSV loading failed: {str(e)[:50]}...")
            return pd.DataFrame()

    def _get_cached_results(self, query_ids: List[int]) -> pd.DataFrame:
        """
        SAFE METHOD: Get truly cached results WITHOUT making API calls.
        This method only loads pre-existing cached files and does NOT consume credits.
        """
        print("🔒 SAFE MODE: Loading only pre-cached data (NO API CALLS)")
        
        # Try to load from local cache directory
        cache_dir = "OutputData/dune_cache"
        if not os.path.exists(cache_dir):
            print(f"📁 No cache directory found: {cache_dir}")
            return pd.DataFrame()
        
        combined_df = None
        loaded_files = 0
        
        for qid in query_ids:
            cache_file = os.path.join(cache_dir, f"query_{qid}.csv")
            if os.path.exists(cache_file):
                try:
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    if not df.empty:
                        loaded_files += 1
                        if combined_df is None:
                            combined_df = df
                        else:
                            combined_df = combined_df.join(df, how='outer')
                        print(f"✅ Loaded cached query {qid}: {df.shape}")
                except Exception as e:
                    print(f"⚠️  Failed to load cached query {qid}: {e}")
                    continue
            else:
                print(f"📄 No cache file for query {qid}")
        
        if loaded_files == 0:
            print("ℹ️  No cached Dune data found. To get fresh data:")
            print("   1. Set allow_dune_execution=True (will consume credits)")
            print("   2. Or provide pre-cached CSV files in OutputData/dune_cache/")
        else:
            print(f"✅ Loaded {loaded_files}/{len(query_ids)} cached queries")
            print("💰 NO API CREDITS CONSUMED (local cache only)")
        
        return combined_df if combined_df is not None else pd.DataFrame()

    def _execute_queries(self, query_ids: List[int]) -> pd.DataFrame:
        """Execute fresh Dune queries and return data (costs credits)."""
        try:
            from dune_client.client import DuneClient
            from dune_client.query import QueryBase
            
            print(f"💰 EXECUTING {len(query_ids)} Dune queries (COSTS CREDITS)")
            dune = DuneClient(api_key=self.DUNE_API_KEY, base_url="https://api.dune.com")
            batch_size = self.get_batch_size_for_frequency()
            
            # Date range for filtering
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=self.LOOKBACK_DAYS)
            print(f"📅 Date range: {start_date.date()} to {end_date.date()}")
            
            combined_df = None
            successful = 0
            total_rows = 0
            
            for i, qid in enumerate(query_ids, 1):
                try:
                    print(f"🔄 Query {i}/{len(query_ids)}: {qid}", end="")
                    q = QueryBase(query_id=qid)
                    
                    # Execute query without parameters first (simpler)
                    df = dune.run_query_dataframe(query=q, ping_frequency=2)
                    
                    print(f" -> Raw: {df.shape[0]} rows, {df.shape[1]} cols")
                    
                    if not df.empty:
                        # Process the dataframe
                        processed_df = self._process_dune_dataframe(df, qid)
                        
                        if not processed_df.empty:
                            successful += 1
                            total_rows += len(processed_df)
                            print(f"   ✅ Processed: {processed_df.shape}")
                            
                            if combined_df is None:
                                combined_df = processed_df
                            else:
                                combined_df = combined_df.join(processed_df, how='outer')
                        else:
                            print(f"   ⚠️  Processing failed (empty result)")
                    else:
                        print(f"   ❌ No raw data returned")
                        
                except Exception as e:
                    print(f"   ❌ Failed: {str(e)[:50]}...")
                    continue
            
            print(f"\n📊 EXECUTION SUMMARY:")
            print(f"   • Successful queries: {successful}/{len(query_ids)}")
            print(f"   • Total rows collected: {total_rows}")
            print(f"   • Final combined shape: {combined_df.shape if combined_df is not None else (0, 0)}")
            print(f"   • 💰 Credits used: ~{len(query_ids) * 6} (estimated)")
                
            return combined_df if combined_df is not None else pd.DataFrame()
            
        except ImportError:
            print("❌ dune_client not available")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Execution failed: {e}")
            return pd.DataFrame()

    def _process_dune_dataframe(self, df: pd.DataFrame, query_id: int) -> pd.DataFrame:
        """Process and standardize Dune dataframe."""
        if df.empty:
            print(f"   📝 Query {query_id}: Empty dataframe")
            return df
        
        print(f"   📝 Query {query_id}: Processing {df.shape} with columns: {list(df.columns)[:3]}...")
            
        # Find date column (try common date column names)
        date_col = None
        common_date_cols = ['date', 'time', 'timestamp', 'block_date', 'block_time', 'day', 'created_at']
        
        # First, try common date column names
        for col in common_date_cols:
            if col in df.columns:
                try:
                    pd.to_datetime(df[col].iloc[0], errors='raise')
                    date_col = col
                    print(f"   📅 Found date column: '{col}'")
                    break
                except:
                    continue
        
        # If not found, search all columns
        if date_col is None:
            for col in df.columns:
                try:
                    sample_data = df[col].dropna()
                    if len(sample_data) > 0:
                        pd.to_datetime(sample_data.iloc[0], errors='raise')
                        date_col = col
                        print(f"   📅 Found date column: '{col}'")
                        break
                except (ValueError, TypeError, AttributeError):
                    continue
        
        if date_col is None:
            print(f"   ⚠️  No date column found - using row indices")
            # If no date column, create a simple numbered DataFrame with original column names
            result_df = df.copy()
            result_df.columns = [col.lower() for col in df.columns]
            result_df.index = pd.date_range(start='2024-01-01', periods=len(result_df), freq='D')
            return result_df
        
        # Process the dataframe
        try:
            df = df.rename(columns={date_col: "date"}).set_index("date")
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df.dropna(how='all')  # Remove rows where all values are NaN
            
            if len(df) == 0:
                print(f"   ⚠️  All rows dropped after date conversion")
                return pd.DataFrame()
            
            # Keep original column names from Dune queries
            df.columns = [col.lower() for col in df.columns]
            df.index.name = "date"
            
            print(f"   ✅ Successfully processed: {df.shape}")
            return df
            
        except Exception as e:
            print(f"   ❌ Processing failed: {e}")
            # Return raw data with original column names as fallback
            result_df = df.copy()
            result_df.columns = [col.lower() for col in df.columns]
            result_df.index = pd.date_range(start='2024-01-01', periods=len(result_df), freq='D')
            print(f"   🔄 Fallback processing: {result_df.shape}")
            return result_df

    def _save_dune_csv(self, df: pd.DataFrame, path: str) -> None:
        """Save dataframe to CSV with directory creation."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path)
            print(f"💾 Saved {len(df)} rows to {path}")
        except Exception as e:
            print(f"⚠️  CSV save failed: {e}")

    def get_batch_size_for_frequency(self) -> int:
        """Get appropriate batch size based on lookback period and frequency."""
        if self.FREQUENCY in ["1H", "1h", "hourly"]:
            return self.LOOKBACK_DAYS * 24  # Hours per day
        else:
            return self.LOOKBACK_DAYS  # Days

    # Legacy method for backward compatibility
    def dune_get_queries(self, query_ids: List[int], force_refresh: bool = False, 
                        allow_execution: bool = False) -> pd.DataFrame:
        """DEPRECATED: Use get_dune_data() instead."""
        print("⚠️  dune_get_queries() is deprecated. Use get_dune_data() instead.")
        
        if force_refresh and allow_execution:
            strategy = "execute_only"
        elif allow_execution:
            strategy = "csv_cached_execute"
        else:
            strategy = "cached_only"
            
        old_execution = self.ALLOW_DUNE_EXECUTION
        if allow_execution:
            self.ALLOW_DUNE_EXECUTION = True
            
        try:
            return self.get_dune_data(query_ids=query_ids, strategy=strategy)
        finally:
            self.ALLOW_DUNE_EXECUTION = old_execution

    # =============================================================================
    # UNIFIED DATA COLLECTION METHODS
    # =============================================================================

    def collect_all_data(self) -> Dict[str, pd.DataFrame]:
        """Collect data from all sources and return as dictionary."""
        print("🔄 Starting data collection...")
        
        # Get universe
        universe = self.coingecko_get_universe(self.TOP_N, output_format="both")
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
        universe = self.coingecko_get_universe(self.TOP_N, output_format="both")
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