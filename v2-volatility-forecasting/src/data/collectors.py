"""
Data Collection Module for Cryptocurrency Volatility Forecasting

This module provides comprehensive data collection capabilities for cryptocurrency 
volatility forecasting models. It integrates multiple data sources including market data,
on-chain analytics, volatility indices, and macroeconomic indicators.

Data Sources:
    - CoinGecko: Historical price and market data for top cryptocurrencies
    - Binance: High-frequency OHLCV data with pagination support  
    - Deribit: DVOL volatility indices for BTC and ETH
    - FRED: Macroeconomic indicators and traditional asset volatility
    - Dune Analytics: On-chain metrics and DeFi analytics (21 curated queries)

Key Features:
    - Unified data collection interface with timezone alignment
    - Automatic frequency conversion across all data sources
    - Credit-conscious API usage with built-in rate limiting
    - Comprehensive error handling and data validation
    - CSV caching for offline analysis and development

Dune Analytics Integration:
    Includes 25 curated on-chain metrics covering:
    - Ethereum staking metrics (validators, deposits, rewards)
    - DeFi activity (TVL, users, transaction volume)
    - Market structure (ETF flows, derivatives, governance)
    - Network health (gas prices, MEV, bridge activity)

Usage:
    ```python
    from data.collectors import CryptoDataCollector
    
    collector = CryptoDataCollector(
        timezone="UTC",
        top_n=10,
        lookback_days=365,
        frequency="1D"
    )
    
    # Collect all data sources
    data = collector.collect_all_data()
    unified = collector.combine_data_sources(data)
    ```

Note:
    Requires environment variables for API keys:
    COINGECKO_API_KEY, DUNE_API_KEY, FRED_API_KEY
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
    """
    Comprehensive cryptocurrency and macroeconomic data collector.
    
    This class provides a unified interface for collecting data from multiple sources
    required for cryptocurrency volatility forecasting. It handles API authentication,
    rate limiting, data alignment, and provides both individual and batch collection methods.
    
    The collector automatically handles frequency conversion, timezone alignment, and
    data quality validation across all supported data sources.
    
    Attributes:
        timezone (str): Target timezone for data alignment
        top_n (int): Number of top cryptocurrencies to collect
        lookback_days (int): Historical data window in days
        frequency (str): Data frequency ("1D" for daily, "1H" for hourly)
        dune_queries (dict): Mapping of Dune query IDs to descriptive names
        fred_series (dict): Mapping of FRED series IDs to descriptive names
    """
    
    @staticmethod
    def _normalize_frequency(freq: str) -> str:
        """
        Normalize frequency string to standard format.
        
        Args:
            freq: Input frequency (e.g., "1H", "hourly", "1D", "daily")
            
        Returns:
            Standardized frequency: "1H" for hourly, "1D" for daily
        """
        freq_lower = freq.lower().strip()
        
        # Hourly patterns
        if freq_lower in ["1h", "h", "hourly", "hour", "1hour"]:
            return "1H"
        # Daily patterns  
        elif freq_lower in ["1d", "d", "daily", "day", "1day"]:
            return "1D"
        else:
            # Default to daily for unknown
            print(f"⚠️ Unknown frequency '{freq}', defaulting to daily (1D)")
            return "1D"
    
    def __init__(self, 
                 config = None,
                 timezone: Optional[str] = None,
                 top_n: Optional[int] = None,
                 lookback_days: Optional[int] = None,
                 frequency: Optional[str] = None,
                 use_cached_dune_only: Optional[bool] = None):
        """
        Initialize the cryptocurrency data collector with centralized configuration.
        
        Args:
            config: Config object from config.py (if None, creates default)
            timezone (str): Override timezone (if None, uses config default)
            top_n (int): Override number of top cryptos (if None, uses config default)
            lookback_days (int): Override historical window (if None, uses config default)
            frequency (str): Override data frequency (if None, uses config default)
            use_cached_dune_only (bool): Override Dune caching (if None, uses config default)
        
        Note:
            - Parameters override config values if provided
            - Uses centralized config.py for all constants and API settings
            - Follows proper Python naming conventions (lowercase instance variables)
        """
        # Import and setup centralized configuration
        from config import load_config, APIConfig
        
        # Use provided config or create default
        if config is None:
            config = load_config()
        
        # Apply parameter overrides or use config defaults
        self.timezone = timezone or config.data.timezone
        self.top_n = top_n or config.data.top_n
        self.lookback_days = lookback_days or config.data.lookback_days
        
        # Normalize and validate frequency
        raw_frequency = frequency or config.data.frequency
        self.frequency = self._normalize_frequency(raw_frequency)
        
        self.use_cached_dune_only = use_cached_dune_only if use_cached_dune_only is not None else True
        
        # Frequency warnings for data sources
        if self.supports_hourly():
            print(f"⚠️ Hourly frequency selected - Note:")
            print(f"   • CoinGecko: Limited to last 90 days for hourly data")
            print(f"   • FRED: Only daily data available (will be resampled)")
            print(f"   • Dune: Mostly daily data (hourly limited)")
        
        # Calculate derived values
        self.start_date = (dt.datetime.now() - dt.timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
        self.today = dt.date.today().strftime('%Y-%m-%d')
        
        # API Configuration from centralized config
        api_config = APIConfig()
        self.api_keys = {
            'coingecko': os.getenv("COINGECKO_API_KEY"),
            'dune': os.getenv("DUNE_API_KEY"),
            'fred': os.getenv("FRED_API_KEY")
        }
        
        # Import all constants from centralized config (no more duplication!)
        self.dune_queries = api_config.dune_queries
        self.fred_series = api_config.fred_series
        
        # Store full config for advanced usage
        self.config = config

    # =============================================================================
    # FREQUENCY CONVERSION METHODS
    # =============================================================================
    
    def get_pandas_freq(self) -> str:
        """
        Convert to pandas resample frequency.
        
        Returns:
            "H" for hourly, "D" for daily
        """
        return "H" if self.frequency == "1H" else "D"
    
    def get_binance_interval(self) -> str:
        """
        Convert to Binance API interval format.
        
        Binance supports: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
        
        Returns:
            "1h" for hourly, "1d" for daily
        """
        return "1h" if self.frequency == "1H" else "1d"
    
    def get_deribit_resolution(self) -> str:
        """
        Convert to Deribit API resolution format.
        
        Deribit DVOL supports: 1, 60, 1D
        (1 = 1 minute, 60 = 1 hour, 1D = 1 day)
        
        Returns:
            "60" for hourly, "1D" for daily
        """
        return "60" if self.frequency == "1H" else "1D"
    
    def get_coingecko_interval(self) -> str:
        """
        Convert to CoinGecko API interval format.
        
        CoinGecko supports: daily, hourly (but hourly only for last 90 days)
        
        Returns:
            "hourly" or "daily"
        """
        return "hourly" if self.frequency == "1H" else "daily"
    
    def supports_hourly(self) -> bool:
        """
        Check if current frequency is hourly.
        
        Returns:
            True if hourly, False if daily
        """
        return self.frequency == "1H"
    
    def get_frequency_description(self) -> str:
        """
        Get human-readable frequency description.
        
        Returns:
            Descriptive string for logging
        """
        return "Hourly (1H)" if self.frequency == "1H" else "Daily (1D)"

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
            n = self.top_n
        
        coingecko_key = self.api_keys.get('coingecko')
        if coingecko_key is None:
            print("No CoinGecko API Key Available")
            if output_format == "both":
                return {"ids": [], "ticker": []}
            else:
                return []
            
        cg_headers = {
            "accept": "application/json",
            "x_cg_demo_api_key": coingecko_key
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
            start = self.start_date
        if freq is None:
            freq = self.get_pandas_freq()
            
        end_timestamp = int(dt.datetime.now().timestamp()) * 1000
        start_timestamp = int(pd.to_datetime(start).timestamp()) * 1000
        
        coingecko_key = self.api_keys.get('coingecko')
        cg_headers = {
            "accept": "application/json",
            "x_cg_demo_api_key": coingecko_key
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
                    timestamps = pd.to_datetime([x[0] for x in js[column]], unit='ms').tz_localize(self.timezone)
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
            max_days = self.lookback_days
        if interval is None:
            interval = self.get_binance_interval()
            
        outbig = None
        if ids is None or tickers is None:
            data = self.coingecko_get_universe(n=self.top_n, output_format="both")
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
            df = df.set_index('date').tz_convert(self.timezone)
            
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
            days = self.lookback_days
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
                df.index = df.index.tz_localize(self.timezone)
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
        """
        Get macroeconomic data from FRED.
        
        Note: FRED only provides daily data. If hourly frequency is requested,
        the data will be forward-filled to match the frequency.
        
        Args:
            series_ids: Dict mapping FRED series IDs to column names
            start: Start date for data collection
            
        Returns:
            DataFrame with FRED economic indicators
        """
        if series_ids is None:
            series_ids = self.fred_series
        if start is None:
            start = self.start_date
            
        key = self.api_keys.get('fred')
        if not key:
            print("⚠️ No FRED API Key Available")
            return pd.DataFrame()
        
        # Warn if hourly requested (FRED only has daily)
        if self.supports_hourly():
            print("ℹ️  FRED provides daily data only - will forward-fill for hourly frequency")
            
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
                index = pd.DatetimeIndex(obs['date'], freq='infer', tz=self.timezone)
                obs = obs.set_index(index)['value'].rename(series_ids[sid])
                obs = pd.to_numeric(obs, errors='coerce')
                
                if df is not None:
                    df = pd.merge(left=df, right=obs, left_index=True, right_index=True)
                else:
                    df = obs
                    
                time.sleep(2)
                
            except Exception as e:
                print(f"⚠️ Error fetching {series_ids[sid]}: {e}")
                continue
        
        # Resample to target frequency
        if df is not None and not df.empty:
            target_freq = self.get_pandas_freq()
            if target_freq == "H":
                # Forward fill daily data to hourly
                df = df.resample('H').ffill()
            # For daily, keep as-is (already daily)
            return df
            
        return pd.DataFrame()

    # =============================================================================
    # DUNE ANALYTICS DATA COLLECTION
    # =============================================================================
    
    def get_dune_latest_results(self) -> pd.DataFrame:
        """
        Retrieve latest cached results from Dune Analytics queries.
        
        Fetches data from all 21 configured Dune queries using cached results when
        available to minimize API credit consumption. Automatically detects date
        columns and aligns data to the configured timezone.
        
        Returns:
            pd.DataFrame: Combined dataset with on-chain metrics indexed by date.
                         Columns use descriptive names (e.g., 'cum_deposited_eth').
                         Returns empty DataFrame if no data is available.
                         
        Note:
            This method uses get_latest_result_dataframe() which retrieves cached
            results when available, minimizing API credit usage. Data is automatically
            saved to 'OutputData/dune_data.csv' for offline analysis.
        """
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            if not self.api_keys['dune']:
                print("❌ No Dune API key available")
                return pd.DataFrame()
            
            # Create custom session to avoid gzip decompression errors
            session = requests.Session()
            session.headers.update({'Accept-Encoding': 'identity'})  # Disable gzip
            
            # Add retry strategy for robustness
            retry = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504]
            )
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            results = {}
            dune_data = None
            # Config has query_name: query_id structure, so we need to swap
            query_mapping = {v: k for k, v in self.dune_queries.items()}  # {query_id: query_name}
            query_ids = list(query_mapping.keys())
            successful_count = 0
            failed_queries = []
            
            print(f"🔄 Processing {len(query_ids)} Dune queries...")
            
            for i, qid in enumerate(query_ids, 1):
                query_name = query_mapping[qid]
                
                try: 
                    print(f"   📊 Query {i}/{len(query_ids)}: {query_name} (ID: {qid})")
                    
                    # Use direct REST API to avoid SDK gzip issues
                    url = f"https://api.dune.com/api/v1/query/{qid}/results"
                    headers = {"X-Dune-API-Key": self.api_keys['dune']}
                    
                    response = session.get(url, headers=headers, timeout=30)
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    if 'result' not in data or 'rows' not in data['result']:
                        print(f"   ⚠️  Unexpected response format for {query_name}")
                        failed_queries.append(f"{query_name} (bad format)")
                        continue
                    
                    df = pd.DataFrame(data['result']['rows'])
                    
                    if df.empty:
                        print(f"   ⚠️  Empty response for {query_name}")
                        failed_queries.append(f"{query_name} (empty)")
                        continue
                        
                    results[query_name] = df
                    # Enhanced date detection with proper timezone handling
                    date_column_found = False
                    
                    for column in df.columns:
                        if df[column].dtype == object and not date_column_found:
                            try: 
                                # Convert to datetime with UTC assumption, then localize to target timezone
                                df[column] = pd.to_datetime(df[column], utc=True)
                                # Convert to target timezone and then to date for consistency
                                if df[column].dt.tz is not None:
                                    df[column] = df[column].dt.tz_convert(self.timezone)
                                else:
                                    df[column] = df[column].dt.tz_localize(self.timezone)
                                # Convert to date-only for consistent joining
                                df[column] = df[column].dt.date
                                df = df.rename(columns={column: 'date'})
                                df = df.set_index('date')
                                # Convert index back to DatetimeIndex for consistency
                                df.index = pd.DatetimeIndex(df.index)
                                date_column_found = True
                                break
                            except Exception as e:
                                continue
                    
                    # Process DataFrame and preserve original column names
                    if not df.empty:
                        # Remove duplicate columns within this DataFrame
                        df = df.loc[:, ~df.columns.duplicated()]
                        
                        # Save individual query data
                        df.to_csv(f"OutputData/dune_data_{qid}.csv")
                        print(f"✅ Successfully collected {query_name} query data: {df.shape}")
                        
                        # Simple joining logic to preserve all columns with original names
                        if dune_data is None: 
                            dune_data = df
                            print(f"   ✅ Set as primary dataset: {df.shape}")
                        else: 
                            before_cols = dune_data.shape[1]
                            # Use outer join - let pandas handle any conflicts naturally
                            dune_data = dune_data.join(df, how='outer', rsuffix='_dup')
                            after_cols = dune_data.shape[1]
                            added_cols = after_cols - before_cols
                            print(f"   ✅ Joined: {df.shape} -> Added {added_cols} columns (total: {after_cols})")
                        
                        successful_count += 1
                        
                except Exception as e:
                    error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
                    print(f"   ❌ Failed: {error_msg}")
                    failed_queries.append(f"{query_name} ({error_msg})")
                    continue
            
            # Save and return with enhanced diagnostics
            if dune_data is not None:
                dune_data.to_csv("OutputData/dune_data_unified.csv")
                print(f"✅ Successfully collected Dune dashboard data: {dune_data.shape}")
                print(f"   • Processed {successful_count}/{len(query_ids)} queries successfully")
                if failed_queries:
                    print(f"   • Failed queries: {failed_queries[:3]}{'...' if len(failed_queries) > 3 else ''}")
                print(f"   • Expected ~{successful_count * 3} columns, got {dune_data.shape[1]} columns")
                return dune_data
            else:
                print("❌ No Dune data collected from any queries")
                if failed_queries:
                    print(f"   • All queries failed: {failed_queries[:5]}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Collection failed: {e}")
            return pd.DataFrame()

    def get_dune_execution_results(self) -> pd.DataFrame:
        """
        Execute fresh Dune Analytics queries and retrieve results.
        
        Runs all 21 configured Dune queries using fresh execution rather than cached
        results. This method may consume API credits but ensures the most recent data.
        
        Returns:
            pd.DataFrame: Combined dataset with fresh on-chain metrics indexed by date.
                         Columns use descriptive names from DUNE_QUERIES mapping.
                         Returns empty DataFrame if execution fails.
                         
        Warning:
            This method executes fresh queries which may consume Dune API credits.
            Use get_dune_latest_results() for credit-free cached access when possible.
            
        Note:
            Results are automatically saved to 'OutputData/dune_data.csv' for caching
            and offline analysis.
        """
        try:
            from dune_client.client import DuneClient
            from dune_client.query import QueryBase
            
            dune_key = self.api_keys.get('dune')
            if not dune_key:
                print("❌ No Dune API key available")
                return pd.DataFrame()
            
            # Create custom session to avoid gzip decompression errors
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            session = requests.Session()
            session.headers.update({'Accept-Encoding': 'identity'})  # Disable gzip
            
            retry = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504]
            )
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            dune_data = None
            # Config has query_name: query_id structure, so we need to swap
            query_mapping = {v: k for k, v in self.dune_queries.items()}  # {query_id: query_name}
            query_ids = list(query_mapping.keys())
            successful_count = 0
            
            print(f"🔄 Executing {len(query_ids)} Dune queries (uses API credits)...")
            
            for i, qid in enumerate(query_ids, 1):
                query_name = query_mapping[qid]
                try:
                    print(f"   📊 Query {i}/{len(query_ids)}: {query_name} (ID: {qid})")
                    
                    # Use execute endpoint for fresh results
                    url = f"https://api.dune.com/api/v1/query/{qid}/execute"
                    headers = {"X-Dune-API-Key": dune_key}
                    
                    # Execute query
                    exec_response = session.post(url, headers=headers, timeout=30)
                    exec_response.raise_for_status()
                    exec_data = exec_response.json()
                    
                    if 'execution_id' not in exec_data:
                        print(f"   ⚠️  No execution ID returned for {query_name}")
                        continue
                    
                    # Wait a bit for execution
                    import time
                    time.sleep(2)
                    
                    # Get results
                    results_url = f"https://api.dune.com/api/v1/execution/{exec_data['execution_id']}/results"
                    results_response = session.get(results_url, headers=headers, timeout=30)
                    results_response.raise_for_status()
                    
                    data = results_response.json()
                    
                    if 'result' not in data or 'rows' not in data['result']:
                        print(f"   ⚠️  Unexpected response format for {query_name}")
                        continue
                    
                    df = pd.DataFrame(data['result']['rows'])
                    
                    if df.empty:
                        print(f"   ⚠️  Empty response for {query_name}")
                        continue
                    
                    # Enhanced date detection with proper timezone handling
                    query_name = query_mapping[qid]
                    date_column_found = False
                    
                    for column in df.columns:
                        if df[column].dtype == object and not date_column_found:
                            try: 
                                # Convert to datetime with UTC assumption, then localize to target timezone
                                df[column] = pd.to_datetime(df[column], utc=True)
                                # Convert to target timezone and then to date for consistency
                                if df[column].dt.tz is not None:
                                    df[column] = df[column].dt.tz_convert(self.timezone)
                                else:
                                    df[column] = df[column].dt.tz_localize(self.timezone)
                                # Convert to date-only for consistent joining
                                df[column] = df[column].dt.date
                                df = df.rename(columns={column: 'date'})
                                df = df.set_index('date')
                                # Convert index back to DatetimeIndex for consistency
                                df.index = pd.DatetimeIndex(df.index)
                                date_column_found = True
                                break
                            except Exception as e:
                                continue
                   
                    # Process DataFrame and preserve original column names
                    if not df.empty:
                        # Remove duplicate columns within this DataFrame
                        df = df.loc[:, ~df.columns.duplicated()]
                        
                        # Save individual query data
                        df.to_csv(f"OutputData/dune_data_{qid}.csv")
                        print(f"✅ Successfully collected {query_name} execution data: {df.shape}")
                        
                        # Simple joining logic to preserve all columns with original names
                        if dune_data is None: 
                            dune_data = df
                        else: 
                            # Use outer join - let pandas handle any conflicts naturally
                            dune_data = dune_data.join(df, how='outer', rsuffix='_dup')
                        
                except Exception as e:
                    continue
            
            # Save and return with enhanced diagnostics
            if dune_data is not None:
                dune_data.to_csv("OutputData/dune_data_unified.csv")
                print(f"✅ Successfully executed Dune dashboard data: {dune_data.shape}")
                print(f"   • Processed {len(query_ids)} queries with enhanced column preservation")
                print(f"   • Final dataset: {dune_data.shape[1]} columns across {dune_data.shape[0]} rows")
                return dune_data
            else:
                print("❌ No Dune data collected from any queries")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Execution failed: {e}")
            return pd.DataFrame()

    def get_dune_data(self, allow_execution: bool = False, try_csv_fallback: bool = True) -> pd.DataFrame:
        """
        Main interface for Dune Analytics data collection with intelligent fallbacks.
        
        Provides a robust data collection pipeline that tries multiple sources:
        1. API cached results (free) or fresh execution (uses credits)
        2. Local CSV cache as fallback
        3. Empty DataFrame if all sources fail
        
        Args:
            allow_execution (bool): Whether to allow fresh query execution
                - False: Use cached results only (recommended, credit-free)
                - True: Allow fresh query execution (may use credits)
            try_csv_fallback (bool): Whether to try CSV cache if API fails
                
        Returns:
            pd.DataFrame: On-chain metrics dataset indexed by date with descriptive
                         column names. Returns empty DataFrame if all sources fail.
                         
        Note:
            Using allow_execution=False with try_csv_fallback=True provides the most
            robust data collection with multiple fallback options and no API credits.
        """
        
        # Try API first if key is available
        csv_data = pd.DataFrame()
        if self.api_keys['dune']:
            try:
                if allow_execution:
                    data = self.get_dune_execution_results()
                else:
                    data = self.get_dune_latest_results()
                    
                if not data.empty:
                    return data
                else:
                    print("💡 No API data available, trying CSV fallback...")
                    
            except Exception as e:
                print(f"⚠️ Dune API failed: {e}")
                if try_csv_fallback:
                    print("💡 Trying CSV fallback...")
                    csv_path = "OutputData/dune_data_unified.csv"
                    csv_data = self._load_dune_csv(csv_path)
                else: 
                    print("💡 CSV fallback disabled")
                if not csv_data.empty:
                    print(f"✅ Loaded Dune data from CSV: {csv_path}")
                    return csv_data
                else:
                    print("💡 No CSV cache found in any location")
        else:
            print("❌ No Dune API key available")
            if try_csv_fallback:
                print("💡 Trying CSV fallback...")
                csv_path = "OutputData/dune_data_unified.csv"
                csv_data = self._load_dune_csv(csv_path)
            else: 
                print("💡 CSV fallback disabled")
            if not csv_data.empty:
                print(f"✅ Loaded Dune data from CSV: {csv_path}")
                return csv_data
            else:
                print("💡 No CSV cache found in any location")
        
        # All sources failed
        print("⚠️ No Dune data available from any source")
        return pd.DataFrame()
        return pd.DataFrame()

    def _load_dune_csv(self, csv_path: str) -> pd.DataFrame:
        """Load Dune data from CSV file."""
        if not os.path.exists(csv_path):
            return pd.DataFrame()
            
        try:
            df = pd.read_csv(csv_path, index_col=0)
            # Convert to date directly (matching Dune API processing)
            date_index = pd.to_datetime(df.index).date
            df.index = pd.DatetimeIndex(date_index)  # Convert back to DatetimeIndex
            df.index.name = "date"
            return df
        except Exception as e:
            print(f"⚠️  CSV loading failed: {str(e)[:50]}...")
            return pd.DataFrame()

    # =============================================================================
    # UNIFIED DATA COLLECTION METHODS
    # =============================================================================

    def collect_all_data(self) -> Dict[str, pd.DataFrame]:
        """Collect data from all sources and return as dictionary."""
        print("🔄 Starting data collection...")
        
        # Get universe
        universe = self.coingecko_get_universe(self.top_n, output_format="both")
        if isinstance(universe, dict):
            ids, tickers = universe["ids"], universe["ticker"]
        else:
            print("❌ Failed to get universe data")
            return {}
        
        data = {}
        
        # Collect price data silently
        data['binance_price'] = self.binance_get_price_action(ids=ids, tickers=tickers, 
                                                              max_days=self.lookback_days)
        
        data['coingecko_price'] = self.coingecko_get_price_action(ids, start=self.start_date)
        
        data['dvol'] = self.deribit_get_dvol(['BTC', 'ETH'], days=self.lookback_days)
        
        data['onchain'] = self.get_dune_data(allow_execution=not self.use_cached_dune_only)
        
        data['macro'] = self.fred_get_series(series_ids=self.fred_series, start=self.start_date)
        
        print("✅ Data collection completed")
        return data

    def combine_data_sources(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine all data sources into unified DataFrame."""
        
        if not data:
            print("⚠️ No data sources provided")
            return pd.DataFrame()
        
        unified = None
        successful = []
        failed = []
        
        for name, df in data.items():
            if df is None or df.empty:
                failed.append(f"{name} (empty)")
                continue
                
            try:
                # Standardize timezone
                if df.index.tz is None:
                    df.index = pd.DatetimeIndex(df.index).tz_localize(self.timezone).date
                else:
                    df.index = pd.DatetimeIndex(df.index).tz_convert(self.timezone).date
                    
                if unified is None:
                    unified = df
                else:
                    unified = unified.join(df, how='outer')
                
                successful.append(f"{name} ({len(df)} rows)")
                unified.index = pd.to_datetime(unified.index)
            except Exception as e:
                failed.append(f"{name} ({str(e)[:30]}...)")
                continue
        
        # Summary reporting
        if successful:
            print(f"✅ Combined {len(successful)} sources: {', '.join(successful)}")
        if failed:
            print(f"⚠️  Failed {len(failed)} sources: {', '.join(failed)}")
        
        return unified if unified is not None else pd.DataFrame()


if __name__ == "__main__":
    # Test the collector
    collector = CryptoDataCollector(top_n=5, lookback_days=30, frequency="1D")
    print(f"Testing collector with frequency: {collector.frequency}")
    data = collector.collect_all_data()
    unified = collector.combine_data_sources(data)
    print(f"Final dataset shape: {unified.shape}")
