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
        TIMEZONE (str): Target timezone for data alignment
        TOP_N (int): Number of top cryptocurrencies to collect
        LOOKBACK_DAYS (int): Historical data window in days
        FREQUENCY (str): Data frequency ("1D" for daily, "1H" for hourly)
        DUNE_QUERIES (dict): Mapping of Dune query IDs to descriptive names
        FRED_KNOWN (dict): Mapping of FRED series IDs to descriptive names
    """
    
    def __init__(self, 
                 timezone: str = "Europe/Madrid",
                 top_n: int = 10,
                 lookback_days: int = 365,
                 frequency: str = "1D",
                 use_cached_dune_only: bool = True):
        """
        Initialize the cryptocurrency data collector.
        
        Sets up API connections, configures collection parameters, and initializes
        data source mappings. All timestamps will be aligned to the specified timezone.
        
        Args:
            timezone (str): Target timezone for data alignment (e.g., "UTC", "Europe/Madrid")
            top_n (int): Number of top cryptocurrencies by market cap to collect
            lookback_days (int): Number of days of historical data to retrieve
            frequency (str): Data collection frequency - "1D" for daily, "1H" for hourly
            use_cached_dune_only (bool): True=use cached data only, False=allow fresh execution
                                       (False may consume API credits)
        
        Raises:
            EnvironmentError: If required API keys are not found in environment variables
            
        Note:
            Requires the following environment variables:
            - COINGECKO_API_KEY: For CoinGecko API access
            - DUNE_API_KEY: For Dune Analytics API access  
            - FRED_API_KEY: For Federal Reserve Economic Data API access
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
        
        # Dune configuration - simplified approach
        self.USE_CACHED_DUNE_ONLY = use_cached_dune_only
        
        # Dune Analytics query configuration (21 curated on-chain metrics)
        # Maps query IDs to descriptive names for better data interpretation
        self.DUNE_QUERIES = {
            5893929: "cum_deposited_eth",
            5893461: "economic_security", 
            5893557: "btc_etf_flows",
            5893307: "eth_etf_flows", 
            5894092: "total_defi_users",
            5894035: "median_gas",
            5893555: "staked_eth_category",
            5893552: "lsd_share",
            5893566: "lsd_tvl",
            5893781: "staking_rewards",
            5893821: "validator_performance",
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
        
        # FRED macroeconomic indicators configuration
        # Maps FRED series IDs to descriptive names for volatility forecasting
        self.FRED_KNOWN = {
            "VIXCLS": "vix_equity_vol",         # CBOE Volatility Index
            "MOVE": "move_bond_vol",            # ICE BofAML MOVE Index  
            "OVXCLS": "ovx_oil_vol",           # CBOE Crude Oil ETF Volatility Index
            "GVZCLS": "gvz_gold_vol",          # CBOE Gold ETF Volatility Index
            "DTWEXBGS": "usd_trade_weighted_index",  # Trade Weighted U.S. Dollar Index
            "DGS2": "us_2y_treasury_yield",    # 2-Year Treasury Constant Maturity Rate
            "DGS10": "us_10y_treasury_yield",  # 10-Year Treasury Constant Maturity Rate
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
            from dune_client.client import DuneClient
            from dune_client.query import QueryBase
            
            if not self.DUNE_API_KEY:
                print("❌ No Dune API key available")
                return pd.DataFrame()
            
            dune = DuneClient(self.DUNE_API_KEY)
            results = {}
            dune_data = None
            query_ids = list(self.DUNE_QUERIES.keys())
            successful_count = 0
            failed_queries = []
            
            print(f"🔄 Processing {len(query_ids)} Dune queries...")
            
            for i, qid in enumerate(query_ids, 1):
                query = QueryBase(query_id=qid)
                query_name = self.DUNE_QUERIES[qid]
                
                try: 
                    print(f"   📊 Query {i}/{len(query_ids)}: {query_name} (ID: {qid})")
                    response = dune.get_latest_result_dataframe(query)
                    
                    if response is None or response.empty:
                        print(f"   ⚠️  Empty response for {query_name}")
                        failed_queries.append(f"{query_name} (empty)")
                        continue
                        
                    results[query_name] = response
                    df = response.copy()
                    # Enhanced date detection and indexing logic with conflict prevention
                    query_name = self.DUNE_QUERIES[qid]
                    date_column_found = False
                    
                    for column in df.columns:
                        if df[column].dtype == object and not date_column_found:
                            try: 
                                # Convert to datetime and then to date
                                df[column] = pd.to_datetime(df[column])
                                df = df.rename(columns={column: 'date'})
                                df = df.set_index('date')
                                date_column_found = True
                                break
                            except: 
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
            
            if not self.DUNE_API_KEY:
                print("❌ No Dune API key available")
                return pd.DataFrame()
            
            dune = DuneClient(self.DUNE_API_KEY)
            dune_data = None
            query_ids = list(self.DUNE_QUERIES.keys())
            
            for qid in query_ids:
                query = QueryBase(query_id=qid)
                try: 
                    response = dune.get_execution_results_csv(query)
                    df = pd.DataFrame(response)
                    
                    # Enhanced date detection and indexing logic with conflict prevention  
                    query_name = self.DUNE_QUERIES[qid]
                    date_column_found = False
                    
                    for column in df.columns:
                        if df[column].dtype == object and not date_column_found:
                            try: 
                                # Convert to datetime and then to date
                                df[column] = pd.to_datetime(df[column])
                                df = df.rename(columns={column: 'date'})
                                df = df.set_index('date')
                                date_column_found = True
                                break
                            except: 
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
        if self.DUNE_API_KEY:
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
                    csv_path = ["OutputData/dune_data_unified.csv"]
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
                csv_path = ["OutputData/dune_data_unified.csv"]
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
        # Simplified logic: choose method based on execution permission
        try:
            if force_refresh and allow_execution:
                return self.get_dune_execution_results()
            else:
                return self.get_dune_latest_results()
        except Exception as e:
            print(f"⚠️ Dune data collection failed: {e}")
            return pd.DataFrame()

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
        
        data['onchain'] = self.get_dune_data(allow_execution=not self.USE_CACHED_DUNE_ONLY)
        
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
        data['onchain'] = self.get_dune_data(allow_execution=False)
        
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
                                  use_cached_dune_only=True)
    data = collector.collect_all_data()
    return collector.combine_data_sources(data)

def collect_crypto_data_with_cached_dune(top_n: int = 10, 
                                         lookback_days: int = 365,
                                         timezone: str = "Europe/Madrid",
                                         frequency: str = "1D") -> pd.DataFrame:
    """
    SAFE function to collect all crypto data including cached Dune results.
    🔒 NO QUERY EXECUTION - uses only cached results, less API credits consumed.
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
                                  use_cached_dune_only=False)
    data = collector.collect_all_data()
    return collector.combine_data_sources(data)


if __name__ == "__main__":
    # Test the collector
    collector = CryptoDataCollector(top_n=5, lookback_days=30, frequency="1D")
    print(f"Testing collector with frequency: {collector.FREQUENCY}")
    data = collector.collect_all_data()
    unified = collector.combine_data_sources(data)
    print(f"Final dataset shape: {unified.shape}")