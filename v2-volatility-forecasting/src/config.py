"""
Configuration Management for Cryptocurrency Volatility Forecasting

This module provides centralized configuration management for the volatility forecasting
pipeline. It defines data collection parameters, feature engineering settings, model
hyperparameters, and API configuration with sensible defaults.

The configuration is organized into logical groups using dataclasses for type safety
and ease of use. All settings can be overridden when initializing the pipeline.

Key Configuration Areas:
    - Data Collection: API settings, data sources, time ranges
    - Feature Engineering: TSFresh settings, technical indicators
    - Machine Learning: Model parameters, optimization settings
    - Infrastructure: Dask cluster, caching, output paths

Environment Variables:
    The following environment variables should be set for full functionality:
    - COINGECKO_API_KEY: CoinGecko Pro API key
    - DUNE_API_KEY: Dune Analytics API key  
    - FRED_API_KEY: Federal Reserve Economic Data API key

Example:
    ```python
    from config import DataConfig, MLConfig
    
    # Use default configuration
    data_config = DataConfig()
    
    # Override specific settings
    custom_config = DataConfig(
        target_coin="bitcoin",
        lookback_days=730,
        frequency="1H"
    )
    ```
"""

import os
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import datetime as dt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class DataConfig:
    """
    Data collection configuration parameters.
    
    Defines settings for collecting cryptocurrency market data, on-chain metrics,
    volatility indices, and macroeconomic indicators. All data sources are aligned
    to the specified timezone and frequency.
    
    Attributes:
        target_coin (str): Primary cryptocurrency for volatility forecasting
        base_fiat (str): Base fiat currency for price data  
        top_n (int): Number of top cryptocurrencies to include
        lookback_days (int): Historical data window in days
        timezone (str): Target timezone for data alignment
        frequency (str): Data collection frequency ("1D" daily, "1H" hourly)
        sleep_time (int): Delay between API calls to respect rate limits
        use_cached_dune_only (bool): True=use cached data only, False=allow fresh execution
        key_dune_queries (List[int], optional): Specific Dune queries to execute
    """
    target_coin: str = "ethereum"
    base_fiat: str = "usd"
    top_n: int = 10
    lookback_days: int = 365
    timezone: str = "Europe/Madrid"
    frequency: str = "1D"
    sleep_time: int = 10  # seconds between API calls
    
    # 🛡️ Dune API Credit Protection Settings
    use_cached_dune_only: bool = True  # SAFE DEFAULT: True=cached only, False=allow fresh execution
    
    # Recommended Dune queries for volatility forecasting (21 curated queries)
    key_dune_queries: Optional[List[int]] = None  # Will use default set: [5893929, 5893461, 5893557, 5893307, 5894092, 5894035, 5893555, 5893552, 5893566, 5893781, 5893821, 5893009, 5892998, 5893911, 5892742, 5892720, 5891651, 5892696, 5892424, 5892227, 5891691]


@dataclass
class TSFreshConfig:
    """
    TSFresh feature engineering configuration.
    
    Controls automated time series feature extraction using the TSFresh library.
    Features are extracted over rolling windows and filtered for relevance.
    
    Attributes:
        time_window (int): Rolling window size in days for feature extraction
        extraction_settings (str): Feature extraction complexity level
                                  - "minimal": Basic statistical features
                                  - "efficient": Balanced feature set (recommended)
                                  - "comprehensive": All available features
        fdr_level (float): False discovery rate for feature selection
        random_seed (int): Random seed for reproducible feature extraction
    """
    time_window: int = 14  # days for rolling window
    extraction_settings: str = "efficient"  # minimal, efficient, comprehensive
    fdr_level: float = 0.05
    random_seed: int = 42


@dataclass
class MLConfig:
    """
    Machine learning pipeline configuration.
    
    Defines parameters for model training, hyperparameter optimization, and
    cross-validation. Uses XGBoost as the primary model with Optuna optimization.
    
    Attributes:
        n_trials (int): Number of Optuna hyperparameter optimization trials
        n_rounds (int): Maximum XGBoost boosting rounds
        early_stopping (int): Early stopping patience for XGBoost training
        n_splits (int): Number of time series cross-validation splits
        fdr_level (float): False discovery rate for feature selection
        random_seed (int): Random seed for reproducible model training  
        tree_method (str): XGBoost tree construction algorithm
        eval_metric (str): Model evaluation metric for optimization
    """
    n_trials: int = 100  # Optuna trials
    n_rounds: int = 200  # XGBoost rounds
    eval_metric: str = 'mae'
    tree_method: str = 'hist'
    early_stopping_rounds: int = 25
    splits: int = 10  # Time series folds
    random_seed: int = 42


@dataclass
class DaskConfig:
    """Dask distributed computing configuration."""
    n_workers: int = 4
    threads_per_worker: int = 5
    memory_limit: str = '8GB'
    dashboard_port: int = 8787
    processes: bool = True


@dataclass
class APIConfig:
    """API configuration and endpoints."""
    # API Keys loaded from environment variables
    coingecko_api_key: str = ""
    dune_api_key: str = ""
    lunarcrush_api_key: str = ""
    lunarcrush_bearer_token: str = ""
    deribit_api_key: str = ""
    x_api_key: str = ""
    x_api_secret: str = ""
    x_bearer_token: str = ""
    merlin_api_key: str = ""
    merlin_user_address: str = ""
    
    # Dune Analytics queries
    dune_queries: Optional[Dict[str, int]] = None
    
    # FRED series mappings  
    fred_series: Optional[Dict[str, str]] = None
    
    # File paths
    dune_csv_path: str = "OutputData/Dune_Metrics.csv"
    
    def __post_init__(self):
        # Load API keys from environment variables
        self.coingecko_api_key = os.getenv('COINGECKO_API_KEY', '')
        self.dune_api_key = os.getenv('DUNE_API_KEY_2', '')  # Using DUNE_API_KEY_2
        self.lunarcrush_api_key = os.getenv('LUNARCRUSH_API_KEY', '')
        self.lunarcrush_bearer_token = os.getenv('LUNARCRUSH_BEARER_TOKEN', '')
        self.deribit_api_key = os.getenv('DERIBIT_API_KEY', '')
        self.x_api_key = os.getenv('X_API_KEY', '')
        self.x_api_secret = os.getenv('X_API_SECRET', '')
        self.x_bearer_token = os.getenv('X_BEARER_TOKEN', '')
        self.merlin_api_key = os.getenv('MERLIN_API_KEY', '')
        self.merlin_user_address = os.getenv('MERLIN_USER_ADDRESS', '')
        
        if self.dune_queries is None:
            # Daily query IDs (21 total)
            self.dune_queries = {
                "query_01": 5893929,
                "query_02": 5893461,
                "query_03": 5893557,
                "query_04": 5893307,
                "query_05": 5894092,
                "query_06": 5894035,
                "query_07": 5893555,
                "query_08": 5893552,
                "query_09": 5893566,
                "query_10": 5893781,
                "query_11": 5893821,
                "query_12": 5893009,
                "query_13": 5892998,
                "query_14": 5893911,
                "query_15": 5892742,
                "query_16": 5892720,
                "query_17": 5891651,
                "query_18": 5892696,
                "query_19": 5892424,
                "query_20": 5892227,
                "query_21": 5891691
            }

# Dune Query Details:
#  All available on my dune dashboards: 
# Daily Frequency:https://dune.com/amaliohidalgo_team_4477/crypto-market-volatility-forecast-indicators-daily
# query_01 (5893929): cum_deposited_eth - Measures total ETH staked over time, indicating network participation.
# query_02 (5893461): economic_security - Assesses the financial security of the Ethereum network by valuing staked ETH in USD.
# query_03 (5893557): btc_etf_flows - Monitors Bitcoin ETF inflows/outflows, reflecting institutional sentiment.
# query_04 (5893307): eth_etf_flows - Monitors Ethereum ETF inflows/outflows, reflecting institutional sentiment.
# query_05 (5894092): total_defi_users - Counts unique users interacting with DeFi protocols, indicating ecosystem growth.
# query_06 (5894035): median_gas - Measures median gas prices on Ethereum, reflecting network congestion and user costs.
# query_07 (5893555): staked_eth_category - Analyzes staked ETH distribution across different categories.
# query_08 (5893552): lsd_share - Tracks liquid staking derivatives market share and distribution.
# query_09 (5893566): lsd_tvl - Monitors total value locked in liquid staking derivative protocols.
# query_10 (5893781): staking_rewards - Tracks staking rewards and yield metrics for validators.
# query_11 (5893821): validator_performance - Monitors validator performance metrics and attestation rates.
# query_12 (5893009): network_activity - Measures daily active addresses and transaction metrics.
# query_13 (5892998): defi_tvl - Tracks total value locked across DeFi protocols on Ethereum.
# query_14 (5893911): nft_volume - Monitors NFT trading volume and market activity metrics.
# query_15 (5892742): bridge_activity - Tracks cross-chain bridge volumes and activity.
# query_16 (5892720): mev_activity - Monitors MEV (Maximal Extractable Value) metrics and trends.
# query_17 (5891651): lending_metrics - Tracks lending protocol metrics including borrow/supply rates.
# query_18 (5892696): derivative_volume - Monitors on-chain derivatives trading volume.
# query_19 (5892424): governance_activity - Tracks DAO governance participation and voting metrics.
# query_20 (5892227): yield_farming - Monitors yield farming rewards and liquidity mining metrics.
# query_21 (5891691): perpetual_volume - Tracks perpetual futures trading volume on-chain.
# 
# Hourly Frequency: **Coming Soon**

        if self.fred_series is None:
            self.fred_series = {
                "VIXCLS": "vix_equity_vol",
                "MOVE": "move_bond_vol",
                "OVXCLS": "ovx_oil_vol",
                "GVZCLS": "gvz_gold_vol",
                "DTWEXBGS": "usd_trade_weighted_index",
                "DGS2": "us_2y_treasury_yield",
                "DGS10": "us_10y_treasury_yield",
            }
# VIXCLS: CBOE Volatility Index, measures market's expectation of 30-day volatility for S&P 500.
# MOVE: Merrill Lynch Option Volatility Estimate, indicates expected volatility in the bond market.
# OVXCLS: CBOE Crude Oil Volatility Index, reflects market's expectation of 30-day volatility for crude oil.
# GVZCLS: CBOE Gold Volatility Index, indicates market's expectation of 30-day volatility for gold.
# DTWEXBGS: Trade Weighted U.S. Dollar Index, measures the value of USD against a basket of foreign currencies.
# DGS2: 2-Year Treasury Constant Maturity Rate, indicates short-term interest rates.
# DGS10: 10-Year Treasury Constant Maturity Rate, indicates long-term interest rates.

@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig
    tsfresh: TSFreshConfig
    ml: MLConfig
    dask: DaskConfig
    api: APIConfig
    
    # Computed properties
    start_date: str = ""
    today: str = ""
    
    def __post_init__(self):
        """Compute derived values."""
        self.start_date = (
            dt.datetime.now() - dt.timedelta(days=self.data.lookback_days)
        ).strftime("%Y-%m-%d")
        self.today = dt.date.today().strftime('%Y-%m-%d')


def create_default_config() -> Config:
    """Create default configuration matching LatestNotebook.ipynb."""
    return Config(
        data=DataConfig(),
        tsfresh=TSFreshConfig(),
        ml=MLConfig(),
        dask=DaskConfig(),
        api=APIConfig()
    )


def load_config_from_dict(config_dict: Dict[str, Any]) -> Config:
    """Load configuration from dictionary."""
    return Config(
        data=DataConfig(**config_dict.get('data', {})),
        tsfresh=TSFreshConfig(**config_dict.get('tsfresh', {})),
        ml=MLConfig(**config_dict.get('ml', {})),
        dask=DaskConfig(**config_dict.get('dask', {})),
        api=APIConfig(**config_dict.get('api', {}))
    )


def load_config_from_file(config_path: str = "config.json") -> Config:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration object
    """
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found. Using defaults.")
        return create_default_config()
    
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return load_config_from_dict(config_dict)
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        print("Using default configuration.")
        return create_default_config()


def save_config_to_file(config: Config, config_path: str = "config.json") -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration object
        config_path: Path to save configuration
    """
    try:
        # Convert to dictionary (excluding computed properties)
        config_dict = {
            'data': {
                'target_coin': config.data.target_coin,
                'base_fiat': config.data.base_fiat,
                'top_n': config.data.top_n,
                'lookback_days': config.data.lookback_days,
                'timezone': config.data.timezone,
                'frequency': config.data.frequency,
                'sleep_time': config.data.sleep_time,
                'use_cached_dune_only': config.data.use_cached_dune_only
            },
            'tsfresh': {
                'time_window': config.tsfresh.time_window,
                'extraction_settings': config.tsfresh.extraction_settings,
                'fdr_level': config.tsfresh.fdr_level,
                'random_seed': config.tsfresh.random_seed
            },
            'ml': {
                'n_trials': config.ml.n_trials,
                'n_rounds': config.ml.n_rounds,
                'eval_metric': config.ml.eval_metric,
                'tree_method': config.ml.tree_method,
                'early_stopping_rounds': config.ml.early_stopping_rounds,
                'splits': config.ml.splits,
                'random_seed': config.ml.random_seed
            },
            'dask': {
                'n_workers': config.dask.n_workers,
                'threads_per_worker': config.dask.threads_per_worker,
                'memory_limit': config.dask.memory_limit,
                'dashboard_port': config.dask.dashboard_port,
                'processes': config.dask.processes
            },
            'api': {
                'dune_queries': config.api.dune_queries,
                'fred_series': config.api.fred_series,
                'dune_csv_path': config.api.dune_csv_path
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
        print(f"✅ Configuration saved to {config_path}")
        
    except Exception as e:
        print(f"Error saving config to {config_path}: {e}")


def get_api_keys() -> Dict[str, Optional[str]]:
    """
    Get API keys from environment variables.
    
    Returns:
        Dictionary with API keys
    """
    return {
        'coingecko': os.getenv("COINGECKO_API_KEY"),
        'dune': os.getenv("DUNE_API_KEY_2"),
        'fred': os.getenv("FRED_API_KEY")
    }


def validate_api_keys() -> Dict[str, bool]:
    """
    Validate that required API keys are available.
    
    Returns:
        Dictionary with validation results
    """
    api_keys = get_api_keys()
    validation = {}
    
    for api, key in api_keys.items():
        validation[api] = key is not None and len(key.strip()) > 0
        
    return validation


def print_config_summary(config: Config) -> None:
    """Print a summary of the current configuration."""
    print("📋 Configuration Summary:")
    print("=" * 50)
    
    print(f"🎯 Target: {config.data.target_coin} ({config.data.base_fiat})")
    print(f"📊 Data: Top {config.data.top_n} coins, {config.data.lookback_days} days")
    print(f"🕐 Period: {config.start_date} to {config.today}")
    print(f"Timezone: {config.data.timezone}")
    
    print(f"\nTSFresh: {config.tsfresh.extraction_settings} features, {config.tsfresh.time_window}d window")
    print(f"FDR Level: {config.tsfresh.fdr_level}")
    
    print(f"\nML: {config.ml.n_trials} trials, {config.ml.n_rounds} rounds")
    print(f"Metric: {config.ml.eval_metric}, Early stopping: {config.ml.early_stopping_rounds}")
    
    print(f"\nDask: {config.dask.n_workers} workers, {config.dask.threads_per_worker} threads/worker")
    print(f"Memory: {config.dask.memory_limit}/worker")
    
    # API key validation
    api_validation = validate_api_keys()
    print(f"\nAPI Keys:")
    for api, valid in api_validation.items():
        status = "VALID" if valid else "MISSING"
        print(f"   {api.upper()}: {status}")


# Convenience function
def get_default_config() -> Config:
    """Get default configuration for quick testing."""
    return create_default_config()


if __name__ == "__main__":
    # Test configuration
    print("Testing configuration module...")
    
    config = create_default_config()
    print_config_summary(config)
    
    # Test saving and loading
    save_config_to_file(config, "test_config.json")
    loaded_config = load_config_from_file("test_config.json")
    
    print(f"✅ Config test completed!")
    print(f"Original target coin: {config.data.target_coin}")
    print(f"Loaded target coin: {loaded_config.data.target_coin}")
    
    # Clean up
    if os.path.exists("test_config.json"):
        os.remove("test_config.json")