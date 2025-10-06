"""
Data Collection Module

Provides comprehensive data collection capabilities for cryptocurrency volatility
forecasting from multiple APIs and data sources. All collectors implement unified
interfaces with automatic timezone alignment, frequency conversion, and error handling.

Available Data Sources:
    - CoinGecko: Cryptocurrency market data and historical prices
    - Binance: High-frequency OHLCV data with extended history
    - Deribit: Bitcoin and Ethereum implied volatility indices (DVOL)
    - FRED: Macroeconomic indicators and traditional asset volatility
    - Dune Analytics: On-chain metrics and DeFi protocol analytics

Main Classes:
    CryptoDataCollector: Unified data collection interface supporting all sources

Example:
    ```python
    from data import CryptoDataCollector
    
    collector = CryptoDataCollector(
        timezone="UTC",
        top_n=10,
        lookback_days=365
    )
    
    # Collect all data sources
    all_data = collector.collect_all_data()
    
    # Get specific data source
    dune_data = collector.get_dune_latest_results()
    ```

Note:
    Requires appropriate API keys set as environment variables.
    See collectors.py for detailed API configuration requirements.
"""

from .collectors import CryptoDataCollector

__all__ = ["CryptoDataCollector"]