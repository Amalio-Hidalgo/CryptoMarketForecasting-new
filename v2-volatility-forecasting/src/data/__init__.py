"""
Data collection module for cryptocurrency volatility forecasting.

This module provides data collectors for various cryptocurrency and 
macroeconomic data sources including CoinGecko, Binance, Deribit DVOL,
FRED macroeconomic data, and Dune Analytics.
"""

from .collectors import CryptoDataCollector

__all__ = ["CryptoDataCollector"]