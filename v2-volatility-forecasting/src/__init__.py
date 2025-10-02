"""
Cryptocurrency Volatility Forecasting Toolkit

Copyright (c) 2025 Amalio Hidalgo. All rights reserved.
This software is proprietary. See LICENSE file for terms.

A comprehensive toolkit for cryptocurrency volatility forecasting using
advanced time series analysis, feature engineering, and machine learning.

Key components:
- Multi-source data collection (CoinGecko, Binance, Deribit, FRED, Dune)
- TSFresh time series feature extraction
- Technical analysis indicators (TA-Lib)
- Distributed processing with Dask
- XGBoost machine learning with Optuna optimization
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes for easy access
from .config import Config, load_config_from_file
from .data.collectors import CryptoDataCollector
from .features.engineering import CryptoFeatureEngineer
from .models.pipeline import CryptoVolatilityMLPipeline

__all__ = [
    "Config",
    "load_config_from_file", 
    "CryptoDataCollector",
    "CryptoFeatureEngineer",
    "CryptoVolatilityMLPipeline",
]