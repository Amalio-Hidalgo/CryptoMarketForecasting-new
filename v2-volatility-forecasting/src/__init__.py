"""
Cryptocurrency Volatility Forecasting Toolkit

Educational demonstration of advanced cryptocurrency volatility forecasting
using multi-source data integration, time series feature engineering, and
machine learning optimization.

This toolkit showcases professional software architecture patterns for
quantitative finance applications, including distributed computing,
automated feature extraction, and hyperparameter optimization.

Key Components:
    - Multi-source data collection (CoinGecko, Binance, Deribit, FRED, Dune Analytics)
    - Advanced time series feature extraction with TSFresh
    - Technical analysis indicators using TA-Lib
    - Distributed processing with Dask computing clusters
    - Machine learning with XGBoost and Optuna optimization
    - Comprehensive evaluation and visualization tools

License:
    Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
    See LICENSE file for full terms and conditions.

Author:
    Amalio Hidalgo - HEC Paris MiF Program
    Portfolio demonstration of quantitative finance and machine learning skills
"""

__version__ = "2.0.0"
__author__ = "Amalio Hidalgo"
__license__ = "CC BY-NC-ND 4.0"
__status__ = "Educational Demo"

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