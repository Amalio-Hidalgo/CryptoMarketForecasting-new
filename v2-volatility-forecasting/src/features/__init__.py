"""
Feature engineering module for cryptocurrency volatility forecasting.

This module provides feature engineering capabilities including:
- TSFresh time series feature extraction
- Technical analysis indicators via TA-Lib
- Distributed processing with Dask
- Target variable preparation
"""

from .engineering import CryptoFeatureEngineer

__all__ = ["CryptoFeatureEngineer"]