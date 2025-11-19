"""
Machine Learning Pipeline Module

Provides XGBoost-based volatility forecasting with Optuna hyperparameter optimization.
"""

from .pipeline import CryptoVolatilityMLPipeline

__all__ = ['CryptoVolatilityMLPipeline']
