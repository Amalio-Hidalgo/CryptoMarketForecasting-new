"""
Utility modules for cryptocurrency volatility forecasting.

This module provides utility functions including:
- Dask cluster management
- Data processing helpers
- Visualization utilities
"""

from .dask_helpers import create_optimized_dask_client, cleanup_dask_client

__all__ = [
    "create_optimized_dask_client",
    "cleanup_dask_client",
]