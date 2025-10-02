#!/usr/bin/env python3
"""
Setup script for Cryptocurrency Volatility Forecasting Toolkit
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    """Read README.md file for long description"""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Cryptocurrency Volatility Forecasting Toolkit"

# Read requirements
def read_requirements():
    """Read requirements.txt file"""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="crypto-volatility-forecast",
    version="1.0.0",
    author="Amalio Hidalgo",
    author_email="amalio.hidalgo-pickrell@hec.edu",
    description="Cryptocurrency volatility forecasting demonstration with TSFresh and XGBoost",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Amalio-Hidalgo/crypto-volatility-forecast",
    license="CC BY-NC-ND 4.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        # Core ML and distributed computing extensions
        "distributed": [
            "dask[complete]>=2023.1.0",
        ],
        "technical": [
            "TA-Lib>=0.4.25",  # Note: May require system-level installation
        ],
        "optimization": [
            "optuna>=3.0.0",
            "xgboost>=1.7.0",
        ],
        # Development and testing tools
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
        ],
        # Documentation generation
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        # Cloud deployment support
        "cloud": [
            "coiled>=0.3.0",
            "s3fs>=2022.8.0",
            "gcsfs>=2022.8.0",
        ],
        # Complete installation
        "all": [
            "dask[complete]>=2023.1.0",
            "TA-Lib>=0.4.25", 
            "optuna>=3.0.0",
            "xgboost>=1.7.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "crypto-forecast=cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/Amalio-Hidalgo/crypto-volatility-forecast/issues",
        "Source": "https://github.com/Amalio-Hidalgo/crypto-volatility-forecast",
        "Contact (Professional)": "mailto:amalio.hidalgo-pickrell@hec.edu",
        "Contact (Permanent)": "mailto:amaliohidalgo1@gmail.com",
    },
    include_package_data=True,
    package_data={
        "src": ["config/*.json"],
    },
    keywords=[
        "cryptocurrency", 
        "volatility", 
        "forecasting", 
        "machine-learning", 
        "time-series",
        "tsfresh", 
        "xgboost", 
        "dask", 
        "binance", 
        "coingecko",
        "financial-modeling",
        "quantitative-finance"
    ],
    zip_safe=False,
)