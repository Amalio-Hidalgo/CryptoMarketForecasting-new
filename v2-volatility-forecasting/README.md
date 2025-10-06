# Cryptocurrency Volatility Forecasting Toolkit

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](../LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive demonstration of cryptocurrency volatility prediction techniques, showcasing distributed computing, advanced time series feature engineering, and machine learning optimization. This implementation demonstrates professional software architecture patterns while integrating multiple data sources with TSFresh feature extraction and XGBoost modeling for sophisticated volatility analysis.

## License and Usage

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License - see the [LICENSE](../LICENSE) file for details.

**Educational and Portfolio Use:**
- ✅ Viewing and studying code for learning purposes
- ✅ Referencing in academic or professional contexts
- ✅ Using for technical interviews and skill demonstration
- ✅ Personal learning and development

**Prohibited Uses:**
- ❌ Commercial use or monetization
- ❌ Production deployment for trading or financial services
- ❌ Redistribution of modified versions
- ❌ Integration into commercial products

**Important**: This software is provided for educational demonstration. Users are responsible for ensuring compliance with all applicable data provider terms of service and API usage agreements.

## Core Components

### Data Collection Infrastructure
The toolkit aggregates data from five primary sources, each serving a specific analytical purpose- This multi-source approach addresses the limitation of single-source bias common in cryptocurrency analysis and provides a comprehensive view of market dynamics.:

- **CoinGecko API**: Provides comprehensive cryptocurrency universe data including daily/hourly OHLCV time series, market capitalization rankings, trading volume metrics, and social sentiment indicators. Delivers up to 365 days of historical data with 1-hour granularity for volatility modeling and supports over 10,000 cryptocurrency assets with normalized price feeds.

- **Binance API**: Delivers high-frequency price action data with 1-minute to 1-day OHLCV granularity, supporting up to 1000 data points per request. Provides real-time order book depth, trade execution data, and 24-hour rolling statistics essential for short-term volatility pattern recognition and market microstructure analysis.

- **Deribit DVOL**: Supplies Bitcoin and Ethereum implied volatility indices calculated from options pricing models, updated every 10 minutes during market hours. These forward-looking volatility measures complement historical price volatility and provide market sentiment indicators crucial for volatility forecasting accuracy.

- **FRED (Federal Reserve Economic Data)**: Incorporates daily/weekly macroeconomic time series including federal funds rates, VIX volatility index, USD strength indices, and inflation expectations. These traditional market indicators serve as external features that often correlate with cryptocurrency volatility during market stress periods.
1. VIXCLS: CBOE Volatility Index, measures market's expectation of 30-day volatility for S&P 500
2. MOVE: Merrill Lynch Option Volatility Estimate, indicates expected volatility in the bond market.
3. OVXCLS: CBOE Crude Oil Volatility Index, reflects market's expectation of 30-day volatility for crude oil.
4. GVZCLS: CBOE Gold Volatility Index, indicates market's expectation of 30-day volatility for gold.
5. DTWEXBGS: Trade Weighted U.S. Dollar Index, measures the value of USD against a basket of foreign currencies.
6. DGS2: 2-Year Treasury Constant Maturity Rate, indicates short-term interest rates.
7. DGS10: 10-Year Treasury Constant Maturity Rate, indicates long-term interest rates.

- **Dune Analytics**: Offers daily on-chain metrics including transaction volumes, active addresses, network fees, and DeFi protocol activity. These blockchain-native indicators provide fundamental analysis components that traditional financial data cannot capture, essential for understanding crypto-specific volatility drivers. In the notebook, I use queries sourced from the following complementary dashboard: https://dune.com/amaliohidalgo/crypto-market-volatility-forecast-indicators-daily

### Feature Engineering Pipeline
The feature engineering component combines traditional technical analysis with automated time series feature extraction:

**TSFresh Integration**: Automatically generates over 700 time series features including statistical measures, trend indicators, and frequency domain characteristics. The system uses configurable complexity levels (minimal, efficient, comprehensive) to balance computational resources with feature richness.

**Technical Analysis Indicators**: Implements TA-Lib integration for standard technical indicators including moving averages, momentum oscillators, and volatility measures. These indicators provide domain-specific features that complement the automated TSFresh extraction.

**Distributed Processing**: Utilizes Dask for parallel feature computation across multiple CPU cores, enabling efficient processing of large time series datasets with rolling window operations.

### Machine Learning Pipeline
The modeling framework employs XGBoost with Bayesian optimization for hyperparameter tuning:

**Optuna Optimization**: Implements Tree-structured Parzen Estimator (TPE) for intelligent hyperparameter search across learning rate, tree depth, regularization parameters, and sampling ratios.

**Cross-Validation Strategy**: Uses time series-aware splitting to prevent data leakage while maintaining temporal structure in validation sets.

**Performance Metrics**: Evaluates models using multiple metrics including R², Mean Absolute Error (MAE), and Mean Absolute Scaled Error (MASE) to assess both absolute and relative forecasting performance.

## Installation

```bash
git clone https://github.com/yourusername/crypto-volatility-forecast.git
cd crypto-volatility-forecast

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Configuration

The system requires API credentials for data collection. Copy the environment template and add your API keys:

```bash
cp .env.example .env
```

Required API keys:
- `COINGECKO_API_KEY`: CoinGecko professional API access
- `DUNE_API_KEY`: Dune Analytics API for on-chain data
- `FRED_API_KEY`: Federal Reserve Economic Data API

Optional but recommended:
- `BINANCE_API_KEY` and `BINANCE_SECRET_KEY`: For enhanced rate limits on price data

## Usage

### Command Line Interface
```bash
# Basic execution with default parameters
crypto-forecast --target BTC --days 90

# Advanced configuration with custom optimization
crypto-forecast --target ETH --days 180 --trials 100 --quick-run
```

### Python Integration
```python
from src.config import load_config_from_file
from src.data.collectors import CryptoDataCollector
from src.features.engineering import CryptoFeatureEngineer
from src.models.pipeline import CryptoVolatilityMLPipeline

# Load configuration
config = load_config_from_file("config.json")

# Initialize components
collector = CryptoDataCollector(config.data)
engineer = CryptoFeatureEngineer(config.tsfresh)
pipeline = CryptoVolatilityMLPipeline(config.ml)

# Execute pipeline
data = collector.collect_all_data()
features = engineer.create_final_feature_set(data)
results = pipeline.run_complete_pipeline(features)
```

### Jupyter Notebooks
Two notebooks provide different entry points:

- `notebooks/main_pipeline.ipynb`: Complete pipeline execution with detailed analysis
- `notebooks/quick_start_example.ipynb`: Simplified workflow for rapid prototyping

## System Architecture

The toolkit uses a modern Python package structure with distributed computing capabilities:

```
src/
├── config.py              # Type-safe configuration management
├── pipeline.py             # Command-line interface and automation
├── data/
│   └── collectors.py       # Multi-source API integration
├── features/
│   └── engineering.py      # Automated feature extraction (TSFresh + TA-Lib)
├── models/
│   └── pipeline.py         # Machine learning with distributed optimization
└── utils/
    └── dask_helpers.py     # Distributed computing infrastructure
```

### Key Design Decisions

**Why Dask for distributed computing?** Large cryptocurrency datasets quickly exceed single-machine memory. Dask enables processing datasets that don't fit in RAM while maintaining the familiar pandas API.

**Why TSFresh for feature engineering?** Manual feature engineering introduces human bias. TSFresh automatically generates 700+ statistical features and includes significance testing to prevent overfitting.

**Why the src/ layout?** This structure prevents import issues during development and follows Python packaging best practices. It ensures the package installs correctly across different environments.

For detailed technical architecture and rationale behind each technology choice, see [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md).

## Component Rationale

### Why Dask for Distributed Computing
Traditional pandas operations become memory-intensive with large time series datasets. Dask provides:
- Out-of-core computation for datasets larger than available RAM
- Parallel processing across CPU cores for feature engineering operations
- Lazy evaluation enabling memory-efficient pipeline construction

### Why TSFresh for Feature Engineering
Manual feature engineering in time series analysis often misses complex temporal patterns. TSFresh addresses this by:
- Automatically generating hundreds of statistically relevant features
- Providing feature selection based on statistical significance testing
- Offering domain-agnostic feature extraction suitable for financial time series

### Why XGBoost with Optuna
XGBoost provides robust performance for structured data while Optuna enables efficient hyperparameter optimization:
- XGBoost handles mixed data types (continuous prices, categorical market conditions)
- Gradient boosting captures non-linear relationships in financial data
- Optuna's TPE algorithm converges faster than grid search or random search

### Why Multi-Source Data Integration
Single-source cryptocurrency analysis suffers from limited perspective. The multi-source approach provides:
- Market structure data (prices, volumes) from exchanges
- Sentiment and derivatives data (implied volatility) from options markets
- Macroeconomic context from traditional financial markets
- On-chain activity metrics from blockchain data

## Performance Considerations

The system implements several optimizations for production use:

**Memory Management**: Dask lazy evaluation and chunked processing prevent memory overflow with large datasets.

**Computational Efficiency**: Configurable TSFresh complexity levels allow trading feature richness for computation time.

**API Rate Limiting**: Built-in retry logic and rate limiting prevent API quota exhaustion.

**Caching**: Optional result caching reduces redundant API calls during development and testing.

## Package Distribution

This project is designed for multiple distribution scenarios:

### Development Installation
```bash
pip install -e .  # Editable install - changes reflected immediately
```

### Production Installation
```bash
pip install git+https://github.com/yourusername/crypto-volatility-forecast
```

### Optional Dependencies
The package supports flexible installation based on your needs:
```bash
pip install crypto-forecaster[distributed]  # Adds Dask support
pip install crypto-forecaster[technical]    # Adds TA-Lib indicators
pip install crypto-forecaster[all]          # Full installation
```

### Why This Packaging Approach?

**Flexible Dependencies**: Not everyone needs distributed computing or technical indicators. Optional extras keep the core package lightweight while allowing advanced features.

**Multiple Install Methods**: Supports both development (editable installs) and production (standard installs) workflows.

**Configuration Separation**: Settings live in JSON files, not code. Non-programmers can modify analysis parameters without touching Python.

## Validation and Testing

Run the setup verification script to ensure all components function correctly:

```bash
python test_setup.py
```

This script validates:
- Module imports and dependencies
- API connectivity (without consuming quota)
- Dask cluster initialization
- Basic pipeline functionality

## Contributing

Contributors should follow the established code structure and testing protocols. The CI/CD pipeline automatically validates code quality, runs tests, and builds distribution packages.

## About the Author

This toolkit was developed by **Amalio Hidalgo**, combining advanced quantitative finance techniques with modern machine learning engineering practices. The project demonstrates expertise in distributed computing, automated feature engineering, and production-ready software development.

## Contact

**Professional**: amalio.hidalgo-pickrell@hec.edu (HEC Paris, expires in 2026)  
**General Contact**: amaliohidalgo1@gmail.com (Permanent)

For commercial licensing, collaboration opportunities, or technical questions, please use the appropriate contact method above.

## License

Proprietary License - see [LICENSE](LICENSE) file for full terms.

## Disclaimer

This software is provided for educational and research purposes. Cryptocurrency markets involve substantial financial risk. Users should conduct their own research and risk assessment before making investment decisions.