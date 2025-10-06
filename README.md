# Cryptocurrency Market Forecasting Suite

A comprehensive demonstration of cryptocurrency market analysis and forecasting techniques, showcasing the evolution from basic price prediction to advanced volatility forecasting through multiple development iterations.

This repository has a complementary Dune dashboard- with SQL queries sourcing a broad collection of on-chain-analytics- located at: https://dune.com/amaliohidalgo/crypto-market-volatility-forecast-indicators-daily

## 🚀 Projects Overview

### 🎯 V2: Volatility Forecasting (Latest & Recommended)
**Location:** `v2-volatility-forecasting/`

Advanced cryptocurrency volatility prediction system demonstrating production-ready architecture and features:

- **🔗 Multi-Source Data Collection**: Binance, CoinGecko, Deribit DVOL, FRED Economics, Dune Analytics
- **🧠 Intelligent Feature Engineering**: TSFresh automated feature extraction + custom technical indicators
- **⚡ Optimized ML Pipeline**: XGBoost with Optuna hyperparameter tuning
- **🔄 Distributed Computing**: Dask for scalable processing
- **📊 Professional Implementation**: Configurable workflows, error handling, monitoring

**Key Features:**
- Target any cryptocurrency for volatility prediction
- Configurable data collection strategies (CSV → Cached → Execute)
- Credit-safe API usage with batch optimization
- Time-series cross-validation with proper data leakage prevention
- Comprehensive logging and error reporting

### 📈 V1: Price Action Forecasting
**Location:** `v1-price-forecasting/`

Original price prediction system focusing on directional price movements:

- **📊 Price-focused modeling**: OHLCV prediction and trend analysis
- **🔧 TSFresh integration**: Automated time series feature extraction
- **🎯 XGBoost implementation**: Gradient boosting for price prediction
- **📈 Technical analysis**: Custom indicators and market metrics

### 🔬 Development Workspace
**Location:** `development-workspace`

Research environment containing experimental work and development iterations:

- **🧪 Prototype notebooks**: Early development and testing
- **📝 Research notes**: Model experiments and findings
- **🔄 Iteration history**: Evolution from V1 to V2
- **⚙️ Work-in-progress**: Future features and enhancements

## 🎯 Which Version Should I Use?

| Use Case | Recommended Version | Why |
|----------|-------------------|-----|
| **Production volatility forecasting** | V2 | Complete feature set, robust error handling, scalable |
| **Learning/Educational** | V2 | Well-documented, configurable, safer API usage |
| **Price prediction research** | V1 | Simpler, focused on price movements |
| **Development/Contribution** | V2 + Dev Workspace | Latest codebase + research context |

## 🚀 Explore the Code (V2 - Latest Implementation)

```bash
# Navigate to V2 project
cd v2-volatility-forecasting

# Install dependencies for exploration
pip install -e .

# Set up environment (for code exploration)
cp .env.example .env
# Note: Add your API keys only if you plan to run the examples

# Explore the implementation
cd notebooks
jupyter lab main_pipeline.ipynb
```

**Note**: This is primarily a code demonstration. See [License](#license) for usage terms.

## 📊 Architecture Comparison

| Feature | V1 | V2 |
|---------|----|----|
| **Target** | Price Action | Volatility |
| **Data Sources** | Limited | 5+ APIs |
| **Feature Engineering** | Basic TSFresh | Advanced + Custom |
| **ML Framework** | XGBoost | XGBoost + Optuna |
| **Scalability** | Single-threaded | Dask Distributed |
| **Error Handling** | Basic | Enterprise-grade |
| **Configuration** | Hardcoded | Fully configurable |
| **API Safety** | Manual | Credit-safe automation |

## 🛣️ Development Evolution

This suite represents the evolution of cryptocurrency forecasting approaches:

1. **V1 (2024)**: Initial price prediction system with basic ML pipeline
2. **Dev Phase**: Extensive experimentation and research (CVA folder)
3. **V2 (2025)**: Complete rewrite focusing on volatility, scalability, and production readiness

## 📚 Documentation

- **[V2 Technical Architecture](v2-volatility-forecasting/TECHNICAL_ARCHITECTURE.md)**: Detailed system design
- **[Installation Guide](v2-volatility-forecasting/README.md)**: Setup instructions
- **[API Configuration](v2-volatility-forecasting/.env.example)**: Required API keys
- **[Examples](v2-volatility-forecasting/notebooks/)**: Jupyter notebooks with full workflows

## 🤝 Code Review and Discussion

This repository is designed for technical demonstration and review:

1. **Architecture Review**: Examine the V2 implementation for system design patterns
2. **Code Quality**: Review software engineering practices and patterns
3. **Technical Discussion**: Open issues for questions about implementation approaches
4. **Documentation**: Improvements to educational value welcome

See [CODE_REVIEW_GUIDE.md](CODE_REVIEW_GUIDE.md) for detailed exploration guidance.

## 📈 Performance Benchmarks

| Metric | V1 | V2 |
|--------|----|----|
| **Data Collection Speed** | ~10 min | ~2 min (parallelized) |
| **Feature Engineering** | ~15 min | ~5 min (optimized) |
| **Model Training** | ~20 min | ~8 min (Optuna + early stopping) |
| **Memory Usage** | ~4GB | ~2GB (Dask chunking) |
| **API Credit Usage** | High risk | Optimized + safe |

## 🔄 Migration Guide

### From V1 to V2:
- **Data compatibility**: V2 can process V1 datasets
- **Configuration**: Migrate from hardcoded to config files
- **API keys**: Same keys work, better safety controls
- **Features**: All V1 features available + many more

## 📜 License

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

**Educational and Professional Use:**
- ✅ Code review and technical assessment
- ✅ Learning and skill development
- ✅ Academic and research reference
- ✅ Portfolio demonstration

**Commercial use is prohibited.** See [LICENSE](LICENSE) for full terms.

## 🌟 Acknowledgments

This project demonstrates the integration of several excellent open-source technologies:

- **TSFresh Team**: Automated feature extraction framework
- **XGBoost Developers**: Gradient boosting machine learning library
- **Dune Analytics**: On-chain data API platform
- **Dask Community**: Distributed computing framework

## 🎯 About This Repository

This repository serves as a technical demonstration and educational resource, showcasing the evolution of cryptocurrency market analysis techniques from basic price prediction to advanced volatility forecasting.
