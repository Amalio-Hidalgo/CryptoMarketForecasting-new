# V1: Price Action Forecasting

**⚠️ This is the legacy version. For new projects, use [V2: Volatility Forecasting](../v2-volatility-forecasting/)**

## Overview

Original cryptocurrency price prediction system focusing on directional price movements.

### Key Features
- **Price Direction Prediction**: Up/down movement forecasting
- **TSFresh Integration**: Automated time series feature extraction  
- **XGBoost Implementation**: Gradient boosting for price prediction
- **Technical Analysis**: Custom indicators and market metrics

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run example
python examples/basic_price_prediction.py
```

### Architecture
```
Data Collection → Feature Engineering → XGBoost → Price Predictions
   (CoinGecko)      (TSFresh Basic)    (Default)   (Direction/OHLC)
```

### Limitations
- Manual data collection process
- Limited to single cryptocurrency
- High variance in performance
- No proper time series cross-validation
- Basic error handling

### Migration to V2
If you're using V1, consider migrating to V2 for:
- ✅ Better prediction accuracy (volatility vs price)
- ✅ Automated data collection from multiple sources
- ✅ Production-ready architecture
- ✅ Cost-efficient API usage
- ✅ Comprehensive error handling

See [Migration Guide](../docs/comparison.md) for detailed comparison.

### Status
- **Maintenance**: Bug fixes only
- **New Features**: None planned
- **Recommended**: Use V2 for new projects

---

**🚀 Ready to upgrade?** Check out [V2: Volatility Forecasting](../v2-volatility-forecasting/) for the latest and greatest!