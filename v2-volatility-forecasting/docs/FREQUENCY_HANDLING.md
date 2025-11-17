# Frequency Handling - Complete Guide

## 🎯 Overview
The cryptocurrency volatility forecasting pipeline now supports both **daily (1D)** and **hourly (1H)** data frequencies with proper normalization and API-specific conversions.

## ✅ Supported Frequencies

### Standardized Formats
- **Daily**: `"1D"` (normalized format)
- **Hourly**: `"1H"` (normalized format)

### Accepted Input Formats

**Hourly:**
- `"1H"`, `"1h"`, `"H"`, `"h"`
- `"hourly"`, `"hour"`, `"1hour"`

**Daily:**
- `"1D"`, `"1d"`, `"D"`, `"d"`
- `"daily"`, `"day"`, `"1day"`

All inputs are automatically normalized to `"1H"` or `"1D"`.

## 🔧 Implementation

### Frequency Normalization

```python
@staticmethod
def _normalize_frequency(freq: str) -> str:
    """
    Normalize any frequency input to standard format.
    Returns: "1H" or "1D"
    """
    freq_lower = freq.lower().strip()
    
    if freq_lower in ["1h", "h", "hourly", "hour", "1hour"]:
        return "1H"
    elif freq_lower in ["1d", "d", "daily", "day", "1day"]:
        return "1D"
    else:
        print(f"⚠️ Unknown frequency '{freq}', defaulting to daily (1D)")
        return "1D"
```

### API-Specific Conversions

Each data source requires different frequency formats:

| Method | Daily | Hourly | Notes |
|--------|-------|--------|-------|
| `get_pandas_freq()` | `"D"` | `"H"` | For pandas `.resample()` |
| `get_binance_interval()` | `"1d"` | `"1h"` | Binance API format |
| `get_deribit_resolution()` | `"1D"` | `"60"` | Deribit uses minutes (60=1hr) |
| `get_coingecko_interval()` | `"daily"` | `"hourly"` | CoinGecko API format |

### Helper Methods

```python
def supports_hourly(self) -> bool:
    """Check if hourly frequency is selected."""
    return self.frequency == "1H"

def get_frequency_description(self) -> str:
    """Get human-readable description."""
    return "Hourly (1H)" if self.frequency == "1H" else "Daily (1D)"
```

## 📊 Data Source Limitations

### CoinGecko
- ✅ **Daily**: Full historical data
- ⚠️ **Hourly**: Limited to last 90 days only
- **Action**: Automatically handles via API parameters

### Binance  
- ✅ **Daily**: Full historical data
- ✅ **Hourly**: Full historical data
- **Note**: Uses pagination for >1000 candles

### Deribit (DVOL)
- ✅ **Daily**: Full historical data
- ✅ **Hourly**: Full historical data
- **Format**: Uses resolution=60 for hourly

### FRED (Macroeconomic)
- ✅ **Daily**: Native daily data
- ⚠️ **Hourly**: Daily data forward-filled to hourly
- **Warning**: FRED only provides daily observations

### Dune Analytics
- ✅ **Daily**: Most queries provide daily data
- ⚠️ **Hourly**: Limited query support
- **Action**: Resampled where possible

## 🚀 Usage Examples

### Example 1: Daily Frequency (Default)
```python
from data.collectors import CryptoDataCollector

# Using string formats - all equivalent
collector = CryptoDataCollector(frequency="1D")
collector = CryptoDataCollector(frequency="daily")
collector = CryptoDataCollector(frequency="d")

print(collector.get_frequency_description())
# Output: "Daily (1D)"

print(collector.get_binance_interval())
# Output: "1d"
```

### Example 2: Hourly Frequency
```python
# Using string formats - all equivalent
collector = CryptoDataCollector(frequency="1H")
collector = CryptoDataCollector(frequency="hourly")
collector = CryptoDataCollector(frequency="h")

print(collector.get_frequency_description())
# Output: "Hourly (1H)"

# Warnings displayed at initialization:
# ⚠️ Hourly frequency selected - Note:
#    • CoinGecko: Limited to last 90 days for hourly data
#    • FRED: Only daily data available (will be resampled)
#    • Dune: Mostly daily data (hourly limited)
```

### Example 3: Configuration-Based
```python
from config import create_default_config, DataConfig

# Override frequency in config
config = create_default_config()
config.data.frequency = "hourly"

collector = CryptoDataCollector(config=config)
```

### Example 4: Check Frequency in Code
```python
collector = CryptoDataCollector(frequency="1H")

if collector.supports_hourly():
    print("Running with hourly data")
    # Adjust lookback days for hourly (more data points)
    lookback = 30  # 30 days of hourly = 720 data points
else:
    print("Running with daily data")
    lookback = 365  # 365 days of daily data
```

## ⚙️ Automatic Resampling

The pipeline automatically handles frequency conversion:

### FRED Data (Daily Only)
```python
# When hourly requested:
fred_data = collector.fred_get_series()
# ℹ️  FRED provides daily data only - will forward-fill for hourly frequency
# Result: Daily values forward-filled to create hourly timestamps
```

### DVOL Data
```python
# Automatic resampling based on frequency:
dvol_data = collector.deribit_get_dvol()
# Uses .resample(collector.get_pandas_freq()).last()
```

## 🎯 Best Practices

### For Backtesting (Daily Recommended)
```python
collector = CryptoDataCollector(
    frequency="1D",
    lookback_days=730  # 2 years of daily data
)
```

### For Real-Time Trading (Hourly)
```python
collector = CryptoDataCollector(
    frequency="1H",
    lookback_days=30  # 30 days = 720 hourly candles
)
```

### For Research (Mixed Approach)
```python
# Collect daily for long-term analysis
daily_collector = CryptoDataCollector(frequency="1D", lookback_days=1095)  # 3 years
long_term_data = daily_collector.collect_all_data()

# Collect hourly for recent detailed analysis  
hourly_collector = CryptoDataCollector(frequency="1H", lookback_days=90)
recent_data = hourly_collector.collect_all_data()
```

## 🔍 Debugging Frequency Issues

### Check Current Frequency
```python
collector = CryptoDataCollector(frequency="hourly")

print(f"Normalized frequency: {collector.frequency}")
# Output: "1H"

print(f"Pandas format: {collector.get_pandas_freq()}")
# Output: "H"

print(f"Binance format: {collector.get_binance_interval()}")
# Output: "1h"

print(f"Is hourly: {collector.supports_hourly()}")
# Output: True
```

### Verify Data Frequency
```python
# After collecting data
binance_data = collector.binance_get_price_action(["bitcoin"], ["BTC"])

# Check actual frequency
freq = pd.infer_freq(binance_data.index)
print(f"Detected frequency: {freq}")
# Hourly: "H" or "60T" (60 minutes)
# Daily: "D"
```

## 📝 Configuration File Support

### config.json
```json
{
    "data": {
        "frequency": "1H",
        "lookback_days": 30
    }
}
```

### Environment Variable
```bash
# .env file
DATA_FREQUENCY=hourly
LOOKBACK_DAYS=30
```

## ⚠️ Important Notes

1. **CoinGecko Hourly Limitation**: Only last 90 days available for hourly
2. **FRED Resampling**: Daily data is forward-filled when hourly is requested
3. **Dune Limitations**: Most queries provide daily data only
4. **Data Alignment**: All data sources aligned to common frequency before integration
5. **Memory Usage**: Hourly data uses ~24x more memory than daily for same time period

## 🎉 Result

The frequency handling system now:
- ✅ **Accepts multiple input formats** and normalizes them
- ✅ **Converts correctly** for each API
- ✅ **Warns users** about limitations
- ✅ **Handles resampling** automatically
- ✅ **Supports both daily and hourly** workflows
- ✅ **Professional implementation** following best practices