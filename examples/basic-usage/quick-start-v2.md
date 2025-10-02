# Quick Start: V2 Volatility Forecasting

A 5-minute example to get you started with cryptocurrency volatility prediction.

## Setup

```bash
# Install V2 system
cd ../v2-volatility-forecasting
pip install -e .

# Configure API keys (copy the example and edit)
cp .env.example .env
```

## Minimal Working Example

```python
import sys
sys.path.append('../v2-volatility-forecasting/src')

from data.collectors import CryptoDataCollector

# Initialize with safe defaults
collector = CryptoDataCollector(
    target_coin="ethereum",
    lookback_days=30,
    frequency="1D",
    dune_strategy="csv_only",  # Safe: no API calls
    allow_dune_execution=False  # Safety flag
)

# Collect data (this will use cached/CSV data only)
data = collector.collect_all_data()

# Show results
for source, df in data.items():
    if not df.empty:
        print(f"{source}: {df.shape[0]} records")
    else:
        print(f"{source}: No data")
```

## Expected Output
```
📊 Collecting cryptocurrency universe...
✅ Binance: 8 coins collected
✅ CoinGecko: 5 coins collected  
✅ DVOL: 30 records
🏛️ Collecting on-chain analytics (strategy: csv_only)...
⚠️ Dune: No data (CSV file not found)
✅ FRED: 30 records

binance_price: 30 records
coingecko_price: 30 records
dvol: 30 records
onchain: No data
macro: 30 records
```

## Next Steps

1. **Add More API Keys**: Configure .env for live data collection
2. **Try Different Coins**: Change target_coin parameter
3. **Enable Dune Data**: Set allow_dune_execution=True (uses credits)
4. **Advanced Examples**: Check other notebooks in this folder

## Common Issues

**No data returned?**
- Check your API keys in .env file
- Verify internet connection
- Try with dune_strategy="cached_only"

**API errors?**
- Make sure DUNE_API_KEY_2 is set correctly
- Check API rate limits
- Use csv_only strategy for testing

---

