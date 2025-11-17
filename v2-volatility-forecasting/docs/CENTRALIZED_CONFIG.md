# Centralized Configuration Management

## Problem Solved
Previously, key constants and configuration values were scattered across multiple files:
- `notebooks/main_pipeline.ipynb` - Pipeline constants (TARGET_COIN, TOP_N, etc.)
- `src/config.py` - API configuration (FRED series, data settings)
- `src/data/collectors.py` - Duplicate FRED series definitions

This created maintenance issues, inconsistencies, and potential bugs.

## Solution: Single Source of Truth

### 1. Configuration Hierarchy
```
src/config.py (MASTER)
├── DataConfig: Data collection parameters
├── TSFreshConfig: Feature engineering settings  
├── MLConfig: Machine learning parameters
├── DaskConfig: Distributed computing setup
└── APIConfig: API keys and endpoints (including FRED series)
```

### 2. Import Pattern

**Before (Scattered Constants):**
```python
# In notebook
TARGET_COIN = "ethereum"
TOP_N = 5
LOOKBACK_DAYS = 365

# In collectors.py  
self.FRED_KNOWN = {
    "VIXCLS": "vix_equity_vol",
    "DGS10": "us_10y_treasury_yield",
    # ... duplicated from config.py
}
```

**After (Centralized):**
```python
# In notebook
from config import create_default_config
config = create_default_config()
TARGET_COIN = config.data.target_coin

# In collectors.py
from ..config import APIConfig
api_config = APIConfig()
self.FRED_KNOWN = api_config.fred_series
```

### 3. Benefits

✅ **Single Source of Truth**: All constants defined once in `config.py`
✅ **No Duplication**: Eliminates scattered constant definitions
✅ **Type Safety**: Dataclass-based configuration with type hints
✅ **Easy Maintenance**: Change once, updates everywhere
✅ **Environment Support**: Centralized environment variable handling
✅ **Version Control**: Clear configuration changes in git history

### 4. Files Modified

#### `src/data/collectors.py`
- **REMOVED**: `self.FRED_KNOWN` dictionary definition
- **ADDED**: Import from centralized `APIConfig`

#### `notebooks/main_pipeline.ipynb`
- **MODIFIED**: Configuration cell to import from centralized config
- **KEPT**: Backward compatibility variables for existing code

#### `src/config.py`
- **ENHANCED**: Comprehensive configuration structure
- **INCLUDES**: All FRED series, API settings, pipeline parameters

### 5. Usage Examples

#### For Notebooks:
```python
from config import create_default_config
config = create_default_config()

# Use configuration values
collector = CryptoDataCollector(
    target_coin=config.data.target_coin,
    top_n=config.data.top_n,
    lookback_days=config.data.lookback_days
)
```

#### For Production Code:
```python
from config import Config, DataConfig, MLConfig

# Custom configuration
custom_config = Config(
    data=DataConfig(target_coin="bitcoin", lookback_days=730),
    ml=MLConfig(n_trials=100)
)
```

#### For API Integration:
```python
from config import APIConfig
api_config = APIConfig()
fred_series = api_config.fred_series  # Centralized FRED definitions
```

### 6. Configuration Files

All configuration is centralized in:
- **`src/config.py`** - Main configuration classes
- **`config.json`** - Optional JSON configuration file
- **`.env`** - Environment variables (API keys)

### 7. Migration Checklist

- [x] Remove FRED series duplication from `collectors.py`
- [x] Update notebook to use centralized config
- [x] Verify all constants import correctly
- [ ] Update other notebooks in `development-workspace/` (optional)
- [ ] Add configuration validation
- [ ] Create configuration tests

### 8. Maintenance Notes

When adding new constants:
1. Add to appropriate dataclass in `src/config.py`
2. Import in files that need it
3. Update this documentation

When removing constants:
1. Remove from `src/config.py`
2. Update all importing files
3. Test thoroughly

## MOVE Index Removal

As part of this centralization, we also removed references to the MOVE index time series code since it's not available on FRED. This eliminates potential API errors and keeps the configuration clean.

**Removed from:**
- `src/config.py` - FRED series dictionary
- `src/data/collectors.py` - FRED_KNOWN dictionary (now imports from config)
- Associated comments and documentation

This centralized approach ensures maintainable, consistent, and error-free configuration management across the entire cryptocurrency volatility forecasting pipeline.