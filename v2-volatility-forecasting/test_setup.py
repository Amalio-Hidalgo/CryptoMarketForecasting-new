#!/usr/bin/env python3
"""
Test script to verify the cryptocurrency volatility forecasting toolkit setup
"""

import sys
import os
import traceback
from pathlib import Path

def test_imports():
    """Test all module imports"""
    print("Testing module imports...")
    
    try:
        # Test basic imports
        import pandas as pd
        import numpy as np
        print("  ✅ pandas, numpy")
        
        # Test our modules
        from src.config import load_config_from_file, Config
        print("  ✅ config module")
        
        from src.data.collectors import CryptoDataCollector
        print("  ✅ data collectors")
        
        from src.features.engineering import CryptoFeatureEngineer
        print("  ✅ feature engineering")
        
        from src.models.pipeline import CryptoVolatilityMLPipeline
        print("  ✅ ML pipeline")
        
        from src.utils.dask_helpers import create_optimized_dask_client
        print("  ✅ utilities")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("\\nTesting configuration...")
    
    try:
        from src.config import load_config_from_file
        
        if os.path.exists("config.json"):
            config = load_config_from_file("config.json")
            print(f"  ✅ Config loaded: target={config.data.target_coin}")
            return True
        else:
            print("  ⚠️ config.json not found")
            return False
            
    except Exception as e:
        print(f"  ❌ Config error: {e}")
        return False

def test_data_collector():
    """Test data collector initialization"""
    print("\\nTesting data collector...")
    
    try:
        from src.data.collectors import CryptoDataCollector
        
        collector = CryptoDataCollector(
            timezone="UTC",
            top_n=5,  # Small number for testing
            lookback_days=7  # Short period for testing
        )
        print("  ✅ CryptoDataCollector initialized")
        return True
        
    except Exception as e:
        print(f"  ❌ Data collector error: {e}")
        return False

def test_feature_engineer():
    """Test feature engineer initialization"""
    print("\\nTesting feature engineer...")
    
    try:
        from src.features.engineering import CryptoFeatureEngineer
        
        engineer = CryptoFeatureEngineer(
            extraction_settings="minimal",
            fdr_level=0.05,
            time_window=5,
            random_seed=42
        )
        print("  ✅ CryptoFeatureEngineer initialized")
        return True
        
    except Exception as e:
        print(f"  ❌ Feature engineer error: {e}")
        return False

def test_ml_pipeline():
    """Test ML pipeline initialization"""
    print("\\nTesting ML pipeline...")
    
    try:
        from src.models.pipeline import CryptoVolatilityMLPipeline
        
        pipeline = CryptoVolatilityMLPipeline(
            n_trials=2,  # Small number for testing
            n_rounds=10,  # Small number for testing
            eval_metric="mae",
            early_stopping_rounds=5,
            splits=2,
            random_seed=42
        )
        print("  ✅ CryptoVolatilityMLPipeline initialized")
        return True
        
    except Exception as e:
        print(f"  ❌ ML pipeline error: {e}")
        return False

def test_dask():
    """Test Dask functionality"""
    print("\\nTesting Dask setup...")
    
    try:
        from src.utils.dask_helpers import create_optimized_dask_client, cleanup_dask_client
        
        client = create_optimized_dask_client(
            n_workers=1,
            threads_per_worker=1,
            memory_limit="1GB",
            processes=False
        )
        
        if client:
            print("  ✅ Dask client created")
            cleanup_dask_client(client)
            print("  ✅ Dask client cleaned up")
            return True
        else:
            print("  ❌ Could not create Dask client")
            return False
            
    except Exception as e:
        print(f"  ❌ Dask error: {e}")
        return False

def test_optional_dependencies():
    """Test optional dependencies"""
    print("\\nTesting optional dependencies...")
    
    optional_deps = {
        "tsfresh": "TSFresh time series features",
        "talib": "Technical Analysis Library", 
        "xgboost": "XGBoost machine learning",
        "optuna": "Hyperparameter optimization",
        "dask": "Distributed computing"
    }
    
    available = []
    missing = []
    
    for dep, description in optional_deps.items():
        try:
            __import__(dep)
            available.append(f"  ✅ {dep}: {description}")
        except ImportError:
            missing.append(f"  ❌ {dep}: {description}")
    
    for dep in available:
        print(dep)
    
    if missing:
        print("\\n  Missing optional dependencies:")
        for dep in missing:
            print(dep)
        return False
    
    return True

def main():
    """Run all tests"""
    print("Cryptocurrency Volatility Forecasting Toolkit - Test Suite")
    print("=" * 70)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_config),
        ("Data Collector", test_data_collector),
        ("Feature Engineer", test_feature_engineer),
        ("ML Pipeline", test_ml_pipeline),
        ("Dask Setup", test_dask),
        ("Optional Dependencies", test_optional_dependencies),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status:10} {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\\nAll tests passed. Setup is ready for use.")
        return 0
    else:
        print(f"\\n{failed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())