#!/usr/bin/env python3
"""
Simple test to verify the code structure and imports work.
"""

import sys
import os

# Add the v2 source path
sys.path.append('v2-volatility-forecasting/src')

def test_imports():
    """Test that we can import our modules."""
    try:
        # Test config import
        from config import DataConfig, APIConfig
        print("✅ Config imports successful")
        
        # Test that config has the right structure
        data_config = DataConfig()
        api_config = APIConfig()
        
        print(f"   - Default Dune strategy: {data_config.dune_strategy}")
        print(f"   - Allow Dune execution: {data_config.allow_dune_execution}")
        print(f"   - Dune API key env var: DUNE_API_KEY_2")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_file_structure():
    """Test that key files exist."""
    required_files = [
        'v2-volatility-forecasting/src/config.py',
        'v2-volatility-forecasting/src/data/collectors.py',
        'v2-volatility-forecasting/src/data/__init__.py',
        'docs/technical-fixes.md',
        'README.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ All required files present")
        return True

def test_collectors_structure():
    """Test that collectors.py has the right structure."""
    try:
        collectors_path = 'v2-volatility-forecasting/src/data/collectors.py'
        with open(collectors_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for key safety features
        safety_checks = [
            'allow_dune_execution: bool = False',  # Default safety
            'collect_crypto_data_with_cached_dune',  # Safe function
            'collect_crypto_data_with_fresh_dune',   # Explicit credit function
            'DUNE_API_KEY_2',  # Correct API key
            'cached_only'      # Safe strategy
        ]
        
        missing_features = []
        for check in safety_checks:
            if check not in content:
                missing_features.append(check)
        
        if missing_features:
            print(f"❌ Missing safety features: {', '.join(missing_features)}")
            return False
        else:
            print("✅ All safety features present in collectors.py")
            return True
            
    except Exception as e:
        print(f"❌ Error checking collectors.py: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Running basic structure and safety tests...\n")
    
    # Run tests
    test1 = test_file_structure()
    test2 = test_imports()
    test3 = test_collectors_structure()
    
    # Summary
    print(f"\n📊 Test Results:")
    print(f"   - File Structure: {'✅ PASSED' if test1 else '❌ FAILED'}")
    print(f"   - Config Imports: {'✅ PASSED' if test2 else '❌ FAILED'}")
    print(f"   - Safety Features: {'✅ PASSED' if test3 else '❌ FAILED'}")
    
    if test1 and test2 and test3:
        print(f"\n🎉 All basic tests passed! Repository structure is ready.")
        print(f"\nNext steps:")
        print(f"   1. Install dependencies: pip install -r v2-volatility-forecasting/requirements.txt")
        print(f"   2. Set up environment variables (DUNE_API_KEY_2, etc.)")
        print(f"   3. Test data collection with cached data")
        print(f"   4. Initialize git when ready to commit")
    else:
        print(f"\n⚠️  Some basic tests failed. Please check the output above.")