"""
Feature Engineering Module for Cryptocurrency Volatility Forecasting

This module contains all feature engineering functions from LatestNotebook.ipynb:
- Technical Analysis indicators using TA-Lib
- Dask-optimized TSFresh feature extraction
- Feature selection and rolling time series operations
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# TSFresh imports
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.feature_extraction import (
    ComprehensiveFCParameters, 
    EfficientFCParameters, 
    MinimalFCParameters
)

# Technical Analysis
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available. Technical indicators will be skipped.")


class CryptoFeatureEngineer:
    """Main class for cryptocurrency feature engineering."""
    
    def __init__(self, 
                 extraction_settings: str = "efficient",
                 fdr_level: float = 0.05,
                 time_window: int = 14,
                 random_seed: int = 42):
        """
        Initialize the feature engineer.
        
        Args:
            extraction_settings: TSFresh complexity level ('minimal', 'efficient', 'comprehensive')
            fdr_level: False discovery rate for feature selection
            time_window: Rolling window size for time series features
            random_seed: Random seed for reproducibility
        """
        self.fdr_level = fdr_level
        self.time_window = time_window
        self.random_seed = random_seed
        
        # Set TSFresh parameters
        if extraction_settings == "minimal":
            self.extraction_settings = MinimalFCParameters()
        elif extraction_settings == "efficient":
            self.extraction_settings = EfficientFCParameters()
        elif extraction_settings == "comprehensive":
            self.extraction_settings = ComprehensiveFCParameters()
        else:
            self.extraction_settings = EfficientFCParameters()

    def compute_ta_indicators(self, 
                             df: pd.DataFrame, 
                             price_prefix: str = "prices_",
                             rsi_period: int = 14,
                             macd_fast: int = 12, 
                             macd_slow: int = 26, 
                             macd_signal: int = 9,
                             sma_windows: Tuple[int, ...] = (10, 20, 50),
                             ema_windows: Tuple[int, ...] = (10, 20, 50)) -> pd.DataFrame:
        """
        Compute technical analysis indicators using TA-Lib.
        
        Args:
            df: DataFrame with OHLCV data
            price_prefix: Prefix for price columns
            rsi_period: RSI calculation period
            macd_fast: MACD fast period
            macd_slow: MACD slow period  
            macd_signal: MACD signal period
            sma_windows: Simple moving average windows
            ema_windows: Exponential moving average windows
            
        Returns:
            DataFrame with technical analysis indicators
        """
        if not TALIB_AVAILABLE:
            print("TA-Lib not available. Returning empty DataFrame.")
            return pd.DataFrame(index=df.index)
            
        out = pd.DataFrame(index=df.index)
        price_cols = [c for c in df.columns if c.startswith(price_prefix)]
        coins = [c[len(price_prefix):] for c in price_cols]
        
        for coin in coins:
            try:
                p = df[f"{price_prefix}{coin}"]
                
                # Ensure we have required OHLCV columns
                high_col = f"high_{coin}"
                low_col = f"low_{coin}"
                volume_col = f"volume_{coin}"
                
                # RSI
                out[f"rsi{rsi_period}_{coin}"] = talib.RSI(p.values, timeperiod=rsi_period)
                
                # MACD
                macd, macd_sig, macd_hist = talib.MACD(p.values, 
                                                       fastperiod=macd_fast, 
                                                       slowperiod=macd_slow, 
                                                       signalperiod=macd_signal)
                out[f"macd_{coin}"] = macd
                out[f"macd_signal_{coin}"] = macd_sig
                out[f"macd_hist_{coin}"] = macd_hist
                
                # Moving Averages
                for w in sma_windows:
                    out[f"sma{w}_{coin}"] = talib.SMA(p.values, timeperiod=w)
                for w in ema_windows:
                    out[f"ema{w}_{coin}"] = talib.EMA(p.values, timeperiod=w)
                
                # Bollinger Bands
                out[f"bb_upper_{coin}"], out[f"bb_middle_{coin}"], out[f"bb_lower_{coin}"] = talib.BBANDS(p.values)
                
                # Additional indicators (if OHLCV data available)
                if high_col in df.columns and low_col in df.columns:
                    high_vals = df[high_col].values
                    low_vals = df[low_col].values
                    
                    out[f"atr_{coin}"] = talib.ATR(high_vals, low_vals, p.values)
                    out[f"adx_{coin}"] = talib.ADX(high_vals, low_vals, p.values)
                    out[f"stoch_k_{coin}"], out[f"stoch_d_{coin}"] = talib.STOCH(high_vals, low_vals, p.values)
                    out[f"cci_{coin}"] = talib.CCI(high_vals, low_vals, p.values)
                    out[f"willr_{coin}"] = talib.WILLR(high_vals, low_vals, p.values)
                    
                    # Volume indicators
                    if volume_col in df.columns:
                        volume_vals = df[volume_col].values
                        out[f"obv_{coin}"] = talib.OBV(p.values, volume_vals)
                        out[f"mfi_{coin}"] = talib.MFI(high_vals, low_vals, p.values, volume_vals)
                
                # Momentum indicators
                out[f"mom_{coin}"] = talib.MOM(p.values)
                out[f"roc_{coin}"] = talib.ROC(p.values)
                
            except Exception as e:
                print(f"Error computing TA indicators for {coin}: {e}")
                continue
                
        out.index = df.index
        return out

    def roll_dask_partition(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single Dask partition for time series rolling.
        
        Args:
            df: DataFrame partition
            
        Returns:
            Rolled time series DataFrame
        """
        if len(df) == 0:
            return pd.DataFrame()
            
        print(f"Processing partition with columns: {df.columns.tolist()}")
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        try:
            rolled = roll_time_series(
                df,
                column_id='variable',
                column_sort='date',
                max_timeshift=self.time_window,
                min_timeshift=1,
                rolling_direction=1,
                n_jobs=1
            )
            return rolled
        except Exception as e:
            print(f"Error in rolling: {e}")
            return pd.DataFrame()

    def extract_dask_partition(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract TSFresh features from a single Dask partition.
        
        Args:
            df: Rolled time series DataFrame partition
            
        Returns:
            Extracted features DataFrame
        """
        df = df.copy().dropna()
        if len(df) == 0:
            return pd.DataFrame()
            
        print(f"Extracting features for partition with columns: {df.columns.tolist()}")
        
        try:
            features = extract_features(
                df,
                column_id='id',
                column_sort='date',
                column_kind='variable',
                column_value='value',
                default_fc_parameters=self.extraction_settings,
                n_jobs=1
            )
            return features
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            return pd.DataFrame()

    def select_dask_partition(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Select relevant features from a single Dask partition.
        
        Args:
            df: Features DataFrame
            y: Target series
            
        Returns:
            Selected features DataFrame
        """
        df = df.reset_index(level=0, drop=True).join(y, how='inner').dropna()
        if len(df) == 0:
            return pd.DataFrame()
            
        try:
            features = select_features(
                df.drop('target', axis=1),
                df['target'],
                ml_task='regression',
                fdr_level=self.fdr_level,
                hypotheses_independent=False,
                n_jobs=1
            )
            return features
        except Exception as e:
            print(f"Error in feature selection: {e}")
            return pd.DataFrame()

    def prepare_target_variable(self, 
                               df: pd.DataFrame, 
                               target_coin: str = "ethereum",
                               method: str = "realized_volatility") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix and target variable.
        
        Args:
            df: Input DataFrame with price data
            target_coin: Target cryptocurrency for prediction
            method: Target calculation method
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        print(f"📊 Preparing target variable for {target_coin}")
        
        # Create features matrix
        X = df.iloc[-365:].dropna(axis=1, thresh=int(0.1 * len(df))).ffill(limit=3)
        
        # Calculate log returns
        price_col = f'prices_{target_coin}'
        if price_col not in X.columns:
            raise ValueError(f"Price column {price_col} not found in data")
            
        X[f'log_returns_{target_coin}'] = np.log(X[price_col]) - np.log(X[price_col].shift(1))
        
        # Calculate realized volatility
        if method == "realized_volatility":
            X[f'realized_vol_{target_coin}'] = abs(X[f'log_returns_{target_coin}'])
        else:
            raise ValueError(f"Unknown target method: {method}")
        
        # Difference the data
        X = X.diff().dropna()
        
        # Create target (next period volatility)
        y = X[f'realized_vol_{target_coin}'].shift(-1).dropna().rename("target")
        
        # Align indices
        X.rename_axis(index='date', inplace=True)
        y.rename_axis(index='date', inplace=True)
        
        # Ensure alignment
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        print(f"✅ Features shape: {X.shape}, Target shape: {y.shape}")
        
        return X, y

    def create_dask_feature_container(self, 
                                     X: pd.DataFrame, 
                                     n_partitions: Optional[int] = None) -> dd.DataFrame:
        """
        Create Dask DataFrame from feature matrix for distributed processing.
        
        Args:
            X: Feature matrix
            n_partitions: Number of Dask partitions (defaults to number of variables)
            
        Returns:
            Dask DataFrame ready for TSFresh processing
        """
        print("🔄 Creating Dask feature container...")
        
        # Melt DataFrame to long format for TSFresh
        FC = X.reset_index().melt(id_vars=['date']).sort_values(by='variable')
        
        if n_partitions is None:
            n_partitions = FC.variable.nunique()
            
        FC_dask = dd.from_pandas(FC, npartitions=n_partitions)
        
        # Verify partitioning
        unique_vars_per_partition = FC_dask.map_partitions(
            lambda df: df['variable'].nunique()
        ).compute().unique()
        
        print(f"✅ Created Dask container with {n_partitions} partitions")
        print(f"Variables per partition: {unique_vars_per_partition}")
        
        return FC_dask

    def run_tsfresh_pipeline(self, 
                            X: pd.DataFrame, 
                            y: pd.Series,
                            client,
                            n_partitions: Optional[int] = None) -> pd.DataFrame:
        """
        Run complete TSFresh feature engineering pipeline with Dask.
        
        Args:
            X: Feature matrix
            y: Target series
            client: Dask client
            n_partitions: Number of partitions
            
        Returns:
            Processed features DataFrame
        """
        print("Starting TSFresh + Dask pipeline...")
        
        # Create Dask feature container
        FC_dask = self.create_dask_feature_container(X, n_partitions)
        
        # Test rolling on one partition for metadata
        print("Testing rolling operation...")
        df_test = FC_dask.partitions[0].compute()
        df_test['date'] = pd.to_datetime(df_test['date'])
        rolled_test = roll_time_series(
            df_test,
            column_id='variable',
            column_sort='date',
            max_timeshift=self.time_window,
        )
        
        # Rolling - No persist (fast operation)
        print("🔄 Rolling time series...")
        rolled_dask = FC_dask.map_partitions(
            self.roll_dask_partition, 
            meta=rolled_test
        ).persist()
        
        # Feature extraction - Persist (expensive step)
        print("Extracting features...")
        features_dask = rolled_dask.map_partitions(
            self.extract_dask_partition, 
            enforce_metadata=False
        ).persist()
        
        # Feature selection - Compute directly (result)
        print("🎯 Selecting features...")
        selected_dask = features_dask.map_partitions(
            self.select_dask_partition, 
            y=y, 
            enforce_metadata=False
        ).persist()
        
        # Materialize and join results
        print("🔗 Materializing results...")
        out = None
        selected_futures = client.compute(selected_dask.to_delayed())
        
        for i, future in enumerate(selected_futures):
            try:
                df = future.result()
                if len(df) > 0:
                    if out is None:
                        out = df
                    else:
                        out = out.join(df, how='outer')
            except Exception as e:
                print(f"Error processing partition {i}: {e}")
                continue
        
        if out is not None:
            print(f"✅ TSFresh pipeline completed. Features shape: {out.shape}")
        else:
            print("⚠️ No features extracted from TSFresh pipeline")
            out = pd.DataFrame()
            
        return out

    def create_final_feature_set(self, 
                                X_base: pd.DataFrame,
                                y: pd.Series,
                                tsfresh_features: pd.DataFrame,
                                include_ta_indicators: bool = True) -> pd.DataFrame:
        """
        Create final feature set combining base features, TSFresh features, and TA indicators.
        
        Args:
            X_base: Base feature matrix
            y: Target series  
            tsfresh_features: TSFresh extracted features
            include_ta_indicators: Whether to include technical analysis indicators
            
        Returns:
            Combined feature matrix
        """
        print("🎯 Creating final feature set...")
        
        # Select relevant base features
        print("Selecting base features...")
        base_selected = select_features(
            X_base, y,
            fdr_level=self.fdr_level,
            ml_task='regression',
            hypotheses_independent=False
        )
        
        # Add technical analysis indicators
        if include_ta_indicators and TALIB_AVAILABLE:
            print("Computing technical analysis indicators...")
            ta_indicators = self.compute_ta_indicators(X_base, price_prefix="prices_")
            
            if not ta_indicators.empty:
                # Align TA indicators with features
                X_base = X_base.join(ta_indicators, how='left').dropna()
                X_base = X_base.loc[X_base.join(y, how='inner').dropna().index]
                y = y.loc[X_base.index]
        
        # Combine all features
        print("Combining feature sets...")
        if not tsfresh_features.empty:
            final_features = tsfresh_features.join(base_selected, how='left')
        else:
            final_features = base_selected
            
        # Add target for final alignment
        final_features = final_features.join(y, how='left')
        
        print(f"✅ Final feature set shape: {final_features.shape}")
        return final_features


# Convenience functions
def quick_feature_engineering(df: pd.DataFrame, 
                             target_coin: str = "ethereum",
                             client=None,
                             extraction_settings: str = "efficient") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Quick feature engineering pipeline for testing.
    
    Args:
        df: Input data DataFrame
        target_coin: Target cryptocurrency
        client: Dask client (optional)
        extraction_settings: TSFresh complexity level
        
    Returns:
        Tuple of (features, target)
    """
    engineer = CryptoFeatureEngineer(extraction_settings=extraction_settings)
    
    # Prepare target
    X, y = engineer.prepare_target_variable(df, target_coin)
    
    # Run TSFresh pipeline if client provided
    if client is not None:
        tsfresh_features = engineer.run_tsfresh_pipeline(X, y, client)
        final_features = engineer.create_final_feature_set(X, y, tsfresh_features)
    else:
        # Just use base features with TA indicators
        final_features = engineer.create_final_feature_set(X, y, pd.DataFrame(), include_ta_indicators=True)
    
    return final_features.drop('target', axis=1), final_features['target']


if __name__ == "__main__":
    # Test feature engineering
    print("Testing feature engineering module...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'prices_bitcoin': np.cumsum(np.random.randn(100)) + 50000,
        'prices_ethereum': np.cumsum(np.random.randn(100)) + 3000,
        'dvol_btc': np.random.gamma(2, 0.5, 100),
    }, index=dates)
    
    engineer = CryptoFeatureEngineer()
    X, y = engineer.prepare_target_variable(sample_data, target_coin="ethereum")
    
    print(f"Sample features shape: {X.shape}")
    print(f"Sample target shape: {y.shape}")
    print("✅ Feature engineering module test completed!")