"""
Crypto Volatility Forecasting Toolkit - Dask-Optimized Pipeline
================================================================

A comprehensive toolkit for cryptocurrency volatility prediction using:
- Dask-distributed feature engineering and time series rolling
- TSFresh feature extraction with Dask backend
- XGBoost training with DMatrix optimization
- Professional model evaluation and visualization

Author: Your Name
License: MIT
"""

import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
from dask.distributed import Client, as_completed
from dask import delayed
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# TSFresh imports for Dask distribution
from tsfresh import extract_relevant_features, extract_features
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters, ComprehensiveFCParameters
from tsfresh.utilities.distribution import ClusterDaskDistributor
from tsfresh.feature_selection.relevance import calculate_relevance_table

class DaskCryptoVolatilityPipeline:
    """
    Main pipeline class for cryptocurrency volatility forecasting using Dask.
    
    Features:
    - Distributed time series rolling and feature extraction
    - Memory-efficient processing with Dask
    - XGBoost optimization with DMatrix
    - Comprehensive model evaluation and visualization
    """
    
    def __init__(self, client: Client, random_state: int = 42):
        """
        Initialize the pipeline with Dask client.
        
        Parameters:
        -----------
        client : dask.distributed.Client
            Active Dask client for distributed computing
        random_state : int
            Random seed for reproducibility
        """
        self.client = client
        self.random_state = random_state
        self.feature_names_ = None
        self.model_ = None
        self.scaler_ = None
        
    def dask_roll_time_series(self, 
                             df: pd.DataFrame,
                             column_id: str = 'variable',
                             column_sort: str = 'date', 
                             column_value: str = 'value',
                             max_timeshift: int = 7,
                             min_timeshift: int = 0,
                             n_partitions: int = 10) -> dd.DataFrame:
        """
        Dask-distributed time series rolling for large datasets.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Long-format dataframe with id, time, and value columns
        column_id : str
            Column name for series identifier
        column_sort : str  
            Column name for time/date sorting
        column_value : str
            Column name for values
        max_timeshift : int
            Maximum lag for rolling windows
        min_timeshift : int
            Minimum lag for rolling windows
        n_partitions : int
            Number of Dask partitions
            
        Returns:
        --------
        dd.DataFrame
            Dask DataFrame with rolled time series
        """
        print(f"🔄 Rolling time series with Dask - Max timeshift: {max_timeshift}")
        
        # Convert to Dask DataFrame if needed
        if isinstance(df, pd.DataFrame):
            ddf = dd.from_pandas(df, npartitions=n_partitions)
        else:
            ddf = df
            
        def roll_partition(partition):
            """Roll time series for a single partition"""
            try:
                if len(partition) == 0:
                    return pd.DataFrame()
                    
                rolled = roll_time_series(
                    partition,
                    column_id=column_id,
                    column_sort=column_sort,
                    max_timeshift=max_timeshift,
                    min_timeshift=min_timeshift,
                    rolling_direction=1
                )
                return rolled
            except Exception as e:
                print(f"⚠️  Error in partition rolling: {e}")
                return pd.DataFrame()
        
        # Define meta for the output
        meta = pd.DataFrame({
            'id': pd.Series([], dtype='int64'),
            column_id: pd.Series([], dtype='object'),
            column_sort: pd.Series([], dtype='datetime64[ns]'),
            column_value: pd.Series([], dtype='float64')
        })
        
        # Apply rolling to partitions
        rolled_ddf = ddf.map_partitions(
            roll_partition,
            meta=meta
        )
        
        return rolled_ddf.persist()
    
    def dask_extract_features(self,
                             rolled_df: dd.DataFrame,
                             column_id: str = 'id',
                             column_sort: str = 'date',
                             column_kind: str = 'variable', 
                             column_value: str = 'value',
                             fc_parameters: Any = None,
                             n_jobs: int = -1) -> pd.DataFrame:
        """
        Extract features using TSFresh with Dask distribution.
        
        Parameters:
        -----------
        rolled_df : dd.DataFrame
            Rolled time series DataFrame
        column_id : str
            ID column for grouping
        column_sort : str
            Sort column (time)
        column_kind : str
            Kind column for feature types
        column_value : str
            Value column
        fc_parameters : dict or FeatureCalculationSettings
            TSFresh feature calculation parameters
        n_jobs : int
            Number of parallel jobs
            
        Returns:
        --------
        pd.DataFrame
            Extracted features matrix
        """
        print("🧠 Extracting features with TSFresh + Dask distribution")
        
        if fc_parameters is None:
            fc_parameters = EfficientFCParameters()
        
        # Convert to pandas for TSFresh (compute in chunks if needed)
        if isinstance(rolled_df, dd.DataFrame):
            # For large datasets, process in chunks
            total_size = rolled_df.map_partitions(len).sum().compute()
            print(f"📊 Processing {total_size:,} rolled observations")
            
            if total_size > 1_000_000:  # Process in chunks for very large datasets
                print("📦 Processing in chunks due to large dataset size")
                chunk_dfs = []
                for i in range(rolled_df.npartitions):
                    chunk = rolled_df.get_partition(i).compute()
                    if len(chunk) > 0:
                        chunk_features = extract_features(
                            chunk,
                            column_id=column_id,
                            column_sort=column_sort,
                            column_kind=column_kind,
                            column_value=column_value,
                            default_fc_parameters=fc_parameters,
                            distributor=ClusterDaskDistributor(
                                address=self.client.scheduler.address
                            )
                        )
                        chunk_dfs.append(chunk_features)
                
                # Combine chunks
                if chunk_dfs:
                    features_df = pd.concat(chunk_dfs, axis=0)
                else:
                    features_df = pd.DataFrame()
            else:
                # Small enough to process all at once
                rolled_pd = rolled_df.compute()
                features_df = extract_features(
                    rolled_pd,
                    column_id=column_id,
                    column_sort=column_sort, 
                    column_kind=column_kind,
                    column_value=column_value,
                    default_fc_parameters=fc_parameters,
                    distributor=ClusterDaskDistributor(
                        address=self.client.scheduler.address
                    )
                )
        else:
            # Already pandas DataFrame
            features_df = extract_features(
                rolled_df,
                column_id=column_id,
                column_sort=column_sort,
                column_kind=column_kind, 
                column_value=column_value,
                default_fc_parameters=fc_parameters,
                distributor=ClusterDaskDistributor(
                    address=self.client.scheduler.address
                )
            )
        
        print(f"✅ Extracted {features_df.shape[1]} features from {features_df.shape[0]} samples")
        return features_df
    
    def dask_feature_selection(self,
                              features_df: pd.DataFrame,
                              y: pd.Series,
                              fdr_level: float = 0.05,
                              test_for_binary_target_binary_feature: str = 'fisher') -> pd.DataFrame:
        """
        Distributed feature selection using Dask.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Features matrix
        y : pd.Series
            Target variable
        fdr_level : float
            False discovery rate level
        test_for_binary_target_binary_feature : str
            Statistical test type
            
        Returns:
        --------
        pd.DataFrame
            Selected features matrix
        """
        print(f"🎯 Selecting relevant features (FDR level: {fdr_level})")
        
        # Align indices
        common_idx = features_df.index.intersection(y.index)
        features_aligned = features_df.loc[common_idx]
        y_aligned = y.loc[common_idx]
        
        print(f"📊 Feature selection on {features_aligned.shape[0]} samples, {features_aligned.shape[1]} features")
        
        # Calculate relevance table with Dask
        relevance_table = calculate_relevance_table(
            features_aligned, 
            y_aligned,
            fdr_level=fdr_level,
            test_for_binary_target_binary_feature=test_for_binary_target_binary_feature
        )
        
        # Select relevant features
        relevant_features = relevance_table[relevance_table.relevant == True]['feature'].tolist()
        selected_features_df = features_aligned[relevant_features]
        
        print(f"✅ Selected {len(relevant_features)} relevant features")
        self.feature_names_ = relevant_features
        
        return selected_features_df
    
    def prepare_dmatrix(self, 
                       X: pd.DataFrame, 
                       y: pd.Series = None,
                       enable_categorical: bool = False) -> xgb.DMatrix:
        """
        Prepare optimized XGBoost DMatrix for training/prediction.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features matrix
        y : pd.Series, optional
            Target variable (for training)
        enable_categorical : bool
            Whether to enable categorical features
            
        Returns:
        --------
        xgb.DMatrix
            Optimized DMatrix for XGBoost
        """
        print(f"🔧 Preparing DMatrix: {X.shape[0]} samples × {X.shape[1]} features")
        
        # Handle missing values
        X_clean = X.fillna(X.median())
        
        # Handle infinite values
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan).fillna(X_clean.median())
        
        if y is not None:
            y_clean = y.fillna(y.median())
            dmatrix = xgb.DMatrix(
                X_clean.values,
                label=y_clean.values,
                feature_names=[str(col) for col in X_clean.columns],
                enable_categorical=enable_categorical
            )
        else:
            dmatrix = xgb.DMatrix(
                X_clean.values,
                feature_names=[str(col) for col in X_clean.columns],
                enable_categorical=enable_categorical
            )
            
        return dmatrix
    
    def train_xgboost_model(self,
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           X_val: pd.DataFrame = None,
                           y_val: pd.Series = None,
                           xgb_params: Dict = None,
                           num_boost_round: int = 1000,
                           early_stopping_rounds: int = 50,
                           verbose_eval: int = 100) -> xgb.Booster:
        """
        Train XGBoost model with optimized DMatrix and early stopping.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_val : pd.DataFrame, optional
            Validation features
        y_val : pd.Series, optional
            Validation target
        xgb_params : dict, optional
            XGBoost parameters
        num_boost_round : int
            Maximum boosting rounds
        early_stopping_rounds : int
            Early stopping patience
        verbose_eval : int
            Verbose evaluation frequency
            
        Returns:
        --------
        xgb.Booster
            Trained XGBoost model
        """
        print("🚀 Training XGBoost model with DMatrix optimization")
        
        if xgb_params is None:
            xgb_params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'mae',
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'tree_method': 'hist',
                'random_state': self.random_state,
                'verbosity': 0
            }
        
        # Prepare training DMatrix
        dtrain = self.prepare_dmatrix(X_train, y_train)
        
        # Prepare evaluation sets
        eval_sets = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = self.prepare_dmatrix(X_val, y_val)
            eval_sets.append((dval, 'validation'))
        
        # Train model
        model = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=eval_sets,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval
        )
        
        self.model_ = model
        print(f"✅ Model training completed. Best iteration: {model.best_iteration}")
        
        return model
    
    def predict(self, X: pd.DataFrame, model: xgb.Booster = None) -> np.ndarray:
        """
        Make predictions using trained XGBoost model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features for prediction
        model : xgb.Booster, optional
            Trained model (uses self.model_ if None)
            
        Returns:
        --------
        np.ndarray
            Predictions
        """
        if model is None:
            model = self.model_
        
        if model is None:
            raise ValueError("No trained model available. Train model first.")
        
        # Ensure we have the same features as training
        if self.feature_names_ is not None:
            X_pred = X[self.feature_names_]
        else:
            X_pred = X
            
        dtest = self.prepare_dmatrix(X_pred)
        predictions = model.predict(dtest)
        
        return predictions
    
    def evaluate_model(self, 
                      y_true: pd.Series, 
                      y_pred: np.ndarray,
                      y_naive: pd.Series = None) -> Dict[str, float]:
        """
        Comprehensive model evaluation with multiple metrics.
        
        Parameters:
        -----------
        y_true : pd.Series
            True values
        y_pred : np.ndarray
            Predicted values
        y_naive : pd.Series, optional
            Naive baseline predictions for MASE calculation
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Basic regression metrics
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        metrics['MSE'] = np.mean((y_true - y_pred) ** 2)
        metrics['RMSE'] = np.sqrt(metrics['MSE'])
        metrics['R2'] = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        non_zero_mask = y_true != 0
        if non_zero_mask.sum() > 0:
            metrics['MAPE'] = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            metrics['MAPE'] = np.nan
        
        # Mean Absolute Scaled Error (MASE)
        if y_naive is not None:
            naive_mae = mean_absolute_error(y_true, y_naive)
            metrics['MASE'] = metrics['MAE'] / naive_mae if naive_mae != 0 else np.nan
        
        # Directional accuracy (for volatility forecasting)
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            metrics['Directional_Accuracy'] = np.mean(true_direction == pred_direction) * 100
        
        return metrics
    
    def plot_predictions(self, 
                        y_true: pd.Series, 
                        y_pred: np.ndarray,
                        title: str = "Volatility Predictions",
                        figsize: Tuple[int, int] = (15, 8)):
        """
        Create comprehensive prediction visualization.
        
        Parameters:
        -----------
        y_true : pd.Series
            True values with datetime index
        y_pred : np.ndarray
            Predicted values
        title : str
            Plot title
        figsize : tuple
            Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Time series plot
        axes[0, 0].plot(y_true.index, y_true.values, label='Actual', alpha=0.7, linewidth=1.5)
        axes[0, 0].plot(y_true.index, y_pred, label='Predicted', alpha=0.7, linewidth=1.5)
        axes[0, 0].set_title('Time Series Comparison')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Realized Volatility')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[0, 1].scatter(y_true.values, y_pred, alpha=0.6, s=20)
        axes[0, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', alpha=0.8)
        axes[0, 1].set_title('Actual vs Predicted')
        axes[0, 1].set_xlabel('Actual')
        axes[0, 1].set_ylabel('Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_true.values - y_pred
        axes[1, 0].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[1, 0].set_title('Residuals vs Predicted')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, density=True)
        axes[1, 1].set_title('Residuals Distribution')
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, 
                               model: xgb.Booster = None,
                               importance_type: str = 'weight',
                               max_features: int = 20,
                               figsize: Tuple[int, int] = (10, 8)):
        """
        Plot feature importance from XGBoost model.
        
        Parameters:
        -----------
        model : xgb.Booster, optional
            Trained model (uses self.model_ if None)
        importance_type : str
            Type of importance ('weight', 'gain', 'cover')
        max_features : int
            Maximum number of features to display
        figsize : tuple
            Figure size
        """
        if model is None:
            model = self.model_
            
        if model is None:
            raise ValueError("No trained model available.")
        
        # Get feature importance
        importance = model.get_score(importance_type=importance_type)
        
        if not importance:
            print("No feature importance available.")
            return
        
        # Sort and limit features
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_importance[:max_features]
        
        features, scores = zip(*top_features)
        
        # Create plot
        plt.figure(figsize=figsize)
        y_pos = np.arange(len(features))
        
        plt.barh(y_pos, scores, alpha=0.8)
        plt.yticks(y_pos, features)
        plt.xlabel(f'Importance ({importance_type})')
        plt.title(f'Top {len(features)} Feature Importances')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def run_complete_pipeline(self,
                             X_wide: pd.DataFrame,
                             y: pd.Series,
                             test_size: float = 0.2,
                             max_timeshift: int = 7,
                             fc_parameters: Any = None,
                             fdr_level: float = 0.05,
                             xgb_params: Dict = None,
                             plot_results: bool = True) -> Dict[str, Any]:
        """
        Run the complete volatility forecasting pipeline.
        
        Parameters:
        -----------
        X_wide : pd.DataFrame
            Wide format features DataFrame with datetime index
        y : pd.Series
            Target variable (realized volatility)
        test_size : float
            Test set proportion
        max_timeshift : int
            Maximum time shift for rolling
        fc_parameters : Any
            TSFresh feature calculation parameters
        fdr_level : float
            False discovery rate for feature selection
        xgb_params : dict
            XGBoost parameters
        plot_results : bool
            Whether to create visualizations
            
        Returns:
        --------
        dict
            Complete pipeline results including model, predictions, and metrics
        """
        print("🚀 Starting complete Dask crypto volatility forecasting pipeline")
        print("=" * 70)
        
        # Step 1: Data preparation
        print("📊 Step 1: Data preparation and train/test split")
        n_samples = len(X_wide)
        n_train = int(n_samples * (1 - test_size))
        
        X_train_wide = X_wide.iloc[:n_train]
        X_test_wide = X_wide.iloc[n_train:]
        y_train = y.iloc[:n_train]
        y_test = y.iloc[n_train:]
        
        print(f"   Training samples: {len(X_train_wide)}")
        print(f"   Test samples: {len(X_test_wide)}")
        
        # Step 2: Convert to long format and roll time series
        print("\n🔄 Step 2: Converting to long format and rolling time series")
        
        # Convert training data to long format
        stacked_train = X_train_wide.reset_index().melt(
            id_vars=['date'], 
            var_name='variable', 
            value_name='value'
        )
        
        # Roll time series with Dask
        rolled_train = self.dask_roll_time_series(
            stacked_train,
            column_id='variable',
            column_sort='date',
            column_value='value',
            max_timeshift=max_timeshift
        )
        
        # Step 3: Feature extraction
        print("\n🧠 Step 3: Feature extraction with TSFresh")
        features_df = self.dask_extract_features(
            rolled_train,
            column_id='id',
            column_sort='date',
            column_kind='variable',
            column_value='value',
            fc_parameters=fc_parameters or EfficientFCParameters()
        )
        
        # Step 4: Feature selection
        print("\n🎯 Step 4: Feature selection")
        # Align features with target
        common_idx = features_df.index.intersection(y_train.index)
        features_aligned = features_df.loc[common_idx]
        y_train_aligned = y_train.loc[common_idx]
        
        selected_features = self.dask_feature_selection(
            features_aligned,
            y_train_aligned,
            fdr_level=fdr_level
        )
        
        # Step 5: Model training
        print("\n🚀 Step 5: XGBoost model training")
        
        # Split training data for validation
        val_split = int(len(selected_features) * 0.8)
        X_train_final = selected_features.iloc[:val_split]
        X_val_final = selected_features.iloc[val_split:]
        y_train_final = y_train_aligned.iloc[:val_split]
        y_val_final = y_train_aligned.iloc[val_split:]
        
        model = self.train_xgboost_model(
            X_train_final,
            y_train_final,
            X_val_final,
            y_val_final,
            xgb_params=xgb_params
        )
        
        # Step 6: Test set prediction
        print("\n🔮 Step 6: Test set predictions")
        
        # Process test data through the same pipeline
        stacked_test = X_test_wide.reset_index().melt(
            id_vars=['date'],
            var_name='variable', 
            value_name='value'
        )
        
        rolled_test = self.dask_roll_time_series(
            stacked_test,
            max_timeshift=max_timeshift
        )
        
        features_test = self.dask_extract_features(
            rolled_test,
            fc_parameters=fc_parameters or EfficientFCParameters()
        )
        
        # Align test features with training feature set
        test_features_aligned = features_test.reindex(
            columns=selected_features.columns
        ).fillna(0)
        
        # Make predictions
        y_pred_test = self.predict(test_features_aligned, model)
        
        # Step 7: Evaluation
        print("\n📈 Step 7: Model evaluation")
        
        # Create naive baseline
        y_naive = y_test.shift(1).fillna(method='bfill')
        
        # Calculate metrics
        metrics = self.evaluate_model(
            y_test.iloc[:len(y_pred_test)],
            y_pred_test,
            y_naive.iloc[:len(y_pred_test)]
        )
        
        print("\n📊 Model Performance Metrics:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"{metric:20s}: {value:.6f}")
        
        # Step 8: Visualization
        if plot_results:
            print("\n📈 Step 8: Creating visualizations")
            
            self.plot_predictions(
                y_test.iloc[:len(y_pred_test)],
                y_pred_test,
                title="Crypto Volatility Forecasting Results"
            )
            
            self.plot_feature_importance(
                model,
                importance_type='gain',
                max_features=20
            )
        
        # Compile results
        results = {
            'model': model,
            'features': selected_features,
            'feature_names': self.feature_names_,
            'predictions': {
                'y_test': y_test.iloc[:len(y_pred_test)],
                'y_pred': y_pred_test,
                'y_naive': y_naive.iloc[:len(y_pred_test)]
            },
            'metrics': metrics,
            'pipeline_config': {
                'max_timeshift': max_timeshift,
                'fdr_level': fdr_level,
                'test_size': test_size,
                'fc_parameters': str(fc_parameters or EfficientFCParameters()),
                'xgb_params': xgb_params
            }
        }
        
        print("\n✅ Pipeline completed successfully!")
        print("=" * 70)
        
        return results

# Utility functions for quick access
def create_dask_client(n_workers: int = 4, 
                      threads_per_worker: int = 2,
                      memory_limit: str = '4GB',
                      dashboard_port: int = 8787) -> Client:
    """
    Create optimized Dask client for crypto volatility forecasting.
    
    Parameters:
    -----------
    n_workers : int
        Number of workers
    threads_per_worker : int
        Threads per worker
    memory_limit : str
        Memory limit per worker
    dashboard_port : int
        Dashboard port
        
    Returns:
    --------
    dask.distributed.Client
        Configured Dask client
    """
    from dask.distributed import LocalCluster
    
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        dashboard_address=f':{dashboard_port}',
        processes=True
    )
    
    client = Client(cluster)
    print(f"🚀 Dask client created: {client.dashboard_link}")
    
    return client

def compute_mase(y_true: np.ndarray, 
                y_pred: np.ndarray, 
                y_naive: np.ndarray) -> float:
    """
    Compute Mean Absolute Scaled Error (MASE).
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    y_naive : np.ndarray
        Naive baseline predictions
        
    Returns:
    --------
    float
        MASE score
    """
    mae_pred = np.mean(np.abs(y_true - y_pred))
    mae_naive = np.mean(np.abs(y_true - y_naive))
    
    return mae_pred / mae_naive if mae_naive != 0 else np.inf

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Dask Crypto Volatility Pipeline")
    
    # Create sample data for testing
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    np.random.seed(42)
    
    # Sample crypto price data
    sample_data = pd.DataFrame({
        'prices_bitcoin': np.cumsum(np.random.randn(1000)) + 50000,
        'prices_ethereum': np.cumsum(np.random.randn(1000)) + 3000,
        'dvol_btc': np.random.gamma(2, 0.5, 1000),
        'vix_equity_vol': np.random.gamma(3, 0.3, 1000) + 15
    }, index=dates)
    
    # Create target variable (realized volatility)
    returns = np.log(sample_data['prices_ethereum']).diff()
    realized_vol = returns.abs().shift(-1).dropna()
    
    # Align data
    sample_data = sample_data.iloc[:-1]  # Remove last row to align with target
    
    print("📊 Sample data created:")
    print(f"   Features shape: {sample_data.shape}")
    print(f"   Target shape: {realized_vol.shape}")
    
    # Test the pipeline
    try:
        client = create_dask_client(n_workers=2, threads_per_worker=2)
        
        pipeline = DaskCryptoVolatilityPipeline(client)
        
        results = pipeline.run_complete_pipeline(
            X_wide=sample_data,
            y=realized_vol,
            test_size=0.2,
            max_timeshift=5,
            fc_parameters=MinimalFCParameters(),  # Use minimal for testing
            plot_results=False  # Disable plots for testing
        )
        
        print("\n✅ Pipeline test completed successfully!")
        print(f"📊 Test MASE: {results['metrics']['MASE']:.4f}")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()