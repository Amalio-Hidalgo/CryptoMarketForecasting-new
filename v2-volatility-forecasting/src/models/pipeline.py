"""
Time Series Machine Learning Pipeline

General-purpose ML pipeline for time series forecasting using XGBoost with Dask
integration for distributed computing. Supports any time series prediction task
including price forecasting, volatility prediction, demand forecasting, etc.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import dask as dxgb
from typing import Dict, Any, Optional, Tuple, List
import logging
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesMLPipeline:
    """
    General-purpose time series forecasting pipeline with XGBoost and Dask.

    This class provides a complete ML workflow for time series prediction tasks,
    with support for distributed computing via Dask. It handles training,
    optimization, and evaluation for any time series forecasting application.

    Features:
        - Dask XGBoost integration for distributed training
        - Optuna hyperparameter optimization
        - Time series-aware train/test splitting
        - Comprehensive evaluation metrics
        - Support for any regression target (volatility, prices, demand, etc.)

    Attributes:
        n_trials (int): Number of Optuna optimization trials
        n_rounds (int): Maximum XGBoost boosting rounds
        eval_metric (str): Evaluation metric for model selection
        tree_method (str): XGBoost tree construction algorithm
        early_stopping_rounds (int): Early stopping patience
        splits (int): Number of time series cross-validation folds
        random_seed (int): Random seed for reproducibility

    Example:
        >>> from dask.distributed import Client
        >>> client = Client()
        >>> pipeline = TimeSeriesMLPipeline(
        ...     n_trials=50,
        ...     n_rounds=200,
        ...     eval_metric='mae'
        ... )
        >>> dtrain, dtest, X_test, y_test = pipeline.prepare_dask_matrices(
        ...     final_features=df, client=client, test_ratio=0.2
        ... )
        >>> study = pipeline.optimize_hyperparameters(client, dtrain)
    """

    def __init__(
        self,
        n_trials: int = 100,
        n_rounds: int = 200,
        eval_metric: str = 'mae',
        tree_method: str = 'hist',
        early_stopping_rounds: int = 25,
        splits: int = 10,
        random_seed: int = 42
    ):
        """
        Initialize the time series ML pipeline.

        Args:
            n_trials: Number of hyperparameter optimization trials
            n_rounds: Maximum number of XGBoost boosting rounds
            eval_metric: Metric for model evaluation ('mae', 'rmse', 'r2')
            tree_method: XGBoost tree construction method
            early_stopping_rounds: Rounds without improvement before stopping
            splits: Number of time series cross-validation splits
            random_seed: Random seed for reproducibility
        """
        self.n_trials = n_trials
        self.n_rounds = n_rounds
        self.eval_metric = eval_metric
        self.tree_method = tree_method
        self.early_stopping_rounds = early_stopping_rounds
        self.splits = splits
        self.random_seed = random_seed

        # Model storage
        self.best_model: Optional[xgb.Booster] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.feature_names: Optional[List[str]] = None

        logger.info(f"Initialized TimeSeriesMLPipeline with {n_trials} trials, "
                   f"{n_rounds} rounds, eval_metric={eval_metric}")

    def prepare_dask_matrices(
        self,
        final_features: pd.DataFrame,
        client: Any,
        test_ratio: float = 0.2,
        target_col: str = 'target'
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Prepare Dask DMatrix objects for distributed XGBoost training.

        This method performs time series-aware train/test splitting and creates
        optimized DMatrix objects for efficient distributed training with Dask.

        Args:
            final_features: DataFrame with features and target column
            client: Dask distributed client
            test_ratio: Proportion of data for testing (default 0.2 = 20%)
            target_col: Name of the target column (default 'target')

        Returns:
            Tuple of (dtrain, dtest, X_test_dask, y_test_dask):
                - dtrain: Dask DMatrix for training
                - dtest: Dask DMatrix for testing
                - X_test_dask: Test features as Dask array
                - y_test_dask: Test target as Dask array

        Raises:
            ValueError: If target column is not found in final_features
        """
        logger.info(f"Preparing Dask matrices from {len(final_features)} samples...")

        if target_col not in final_features.columns:
            raise ValueError(f"Target column '{target_col}' not found in features. "
                           f"Available columns: {list(final_features.columns)}")

        # Time series split
        n_samples = len(final_features)
        split_idx = int(n_samples * (1 - test_ratio))

        # Split data
        train_data = final_features.iloc[:split_idx]
        test_data = final_features.iloc[split_idx:]

        # Separate features and target
        X_train = train_data.drop(columns=[target_col])
        y_train = train_data[target_col]
        X_test = test_data.drop(columns=[target_col])
        y_test = test_data[target_col]

        logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        logger.info(f"Features: {len(X_train.columns)}")

        # Store feature names
        self.feature_names = list(X_train.columns)

        # Convert to Dask arrays
        import dask.array as da
        import dask.dataframe as dd

        # Create Dask DataFrames
        ddf_X_train = dd.from_pandas(X_train, npartitions=4)
        ddf_y_train = dd.from_pandas(y_train, npartitions=4)
        ddf_X_test = dd.from_pandas(X_test, npartitions=2)
        ddf_y_test = dd.from_pandas(y_test, npartitions=2)

        # Convert to Dask arrays
        X_train_dask = ddf_X_train.to_dask_array(lengths=True)
        y_train_dask = ddf_y_train.to_dask_array(lengths=True)
        X_test_dask = ddf_X_test.to_dask_array(lengths=True)
        y_test_dask = ddf_y_test.to_dask_array(lengths=True)

        # Create DMatrix objects for Dask XGBoost
        dtrain = dxgb.DaskDMatrix(client, X_train_dask, y_train_dask)
        dtest = dxgb.DaskDMatrix(client, X_test_dask, y_test_dask)

        logger.info("✅ Dask matrices prepared successfully")

        return dtrain, dtest, X_test_dask, y_test_dask

    def optimize_hyperparameters(
        self,
        client: Any,
        dtrain: Any,
        n_trials: Optional[int] = None
    ) -> Any:
        """
        Optimize XGBoost hyperparameters using Optuna with Dask backend.

        Performs Bayesian optimization to find the best hyperparameters for
        the XGBoost model using the TPE (Tree-structured Parzen Estimator) sampler.

        Args:
            client: Dask distributed client
            dtrain: Dask DMatrix for training
            n_trials: Number of optimization trials (uses self.n_trials if None)

        Returns:
            Optuna Study object with optimization results

        Note:
            The optimization minimizes the eval_metric specified in __init__.
            Common metrics: 'mae', 'rmse', 'mape'
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            logger.error("Optuna not available. Install with: pip install optuna")
            raise

        n_trials = n_trials or self.n_trials
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")

        def objective(trial):
            """Optuna objective function for hyperparameter tuning."""
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': self.eval_metric,
                'tree_method': self.tree_method,
                'random_state': self.random_seed,
                'verbosity': 0,
                # Hyperparameters to optimize
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            }

            # Train with Dask XGBoost
            output = dxgb.train(
                client,
                params,
                dtrain,
                num_boost_round=self.n_rounds,
                evals=[(dtrain, 'train')],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False
            )

            # Extract best score
            if isinstance(output, dict):
                booster = output['booster']
                history = output['history']
                best_score = min(history['train'][self.eval_metric])
            else:
                best_score = output.best_score

            return best_score

        # Create and run study
        sampler = TPESampler(seed=self.random_seed)
        study = optuna.create_study(
            direction='minimize',  # Minimize error metrics
            sampler=sampler
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        self.best_params = study.best_params
        logger.info(f"✅ Optimization complete! Best {self.eval_metric}: {study.best_value:.6f}")
        logger.info(f"Best parameters: {self.best_params}")

        return study

    def train_final_model(
        self,
        client: Any,
        dtrain: Any,
        dtest: Any = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train final XGBoost model with optimized or provided parameters.

        Args:
            client: Dask distributed client
            dtrain: Dask DMatrix for training
            dtest: Dask DMatrix for testing (optional)
            params: Model parameters (uses self.best_params if None)

        Returns:
            Dictionary containing trained model and training history
        """
        if params is None:
            params = self.best_params or {}

        # Build complete parameter dict
        full_params = {
            'objective': 'reg:squarederror',
            'eval_metric': self.eval_metric,
            'tree_method': self.tree_method,
            'random_state': self.random_seed,
            'verbosity': 0,
            **params
        }

        logger.info("Training final model with optimized parameters...")

        # Prepare evaluation sets
        evals = [(dtrain, 'train')]
        if dtest is not None:
            evals.append((dtest, 'test'))

        # Train model
        output = dxgb.train(
            client,
            full_params,
            dtrain,
            num_boost_round=self.n_rounds,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=True
        )

        # Extract model and history
        if isinstance(output, dict):
            self.best_model = output['booster']
            history = output['history']
        else:
            self.best_model = output
            history = {}

        logger.info("✅ Model training completed")

        return {
            'model': self.best_model,
            'history': history,
            'params': full_params
        }

    def predict(
        self,
        client: Any,
        X: Any,
        model: Optional[Any] = None
    ) -> np.ndarray:
        """
        Make predictions using trained model.

        Args:
            client: Dask distributed client
            X: Features (Dask array or DMatrix)
            model: Trained model (uses self.best_model if None)

        Returns:
            Predictions as numpy array
        """
        if model is None:
            model = self.best_model

        if model is None:
            raise ValueError("No trained model available")

        # Create DMatrix if needed
        if not isinstance(X, dxgb.DaskDMatrix):
            dmatrix = dxgb.DaskDMatrix(client, X)
        else:
            dmatrix = X

        # Make predictions
        predictions = dxgb.predict(client, model, dmatrix)

        # Convert to numpy if Dask array
        if hasattr(predictions, 'compute'):
            predictions = predictions.compute()

        return predictions

    def compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_naive: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.

        Args:
            y_true: True target values
            y_pred: Predicted values
            y_naive: Naive baseline predictions (optional, for MASE)

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}

        # Basic regression metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['r2'] = r2_score(y_true, y_pred)

        # MASE (Mean Absolute Scaled Error)
        if y_naive is not None:
            mae_naive = mean_absolute_error(y_true, y_naive)
            metrics['mase'] = metrics['mae'] / mae_naive if mae_naive > 0 else np.nan

        # Normalized MAE
        median_abs = np.median(np.abs(y_true))
        metrics['normalized_mae'] = metrics['mae'] / median_abs if median_abs > 0 else np.nan

        # Summary statistics
        metrics['mean_target'] = float(y_true.mean())
        metrics['std_target'] = float(y_true.std())

        return metrics

    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        if self.best_model is None:
            raise ValueError("No trained model available")

        self.best_model.save_model(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load trained model from file."""
        self.best_model = xgb.Booster()
        self.best_model.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")


# Legacy alias for backward compatibility
CryptoVolatilityMLPipeline = TimeSeriesMLPipeline
