"""
Machine Learning Pipeline for Cryptocurrency Volatility Forecasting

This module implements an end-to-end machine learning pipeline combining:
- XGBoost regression for volatility prediction
- Optuna hyperparameter optimization
- Time series cross-validation
- Feature importance analysis

The pipeline handles the full modeling workflow from feature matrix to final predictions,
with comprehensive evaluation metrics and model diagnostics.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, Optional, Tuple, List
import logging
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoVolatilityMLPipeline:
    """
    XGBoost-based machine learning pipeline for cryptocurrency volatility forecasting.

    This class orchestrates the complete modeling workflow including training,
    cross-validation, hyperparameter optimization, and evaluation. It implements
    time series-aware validation strategies to prevent data leakage.

    Attributes:
        n_trials (int): Number of Optuna optimization trials
        n_rounds (int): Maximum XGBoost boosting rounds
        eval_metric (str): Evaluation metric for model selection
        tree_method (str): XGBoost tree construction algorithm
        early_stopping_rounds (int): Early stopping patience
        splits (int): Number of time series cross-validation folds
        random_seed (int): Random seed for reproducibility

    Example:
        >>> pipeline = CryptoVolatilityMLPipeline(
        ...     n_trials=50,
        ...     n_rounds=200,
        ...     eval_metric='mae',
        ...     splits=5
        ... )
        >>> results = pipeline.train_and_evaluate(X_train, y_train, X_test, y_test)
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
        Initialize the ML pipeline with configuration parameters.

        Args:
            n_trials: Number of hyperparameter optimization trials
            n_rounds: Maximum number of XGBoost boosting rounds
            eval_metric: Metric for model evaluation ('mae', 'rmse', 'r2')
            tree_method: XGBoost tree construction method ('hist', 'exact', 'approx')
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
        self.best_model: Optional[xgb.XGBRegressor] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.cv_scores: Optional[List[float]] = None
        self.feature_importance: Optional[pd.DataFrame] = None

        logger.info(f"Initialized CryptoVolatilityMLPipeline with {n_trials} trials, "
                   f"{n_rounds} rounds, eval_metric={eval_metric}")

    def create_baseline_model(self) -> xgb.XGBRegressor:
        """
        Create a baseline XGBoost model with default parameters.

        Returns:
            Configured XGBoost regressor
        """
        return xgb.XGBRegressor(
            n_estimators=self.n_rounds,
            objective='reg:squarederror',
            tree_method=self.tree_method,
            eval_metric=self.eval_metric,
            random_state=self.random_seed,
            early_stopping_rounds=self.early_stopping_rounds,
            verbosity=0,
            # Reasonable defaults
            max_depth=6,
            learning_rate=0.1,
            min_child_weight=3.0,
            subsample=0.8,
            colsample_bytree=0.8
        )

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> xgb.XGBRegressor:
        """
        Train an XGBoost model with optional validation set.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            params: Custom XGBoost parameters (optional)

        Returns:
            Trained XGBoost model
        """
        if params is None:
            model = self.create_baseline_model()
        else:
            model_params = {
                'n_estimators': params.get('n_estimators', self.n_rounds),
                'objective': 'reg:squarederror',
                'tree_method': self.tree_method,
                'eval_metric': self.eval_metric,
                'random_state': self.random_seed,
                'early_stopping_rounds': self.early_stopping_rounds,
                'verbosity': 0,
                **{k: v for k, v in params.items() if k != 'n_estimators'}
            }
            model = xgb.XGBRegressor(**model_params)

        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        # Train model
        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False
        )

        return model

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[float], List[xgb.XGBRegressor]]:
        """
        Perform time series cross-validation.

        Args:
            X: Feature matrix
            y: Target variable
            params: Model parameters (optional)

        Returns:
            Tuple of (scores, models) from each fold
        """
        tscv = TimeSeriesSplit(n_splits=self.splits)
        scores = []
        models = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = self.train_model(X_train, y_train, X_val, y_val, params)
            models.append(model)

            # Evaluate
            y_pred = model.predict(X_val)
            score = self._compute_metric(y_val, y_pred)
            scores.append(score)

            logger.debug(f"Fold {fold+1}/{self.splits}: {self.eval_metric}={score:.6f}")

        return scores, models

    def _compute_metric(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Compute evaluation metric."""
        if self.eval_metric == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif self.eval_metric == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif self.eval_metric == 'r2':
            return r2_score(y_true, y_pred)
        else:
            return mean_absolute_error(y_true, y_pred)

    def compute_comprehensive_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Dictionary of evaluation metrics
        """
        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # MASE (Mean Absolute Scaled Error) - scaled against naive forecast
        naive_forecast = y_true.shift(1)
        mae_naive = mean_absolute_error(y_true[1:], naive_forecast[1:])
        mase = mae / mae_naive if mae_naive > 0 else np.nan

        # Relative metrics
        median_abs_target = np.median(np.abs(y_true))
        normalized_mae = mae / median_abs_target if median_abs_target > 0 else np.nan

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mase': mase,
            'normalized_mae': normalized_mae,
            'mean_target': float(y_true.mean()),
            'std_target': float(y_true.std())
        }

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train model and evaluate on test set.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            params: Model parameters (optional)

        Returns:
            Dictionary containing model, predictions, and metrics
        """
        logger.info("Training model...")

        # Train model
        model = self.train_model(X_train, y_train, X_test, y_test, params)
        self.best_model = model

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Compute metrics
        train_metrics = self.compute_comprehensive_metrics(y_train, y_pred_train)
        test_metrics = self.compute_comprehensive_metrics(y_test, y_pred_test)

        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info(f"Training complete. Test MAE: {test_metrics['mae']:.6f}")

        return {
            'model': model,
            'predictions_train': y_pred_train,
            'predictions_test': y_pred_test,
            'metrics_train': train_metrics,
            'metrics_test': test_metrics,
            'feature_importance': self.feature_importance
        }

    def run_complete_pipeline(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run complete ML pipeline with train/test split.

        Args:
            X: Complete feature matrix
            y: Complete target variable
            test_size: Proportion of data for testing
            params: Model parameters (optional)

        Returns:
            Dictionary with complete results
        """
        logger.info("Running complete ML pipeline...")

        # Time series split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(f"Split: {len(X_train)} train, {len(X_test)} test samples")

        # Train and evaluate
        results = self.train_and_evaluate(X_train, y_train, X_test, y_test, params)

        # Add cross-validation scores
        logger.info("Performing cross-validation...")
        cv_scores, cv_models = self.cross_validate(X_train, y_train, params)
        self.cv_scores = cv_scores

        results['cv_scores'] = cv_scores
        results['cv_mean'] = np.mean(cv_scores)
        results['cv_std'] = np.std(cv_scores)

        logger.info(f"CV Score: {results['cv_mean']:.6f} (+/- {results['cv_std']:.6f})")

        return results

    def optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        optimization_trials: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.

        Args:
            X: Feature matrix
            y: Target variable
            optimization_trials: Number of optimization trials (uses self.n_trials if None)

        Returns:
            Dictionary with best parameters and optimization results
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            logger.warning("Optuna not available. Skipping hyperparameter optimization.")
            return {'best_params': None, 'best_score': None}

        n_trials = optimization_trials or self.n_trials
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")

        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
            }

            # Cross-validate with these parameters
            scores, _ = self.cross_validate(X, y, params)
            return np.mean(scores)

        # Create study and optimize
        sampler = TPESampler(seed=self.random_seed)
        study = optuna.create_study(
            direction='minimize' if self.eval_metric in ['mae', 'rmse'] else 'maximize',
            sampler=sampler
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        self.best_params = study.best_params

        logger.info(f"Optimization complete. Best {self.eval_metric}: {study.best_value:.6f}")
        logger.info(f"Best parameters: {self.best_params}")

        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study
        }

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top N most important features.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with top features and their importance scores
        """
        if self.feature_importance is None:
            raise ValueError("Model has not been trained yet. Call train_and_evaluate() first.")

        return self.feature_importance.head(top_n)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. Call train_and_evaluate() first.")

        return self.best_model.predict(X)

    def save_model(self, filepath: str) -> None:
        """
        Save trained model to file.

        Args:
            filepath: Path to save model
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. Call train_and_evaluate() first.")

        self.best_model.save_model(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load trained model from file.

        Args:
            filepath: Path to model file
        """
        self.best_model = xgb.XGBRegressor()
        self.best_model.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
