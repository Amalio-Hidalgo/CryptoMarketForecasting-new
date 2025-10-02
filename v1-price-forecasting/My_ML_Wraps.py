"""Machine Learning Model Training Module

This module provides XGBoost model training and optimization functions.
Supports both Pandas and Dask implementations with Optuna optimization.
"""

# Core ML imports
import xgboost as xgb
from xgboost import dask as dxgb
import optuna
from optuna.integration.dask import DaskStorage

# Sklearn components
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

# Data handling
import numpy as np

# Constants
DEFAULT_RGS_METRIC = 'neg_mean_absolute_error'  # For RandomizedSearchCV
DEFAULT_XGB_METRIC = 'mae'  # For XGBoost/Optuna
DEFAULT_TREE_METHOD = 'hist'
DEFAULT_EARLY_STOPPING = 25
DEFAULT_N_TRIALS = 50
DEFAULT_N_ROUNDS = 100
DEFAULT_N_ITER = 20
DEFAULT_CV_SPLITS = 5

def RGS_XGB_Pandas(X_train, X_test, y_train, y_test, 
                   parameter_grid=None, 
                   number_cvs=DEFAULT_CV_SPLITS, 
                   n_iter=DEFAULT_N_ITER, 
                   tree_method=DEFAULT_TREE_METHOD, 
                   eval_metric=DEFAULT_RGS_METRIC):
    """XGBoost with random grid search for hyperparameter tuning
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_train (pd.Series): Training target
        y_test (pd.Series): Test target
        parameter_grid (dict): Optional custom parameter grid
        number_cvs (int): Number of cross-validation splits
        n_iter (int): Number of random search iterations
        tree_method (str): XGBoost tree construction algorithm; pass gpu hist if data cp.arrays or dfs; else hist.

    
    Returns:
        tuple: (trained_model, best_params, cv_results)
    """
    
    if parameter_grid is None:
        parameter_grid = {
            'num_parallel_tree': [1, 3, 5, 10, 20],
            'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
            'max_depth': [3, 6, 12, 24],
            'gamma': [0, 0.01, 0.05, 0.1, 0.5, 1],
            'min_child_weight': [0.5, 1, 3, 5],
            'subsample': [0.5, 0.75, 1.0],
            'colsample_bytree': [0.5, 0.75, 1.0],
            'grow_policy': ['depthwise', 'lossguide']
        }
    
    # Setup cross-validation
    tscv = TimeSeriesSplit(n_splits=number_cvs)
    
    # Base model configuration
    base_model = xgb.XGBRegressor(
        eval_metric='mae',
        early_stopping_rounds=25,
        tree_method=tree_method,
        n_jobs=-1,  # Use all CPU cores
        verbose=0
    )
    
    # Random search with cross-validation
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=parameter_grid,
        n_iter=n_iter,
        cv=tscv,
        verbose=1,
        n_jobs=-1,
        refit=True,
        scoring=eval_metric
    )
    
    # Fit the model
    try:
        search.fit(
            X_train, 
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        return search
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None, None, None

def Optuna_XGB_Dask(client, dtrain, 
                    n_trials=DEFAULT_N_TRIALS, 
                    n_rounds=DEFAULT_N_ROUNDS, 
                    eval_metric=DEFAULT_XGB_METRIC,
                    tree_method=DEFAULT_TREE_METHOD, 
                    parameter_grid=None, 
                    early_stopping_rounds=DEFAULT_EARLY_STOPPING):
    """XGBoost optimization with Optuna using DaskArgs:
        dtrain: Dask DMatrix (already created with client)
        n_trials: Number of optimization trials
        n_rounds: Number of boosting rounds
        eval_metric: Evaluation metric ('mae', 'rmse', etc.)
        tree_method: XGBoost tree construction algorithm
        parameter_grid: Optional custom parameter grid
        early_stopping_rounds: Number of rounds for early stopping
    Returns:
        optuna.Study: Optimization study results
    """    
    def objective(trial):
        if parameter_grid is None:
            param_grid = {
                "verbosity": 1,
                "tree_method": tree_method,
                "eval_metric": eval_metric,
                "lambda": trial.suggest_float("lambda", 1e-8, 100.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-8, 100.0, log=True),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
                "max_depth": trial.suggest_int("max_depth", 2, 20),
                "min_child_weight": trial.suggest_float("min_child_weight", 1e-8, 100, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            }   
        else: 
            param_grid = parameter_grid
            
        output = dxgb.train(
            client,
            param_grid,
            dtrain,
            num_boost_round=n_rounds,
            early_stopping_rounds=early_stopping_rounds,
            evals=[(dtrain, "train")]
        )
        return output["history"]["train"][eval_metric][-1]

    # Create study with parallel optimization
    storage = DaskStorage()
    study = optuna.create_study(direction="minimize", storage= storage)
    study.optimize(
        objective, 
        n_trials=n_trials,
        n_jobs=-1,  # Use all available cores
        gc_after_trial=True,  # Clean memory after each trial
        show_progress_bar=True
        )
    
    return study

