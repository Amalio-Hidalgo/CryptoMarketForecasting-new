# tsxg_pipeline.py
# ---------------------------------------------------------------------
# Purpose: End-to-end feature extraction (tsfresh) + feature selection +
#          XGBoost train/eval for next-day forecasting.
# Input  : X (wide DataFrame indexed by 'date'), y (aligned Series)
# Output : dict with {'final_features', 'xgb_model', 'evaluation_metrics', 'test_pred'}
# Deps   : numpy, pandas, tsfresh, xgboost, scikit-learn, matplotlib
# Notes  : No dask/polars; minimal changes to your notebook style.
# ---------------------------------------------------------------------

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters, MinimalFCParameters
from tsfresh.utilities.distribution import (
    ClusterDaskDistributor, 
    MultiprocessingDistributor
)

def tsxg_multiprocessing(
    X, y,
    id='variable',
    sort='date',
    maxtimeshift=7,
    njobs=18,
    fcparameters=EfficientFCParameters(),
    fdrlvl=0.10,
    split_ratio=0.10,               # <-- tunable test size (last %)
    plot=True,
    xgb_params=None
):
    """
    Minimal, Optuna-ready pipeline:
    - Melt -> roll_time_series -> extract_features
    - Merge per-kind features (inner)
    - tsfresh select_features
    - XGBoost fit (train-only eval_set to mirror your style)
    """
    # stack long
    stacked = X.reset_index().melt(id_vars=sort, var_name='variable', value_name='value')

    # roll windows
    rolled = roll_time_series(
        stacked,
        column_id=id,
        column_sort=sort,
        max_timeshift=maxtimeshift,
        n_jobs=njobs
    ).dropna()

    # extract features
    features_raw = extract_features(
        rolled,
        column_id='id',
        column_sort=sort,
        column_kind=id,
        column_value='value',
        default_fc_parameters=fcparameters,
        n_jobs=njobs
    )

    # merge by kind key
    count = 0
    for key in features_raw.index.levels[0]:
        if count == 0:
            feats = features_raw.loc[key].dropna(axis=1)
        else:
            feats = feats.merge(features_raw.loc[key].dropna(axis=1),
                                left_index=True, right_index=True, how='inner')
        count += 1

    # align + split
    feats.join(X, how='left')
    Xf = feats.loc[y.index]
    test_n = max(1, int(len(Xf) * split_ratio))
    split = len(Xf) - test_n

    X_train = Xf.iloc[:split]
    y_train = y.iloc[:split]
    X_test  = Xf.iloc[split:]
    y_test  = y.iloc[split:]

    # feature selection
    X_train_filtered = select_features(
        X_train, y_train,
        hypotheses_independent=False,
        ml_task='regression',
        n_jobs=njobs,
        fdr_level=fdrlvl
    )

    # model params (minimal; you can pass from Optuna)
    if xgb_params is None:
        xgb_params = dict(
            max_depth=5,
            learning_rate=0.1,        # ('eta' also accepted)
            min_child_weight=3.0,
            subsample=0.9,
            colsample_bytree=0.9,
            n_estimators=600,
            objective='reg:squarederror',
            tree_method='hist',
            eval_metric='mae',
            random_state=42,
            verbosity=0,
        )
    model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=10)

    eval_model = model.fit(
        X_train_filtered, y_train,
        eval_set=[(X_train_filtered, y_train)],
        verbose=False
    )

    preds = model.predict(X_test[X_train_filtered.columns])
    realized = y_test

    # metrics
    mae_model = np.mean(np.abs(realized - preds))
    naive = realized.shift(1)
    mae_naive = np.mean(np.abs(realized[1:] - naive[1:]))
    mase = mae_model / mae_naive if (mae_naive is not None and not np.isnan(mae_naive)) else np.nan

    metrics = {
        "Best Score": getattr(eval_model, "best_score", np.nan),
        "Best Score / Median": (getattr(eval_model, "best_score", np.nan) /
                                np.median(np.abs(y_train)) if len(y_train) else np.nan),
        "MASE": mase,
        "R^2": r2_score(realized, preds) if len(realized) else np.nan
    }

    if plot:
        plt.figure(figsize=(14, 4))
        plt.plot(realized.index, realized, label='Realized')
        plt.plot(realized.index, preds, label='Predicted')
        plt.title('Realized vs Predicted')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    return {
        'final_features': X_train_filtered,
        'xgb_model': model,
        'evaluation_metrics': metrics,
        'test_pred': pd.Series(preds, index=realized.index, name='y_hat')
    }

def tsxg_dask_distributor(
    X, y, scheduler_address,
    id='variable',
    sort='date',
    maxtimeshift=7,
    fcparameters=EfficientFCParameters(),
    fdrlvl=0.10,
    split_ratio=0.10,               # <-- tunable test size (last %)
    plot=True,
    xgb_params=None
):
    """
    Minimal, Optuna-ready pipeline:
    - Melt -> roll_time_series -> extract_features
    - Merge per-kind features (inner)
    - tsfresh select_features
    - XGBoost fit (train-only eval_set to mirror your style)
    """
    # stack long
    stacked = X.reset_index().melt(id_vars=sort, var_name='variable', value_name='value')

    # roll windows
    rolled = roll_time_series(
        stacked,
        column_id=id,
        column_sort=sort,
        max_timeshift=maxtimeshift,
        distributor=ClusterDaskDistributor(scheduler_address)
    ).dropna()

    # extract features
    features_raw = extract_features(
        rolled,
        column_id='id',
        column_sort=sort,
        column_kind=id,
        column_value='value',
        default_fc_parameters=fcparameters,
        distributor=ClusterDaskDistributor(scheduler_address)
    )

    # merge by kind key
    count = 0
    for key in features_raw.index.levels[0]:
        if count == 0:
            feats = features_raw.loc[key].dropna(axis=1)
        else:
            feats = feats.merge(features_raw.loc[key].dropna(axis=1),
                                left_index=True, right_index=True, how='inner')
        count += 1

    # align + split
    Xf = feats.loc[y.index]
    test_n = max(1, int(len(Xf) * split_ratio))
    split = len(Xf) - test_n

    X_train = Xf.iloc[:split]
    y_train = y.iloc[:split]
    X_test  = Xf.iloc[split:]
    y_test  = y.iloc[split:]

    # feature selection
    X_train_filtered = select_features(
        X_train, y_train,
        hypotheses_independent=False,
        ml_task='regression',
        n_jobs= 18,
        fdr_level=fdrlvl
    )

    # model params (minimal; you can pass from Optuna)
    if xgb_params is None:
        xgb_params = dict(
            max_depth=5,
            learning_rate=0.1,        # ('eta' also accepted)
            min_child_weight=3.0,
            subsample=0.9,
            colsample_bytree=0.9,
            n_estimators=600,
            objective='reg:squarederror',
            tree_method='hist',
            eval_metric='mae',
            random_state=42,
            verbosity=0,
        )
    model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=10)

    eval_model = model.fit(
        X_train_filtered, y_train,
        eval_set=[(X_train_filtered, y_train)],
        verbose=False
    )

    preds = model.predict(X_test[X_train_filtered.columns])
    realized = y_test

    # metrics
    mae_model = np.mean(np.abs(realized - preds))
    naive = realized.shift(1)
    mae_naive = np.mean(np.abs(realized[1:] - naive[1:]))
    mase = mae_model / mae_naive if (mae_naive is not None and not np.isnan(mae_naive)) else np.nan

    metrics = {
        "Best Score": getattr(eval_model, "best_score", np.nan),
        "Best Score / Median": (getattr(eval_model, "best_score", np.nan) /
                                np.median(np.abs(y_train)) if len(y_train) else np.nan),
        "MASE": mase,
        "R^2": r2_score(realized, preds) if len(realized) else np.nan
    }

    if plot:
        plt.figure(figsize=(25, 15))
        plt.plot(realized.index, realized, label='Realized')
        plt.plot(realized.index, preds, label='Predicted')
        plt.title('Realized vs Predicted')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    return {
        'final_features': X_train_filtered,
        'xgb_model': model,
        'evaluation_metrics': metrics,
        'test_pred': pd.Series(preds, index=realized.index, name='y_hat')
    }

# --------- Optuna helper (minimal) ---------
# Lets you tune maxtimeshift, fdrlvl, and XGB hyperparams.

def make_optuna_objective(X, y, scheduler_address=None, split_ratio=0.10, njobs=18, plot=False):
    """
    Returns an Optuna objective(trial) that calls tsxg and minimizes MASE.
    """
    def objective(trial):
        maxtimeshift = trial.suggest_int("maxtimeshift", 1, 62)
        fdrlvl       = trial.suggest_float("fdrlvl", 0.02, 0.50)

        xgb_params = dict(
            max_depth          = trial.suggest_int("max_depth", 3, 20),
            learning_rate      = trial.suggest_float("learning_rate", 0.01, 0.9, log=True),
            min_child_weight   = trial.suggest_float("min_child_weight", 1.0, 10.0),
            subsample          = trial.suggest_float("subsample", 0.1, 1.0),
            colsample_bytree   = trial.suggest_float("colsample_bytree", 0.1, 1.0),
            n_estimators       = trial.suggest_int("n_estimators", 10, 3000),
            objective          = "reg:squarederror",
            tree_method        = "hist",
            eval_metric        = "mae",
            verbosity          = 0,
        )

        if scheduler_address is None:
            res = tsxg_multiprocessing(
                X, y,
                maxtimeshift=maxtimeshift,
                njobs=njobs,
                fdrlvl=fdrlvl,
                split_ratio=split_ratio,
                plot=plot,
                xgb_params=xgb_params
            )
        else:
            res = tsxg_dask_distributor(
                X, y,
                maxtimeshift=maxtimeshift,
                scheduler_address=scheduler_address,
                fdrlvl=fdrlvl,
                split_ratio=split_ratio,
                plot=plot,
                xgb_params=xgb_params
            )
        mase = res["evaluation_metrics"]["MASE"]
        return mase if mase == mase else 1e9  # handle NaN -> large penalty

    return objective
