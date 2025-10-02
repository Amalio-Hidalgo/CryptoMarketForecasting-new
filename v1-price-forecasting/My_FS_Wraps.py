"""Feature Selection Module for Cryptocurrency Analysis

This module provides feature selection functionality for time series data.
Supports both statistical feature selection and target variable transformations.
"""

# Core data processing
import pandas as pd
import numpy as np
import dask.dataframe as dd

# tsfresh selection
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

# Constants
DEFAULT_TARGET = 'prices ethereum'
DEFAULT_P_VALUE = 0.05
DEFAULT_FORECAST_PERIODS = 1

def SF_Dask(features, p_value=DEFAULT_P_VALUE):
    """Feature selection with single partition for better statistical validity
    Args:
        features (dask.DataFrame): Input features with y_future column
        p_value (float): Feature selection significance level
    Returns:
        tuple: (selected_features, y) - Features and target after selection
    """    
    features = features.repartition(npartitions=1)
    features = features.dropna(subset='y_future')
    features = features.map_partitions(impute)
    y = features['y_future']

    func = lambda x: select_features(
        x.drop('y_future', axis=1), 
        x['y_future'], 
        fdr_level=p_value, 
        hypotheses_independent=False, 
        ml_task='regression',
        chunksize=None,
        n_jobs=1  
    )
    
    selected = features.map_partitions(
        func, 
        meta=features,
        enforce_metadata=False
    )
    
    return selected, y

def SF_Pandas(features, forecast_periods=DEFAULT_FORECAST_PERIODS, 
                 p_value=DEFAULT_P_VALUE, target=DEFAULT_TARGET, 
                 LR=False, Vol=False, cores=20):
    """Feature selection on Pandas DataFrame using tsfresh"""
    # Calculate transformations
    target_var = features[target]
    features['LR(%)'] = np.log(target_var) - np.log(target_var.shift(1))
    features['LR(%)'] = features['LR(%)'] * 100
    features['1PeriodVol(%)'] = abs(features['LR(%)'])
    
    # Set target variable
    if LR == True:
        target_var = features['LR(%)']
    elif Vol == True:
        target_var = features['1PeriodVol(%)']
    else:
        target_var = features[target]
        
    features['y_future'] = target_var.shift(-forecast_periods)
    
    try:
        features[['LR(%)', 'y_future', target, '1PeriodVol(%)']].tail(30).plot(
            subplots=True, 
            figsize=(25, 25)
        )
    except:
        pass
    
    # Feature selection 
    FC = features.dropna(subset='y_future')
    FC = impute(FC)
    selected_features = select_features(
        FC.drop('y_future', axis=1),
        FC['y_future'],
        fdr_level=p_value,
        hypotheses_independent=False,
        n_jobs=cores  
    ).columns.tolist()
    
    # Add required columns
    selected_features.extend(['y_future', target, 'LR(%)', '1PeriodVol(%)'])
    selected_features = list(set(selected_features))
    
    print('not_enough_sig_feat' if len(selected_features) <= 5 else 'enough_sig_feat')
    
    return FC[selected_features]