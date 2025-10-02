"""Feature Engineering Module for Cryptocurrency Analysis"""

# Core data processing
import pandas as pd
import numpy as np
import dask.dataframe as dd

# tsfresh feature extraction
from tsfresh import extract_features
from tsfresh.feature_extraction import (
    MinimalFCParameters, 
    EfficientFCParameters, 
    ComprehensiveFCParameters
)

# tsfresh utilities
from tsfresh.utilities.distribution import (
    ClusterDaskDistributor, 
    MultiprocessingDistributor
)
from tsfresh.utilities.dataframe_functions import roll_time_series, impute
from tsfresh.convenience.bindings import dask_feature_extraction_on_chunk

# Constants
DEFAULT_TARGET = 'prices ethereum'
DEFAULT_FORECAST_PERIODS = 1
DEFAULT_PARAMETER_COMPLEXITY = 1

def EF_Dask(df, ParameterComplexity=1, target='prices ethereum', LR=False, Vol=False, forecast_periods=1):
    """Feature extraction for Dask DataFrame
    Args:
        df (dask.DataFrame): Input DataFrame in long format
        ParameterComplexity (int): Feature complexity level
        target (str): Target variable name
        LR (bool): If True, use log returns as target
        Vol (bool): If True, use volatility as target
        forecast_periods (int): Number of periods to forecast
    Returns:
        dask.DataFrame: Features DataFrame including y_future column
    """    
    # Set feature complexity
    if ParameterComplexity == 0:
        FC_parameters = MinimalFCParameters()
    elif ParameterComplexity == 1:
        FC_parameters = EfficientFCParameters()
    else:
        FC_parameters = ComprehensiveFCParameters()
            
    # Define metadata for optimization
    meta = {'datetime': 'string', 'variable': 'string', 'value': 'float64', 'id': 'string'}
    # Rolling operation
    func = lambda x: roll_time_series(x, column_id='variable', column_sort='datetime', n_jobs=1, chunksize=None)
    rolled = df.map_partitions(func, meta=meta).dropna()   
    rolled = rolled.astype(meta)
    # Extract dates and create concat
    dates = rolled[['datetime','id']].groupby('id').last()
    concat = rolled[['variable','value', 'datetime','id']].groupby('id').last().reset_index(drop=False)
    # Feature extraction
    rolled_grouped = rolled.groupby(['id','variable'])
    features = dask_feature_extraction_on_chunk(
        rolled_grouped,
        column_id='id',
        column_sort='datetime',
        column_kind='variable',
        column_value='value',
        default_fc_parameters=FC_parameters
    )
    # Transform features
    features = (features
        .reset_index(drop=True)
        .astype({'variable': 'string', 'value': 'float64', 'id': 'string'})
        .join(dates, how='left', on='id')
        .pipe(lambda x: dd.concat([x, concat], axis=0))
        .drop('id', axis=1)
        .categorize('variable')
        .pivot_table(index='datetime', columns='variable', values='value'))
    # Calculate target variables only if needed
    if LR == True:
        features['LR'] = np.log(features[target]) - np.log(features[target].shift(1))
        target_var = features['LR']
    elif Vol == True:
        features['LR'] = np.log(features[target]) - np.log(features[target].shift(1))
        features['1PeriodVol'] = abs(features['LR'])
        target_var = features['1PeriodVol']
    else:
        target_var = features[target]
            
    features['y_future'] = target_var.shift(-forecast_periods)
    
    return features  
    

def EF_Pandas_MultiprocessingDistributor(df, ParameterComplexity=1):
    """Feature extraction using local multiprocessing for Pandas DataFrame
    Args:
        df (pandas.DataFrame): Input DataFrame in long format
        ParameterComplexity (int): Feature complexity level
    Returns:
        pandas.DataFrame: Features DataFrame
    """
    if ParameterComplexity ==0: FC_parameters= MinimalFCParameters()
    elif ParameterComplexity ==1: FC_parameters= EfficientFCParameters()
    else: FC_parameters= ComprehensiveFCParameters()
    feature_extrac_params= {'column_id':'id', 'column_kind':'variable', 'column_sort':'datetime',
                            'column_value':'value','default_fc_parameters': FC_parameters}
    rolled = roll_time_series(df, column_id='variable', column_sort='datetime', n_jobs=20).ffill()
    dates= rolled.groupby('id').last()['datetime']
    concat= rolled.groupby('id').last().reset_index(drop=False)
    raw_features = extract_features(rolled, **feature_extrac_params, n_jobs=20, pivot=False)
    features= pd.DataFrame(raw_features, columns=['id','variable','value'])
    features= features.join(dates, how='left', on='id')
    features= pd.concat([features,concat], axis=0)
    features= features.pivot_table(index="datetime", columns="variable", values="value")
    features = features.apply(pd.to_numeric, errors= 'ignore')
    return features
                                                        

def EF_Pandas_DaskDistributor(df, scheduler_address, ParameterComplexity=1):
    """Distributed feature extraction using Dask scheduler for Pandas DataFrame
    Args:
        df (pandas.DataFrame): Input DataFrame in long format
        scheduler_address (str): Dask scheduler address
        ParameterComplexity (int): Feature complexity level
    Returns:
        pandas.DataFrame: Features DataFrame
    """
    if ParameterComplexity ==0: FC_parameters= MinimalFCParameters()
    elif ParameterComplexity ==1: FC_parameters= EfficientFCParameters()
    else: FC_parameters= ComprehensiveFCParameters()
    feature_extrac_params= {'column_id':'id', 'column_kind':'variable', 'column_sort':'datetime',
                        'column_value':'value','default_fc_parameters': FC_parameters}
    rolled=roll_time_series(df, distributor=ClusterDaskDistributor(scheduler_address), column_id='variable', column_sort='datetime').ffill()
    dates= rolled.groupby('id').last()['datetime']
    concat= rolled.groupby('id').last().reset_index(drop=False)
    raw_features= extract_features(rolled,distributor=ClusterDaskDistributor(scheduler_address), **feature_extrac_params, pivot=False)
    features= pd.DataFrame(raw_features, columns=['id','variable','value'])
    features= features.join(dates, how='left', on='id')
    features= pd.concat([features,concat], axis=0)
    features= features.pivot_table(index="datetime", columns="variable", values="value")
    features = features.apply(pd.to_numeric, errors= 'ignore')
    return features
