# Imports
import pandas as pd
import datetime as dt
from tsfresh import extract_features, extract_relevant_features
from tsfresh.feature_extraction import EfficientFCParameters, ComprehensiveFCParameters, MinimalFCParameters
from tsfresh.utilities.dataframe_functions  import roll_time_series
import numpy as np
from tsfresh.utilities.dataframe_functions import impute
import dask.dataframe as dd
import ctypes
import os
from tsfresh.utilities.distribution import ClusterDaskDistributor
from tsfresh.convenience.bindings import dask_feature_extraction_on_chunk

# Functions to support paralellization as per TsFresh docs by setting Environment variables to single thread 
def set_single_thread():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
# Function to revert environment variables to their default state
def revert_threads():
    os.environ.pop('OMP_NUM_THREADS', None)
    os.environ.pop('MKL_NUM_THREADS', None)
    os.environ.pop('OPENBLAS_NUM_THREADS', None)

# All Feature calculating wraps take a melted dataframe as input and return a pivoted dataframe with variables as columns and a datetime index

# Feature extraction distributing roll of pandas df with dask
def EF_Pandas_DaskDistributor(df, scheduler_address,
                           data_roll_params= {'column_id':'variable', 'column_sort':'datetime', 'chunksize': None},
                           feature_extrac_params= {'column_id':'id', 'column_kind':'variable', 'column_sort':'datetime',
                                                    'column_value':'value','default_fc_parameters': EfficientFCParameters()}):
    roll=roll_time_series(df, distributor=ClusterDaskDistributor(scheduler_address),**data_roll_params)
    Melted_features= extract_features(roll,distributor=ClusterDaskDistributor(scheduler_address), **feature_extrac_params)
    # Melted_features = Melted_features.reset_index(drop=False)
    # Melted_features= Melted_features.join(roll.groupby('id').last()['datetime'], how='left', on='id')
    # Melted_features= pd.concat([Melted_features,df], axis=0)
    # flat_features= Melted_features.pivot_table(index="datetime", columns="variable", values="value")
    return Melted_features

def EF_Dask(df, top_coins, timeframe=1):
    meta= {'datetime': 'string', 'variable': 'string', 'value': 'float', 'id': 'string'}
    func=  lambda x: roll_time_series(x, column_id='variable', column_sort='datetime', n_jobs=1).ffill()
    rolled= df.map_partitions(func, meta = meta)
    rolled= rolled.astype(meta)
    rolled_grouped = rolled.groupby(['id','variable'])
    features= dask_feature_extraction_on_chunk(rolled_grouped, column_id='id', column_sort='datetime',column_kind='variable', column_value='value').reset_index(drop=True)
    features= features.set_index('id')
    dates =rolled[['datetime','id']].groupby('id').last()
    concat = rolled[['variable','value', 'datetime','id']].groupby('id').last().reset_index(drop=True)
    features= features.join(dates, how='left').reset_index(drop=True)
    # features.dropna(subset)
    output= dd.concat([features, concat], axis=0)
    output=output.categorize('variable')
    output=output.pivot_table(index='datetime', columns= 'variable', values='value')
    return output
# 
def EF_Pandas_MultiprocessingDistributor(df, scheduler_address,data_roll_params= {'column_id':'variable', 'column_sort':'datetime'},
                                              feature_extrac_params= {'column_id':'id', 'column_kind':'variable', 'column_sort':'datetime',
                                                                      'column_value':'value', 'default_fc_parameters': MinimalFCParameters()}):
    rolled = roll_time_series(df, data_roll_params, n_jobs=20)
    features = extract_features(rolled, feature_extrac_params, n_jobs=20)
    return features
                                                        # 'chunksize': None, 'default_fc_parameters': EfficientFCParameters()})
                                        
# Feature extraction
# def EF_LocalwithDist(df, scheduler_address,
#                                     data_roll_params= {'column_id':'variable', 'column_sort':'datetime'},
#                                     feature_extrac_params= {'column_id':'id', 'column_kind':'variable', 'column_sort':'datetime', 'column_value':'value', 'default_fc_parameters': MinimalFCParameters()}
#                                                         # 'chunksize': None, 'default_fc_parameters': EfficientFCParameters()})

#META CONSTRUCTION IF DONT WANT TO INPUT EXPLICITLY
#  columns= sample.columns.tolist()
# dtypes= sample.dtypes.tolist()
# meta={}
# count=0
# for item in columns:
#     meta[item] = dtypes[count]
#     count=count+1

# columns= sample.columns.tolist()
# meta={}
# for item in columns:
#     if item == 'value': meta[item] = float
#     else: meta[item] = str
#     count=count+1
