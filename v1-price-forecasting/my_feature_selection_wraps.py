from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import select_features
import datetime as dt
import pandas as pd
import numpy as np
import dask.dataframe as dd

def SF_Pandas_v1(features, , timeframe=1, forecast_periods=1, p_value=0.05, target='LR'):
    # Feature selection on Pandas dataframe with Tsfresh- even if TSFresh doesn't recognise any features as relevant still returns top 100 by p-value
    eth_p = features['prices ethereum']
    features['LR']= np.log(eth_p)- np.log(eth_p.shift(1))
    features['vol']= features['LR'].map(abs)
    features['y_future']= features[target].shift(-forecast_periods)
    features[['LR','y_future','prices ethereum']].tail(30).plot(subplots=True)
    Filtdf= features.dropna(subset=['y_future','LR','vol']).dropna(thresh=drop_thresh*len(features), axis=1).ffill().dropna()
    X= Filtdf.drop('y_future', axis=1)
    y= Filtdf['y_future']
    selected_features= select_features(X, y, hypotheses_independent=False, n_jobs=20, fdr_lvl=p_value).columns.tolist()
    selected_features.append(target)
    selected_features.append('y_future')
    if target != 'prices ethereum': selected_features.append('prices ethereum')
    if target != 'LR': selected_features.append('LR')
    if target != 'vol':selected_features.append('vol')
    if len (selected_features) <= 5: print('not_enough_sig_feat')
    else: print('enough_sig_feat')
    final_features = Filtdf[selected_features]
    return Filtdf

def SF_Dask(features, timeframe=1, forecast_periods=1, p_value=0.05, target='LR'):
    # Feature Selection on Dask Dataframe with TsFresh
    target= target
    eth_p = features['prices ethereum']
    features['LR']= np.log(eth_p)- np.log(eth_p.shift(1))
    features['vol']= abs(features['LR'])
    features['y_future']= features[target].shift(-forecast_periods)
    features[['LR','y_future','prices ethereum']].compute().plot(subplots=True, figsize= (25, 25))
    FC= features.dropna(subset='y_future').dropna(subset='LR').dropna(subset='vol')
    FC = FC.map_partitions(impute).persist()
    func = lambda x: select_features(x.drop('y_future', axis=1), x['y_future'], fdr_level=p_value, hypotheses_independent=False)
    selected_features = FC.map_partitions(func, meta=FC, enforce_metadata=False).compute().columns.tolist()
    selected_features.append(target)
    selected_features.append('y_future')
    if target != 'prices ethereum': selected_features.append('prices ethereum')
    if target != 'LR': selected_features.append('LR')
    if target != 'vol':selected_features.append('vol')
    if len (selected_features) <= 5: print('not_enough_sig_feat')
    else: print('enough_sig_feat')
    final_features = FC[selected_features]
    return final_features

