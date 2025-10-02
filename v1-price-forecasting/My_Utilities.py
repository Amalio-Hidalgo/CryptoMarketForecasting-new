import numpy as np
import pandas as pd
import datetime as dt
import dask.dataframe as dd
from tsfresh.utilities.dataframe_functions import impute

def Dask_Target_Constructor(features, target='prices ethereum', forecast_periods=1, LR=False, Vol=False, PctChange=True, Diff=False, splits=5, impute=True):
    if LR== True: features['y_future']= np.log(features[target])- np.log(features[target].shift(1)).shift(-forecast_periods)
    elif Vol== True: features['y_future']= abs(np.log(features[target])- np.log(features[target].shift(1))).shift(-forecast_periods)
    elif PctChange== True: features['y_future']= features[target].pct_change().shift(-forecast_periods)
    elif Diff== True: features['y_future']= features[target].diff().shift(-forecast_periods)
    else: features['y_future']= features[target].shift(-forecast_periods)
    features= features.dropna(subset=['y_future'])
    if impute== True: features= features.map_partitions(impute)
    return features