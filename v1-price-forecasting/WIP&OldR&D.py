# def SF_Dask_V2(features, fdr_level, timeframe=1, forecast_periods=2, target='prices ethereum'):
#     # FEATURE SELECTION ON DASK- TSFRESH
#     target= target
#     # shift_start= features.index.compute()[-1]
#     # shift_start= pd.to_datetime(shift_start).tz_localize('CET')
#     # if timeframe ==1: 
#     #     shift_end = pd.Timestamp(dt.datetime.now(), tz='CET') + pd.Timedelta(minutes=5*periods*forecast_periods)
#     #     shift = int(round((shift_end-shift_start).seconds / (60*5*periods)))
#     #     fcast= shift*5*periods
#     #     freq= f'{fcast}min'
#     # else: 
#     #     shift_end = pd.Timestamp(dt.datetime.now(), tz='CET') + pd.Timedelta(hours=periods)
#     #     shift = int(round((shift_end-shift_start).seconds / (60*60*periods)))
#     #     fcast= shift*periods
#     #     freq= f'{shift}h'

#     # print(f"forecasting: {fcast} minutes ahead")
#     # print(f"shift: {shift}")
#     eth_p = features['prices ethereum']
#     features['LR']= np.log(eth_p)- np.log(eth_p.shift(1))
#     features['vol']= abs(features['LR'])
#     features['y_future']= features[target].shift(-forecast_periods)
#     features[['LR','y_future','prices ethereum']].compute().plot(subplots=True, figsize= (25, 25))
#     FC= features.dropna(subset='y_future').dropna(subset='LR').dropna(subset='vol')
#     FC = FC.map_partitions(impute).persist()
#     func = lambda x: select_features(x.drop('y_future', axis=1), x['y_future'], fdr_level=fdr_level, n_jobs=1, hypotheses_independent=False)
#     selected_features = FC.map_partitions(func, meta=FC, enforce_metadata=False).compute().columns.tolist()
#     selected_features.append(target)
#     selected_features.append('y_future')
#     if target != 'prices ethereum': selected_features.append('prices ethereum')
#     if target != 'LR': selected_features.append('LR')
#     if target != 'vol':selected_features.append('vol')
#     if len (selected_features) <= 5: print('not_enough_sig_feat')
#     else: print('enough_sig_feat')
#     final_features = FC[selected_features]
#     return final_features

# def SF_Pandas_v2(features, fdr_lvl=0.05, timeframe=1, forecast_periods=1, p_value=None, topNsigfeat=100, drop_thresh=0.85, target='LR'):
#     # Feature selection on Pandas dataframe with Tsfresh- even if TSFresh doesn't recognise any features as relevant still returns top 100 by p-value
#     shift_start= features.index[-1]
#     if timeframe <=1: 
#         shift_end = pd.Timestamp(dt.datetime.now(), tz='CET') + pd.Timedelta(minutes=5*forecast_periods)
#         shift = int(round((shift_end-shift_start).seconds / (60*5*forecast_periods)))
#         fcast= shift*5
#         freq= f'{fcast}min'
#     else: 
#         shift_end = pd.Timestamp(dt.datetime.now(), tz='CET') + pd.Timedelta(hours=forecast_periods)
#         shift = int(round((shift_end-shift_start).seconds / (60*60*forecast_periods)))
#         fcast= shift*60
#         freq= f'{shift}h'
#     if shift > forecast_periods: 
#         print(f"too_slow, forecast: {fcast} minutes ahead")
#     else:
#         print(f'forecasting {fcast} minutes ahead')
#     eth_p = features['prices ethereum']
#     features['LR']= np.log(eth_p)- np.log(eth_p.shift(1))
#     features['vol']= features['LR'].map(abs)
#     features['y_future']= features[target].shift(-shift)
#     features[['LR','y_future','prices ethereum']].tail(30).plot(subplots=True)
#     Filtdf= features.dropna(subset=['y_future','LR','vol']).dropna(thresh=drop_thresh*len(features), axis=1).ffill().dropna()
#     X= Filtdf.drop('y_future', axis=1)
#     y= Filtdf['y_future']
#     relevance_table = calculate_relevance_table(X, y,hypotheses_independent=False, fdr_level=fdr_lvl)
#     if p_value is not None: rel_feat= relevance_table[relevance_table['p_value']<=p_value].index.values.tolist()
#     else: rel_feat= relevance_table.dropna().sort_values(by='p_value', ascending=True).head(topNsigfeat).index
#     X = X[rel_feat]
#     selected= select_features(X, y, hypotheses_independent=False, n_jobs=20)
#     if len(selected.columns) >= 8: X = selected
#     return X, y, shift, freq

# Functions to support paralellization as per TsFresh docs by setting Environment variables to single thread 
# def set_single_thread():
#     os.environ['OMP_NUM_THREADS'] = '1'
#     os.environ['MKL_NUM_THREADS'] = '1'
#     os.environ['OPENBLAS_NUM_THREADS'] = '1'
# # Function to revert environment variables to their default state
# def revert_threads():
#     os.environ.pop('OMP_NUM_THREADS', None)
#     os.environ.pop('MKL_NUM_THREADS', None)
#     os.environ.pop('OPENBLAS_NUM_THREADS', None)
