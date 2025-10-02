model_parameters={
    'num_parallel_tree': [1,3,5,10,20],
    'learning_rate': [0.01,0.01,0.05,0.1,0.5,1],
    'max_depth': [3,6,12,24,48],
    'gamma':[0,0.01,0.05,0.1,0.5,1,5,25],
    'min_child_weight':[0.5,1,3,5],
    'subsample':[1,0.05, 0.1,0.25,0.5],
    'sampling_method':['uniform', 'gradient_based'],
    'colsample_bytree':[1, 0.1, 0.5, 1],
    'grow_policy':['depthwise', 'lossguide']
    }
def my_randomgridsearchcv_xgboost_machine_learning_wrapper(selected_features, param_grid, number_cvs=5, model_parameters=model_parameters):
# GPU Accelerated XGBoost Using Random Gridsearch for Hyperparameter Tuning 
    tscv = TimeSeriesSplit(n_splits=number_cvs)
    X_train_cp = xgb.DMatrix(X_train)
    X_test_cp = cp.array(X_test) 
    y_train_cp = cp.array(y_train) 
    y_test_cp = cp.array(y_test)
    basemodel= xgb.XGBRegressor(eval_metric='mae', early_stopping_rounds=25, 
                            device= 'cuda',tree_method='hist', n_jobs=1, verbose=0)
    rgs= RandomizedSearchCV(estimator=basemodel, verbose= 25, n_jobs=1,
                            cv=tscv,param_distributions=model_parameters, n_iter=20, refit=True).fit()
    
def Optuna_Dask_Xgboost_HyperparameterTuning(selected_features, param_grid, number_cvs=5, model_parameters=model_parameters):