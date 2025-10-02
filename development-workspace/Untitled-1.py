{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Crypto Volatility Forecasting Example\n",
    "\n",
    "Complete example showing how to use the crypto volatility forecasting toolkit."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import the main class\n",
    "from crypto_volatility_toolkit import CryptoVolatilityAPI\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Set up plotting\n",
    "plt.rcParams['figure.figsize'] = (12, 6)\n",
    "plt.style.use('seaborn')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Initialize the API"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize with custom parameters\n",
    "api = CryptoVolatilityAPI(\n",
    "    target_coin=\"ethereum\",  # Target coin for volatility prediction\n",
    "    top_n=15,              # Top 15 cryptocurrencies by market cap\n",
    "    lookback_days=365*3,   # 3 years of historical data\n",
    "    timezone=\"Europe/Madrid\"\n",
    ")\n",
    "\n",
    "print(f\"Target coin: {api.TARGET_COIN}\")\n",
    "print(f\"Lookback period: {api.LOOKBACK_DAYS} days\")\n",
    "print(f\"Start date: {api.START_DATE}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Data Collection Examples"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get top cryptocurrencies\n",
    "top_coins = api.coingecko_get_universe_v2(n=10, output_format=\"both\")\n",
    "print(\"Top 10 cryptocurrencies:\")\n",
    "for i, (id, symbol) in enumerate(zip(top_coins['ids'], top_coins['ticker'])):\n",
    "    print(f\"{i+1:2d}. {symbol:6s} ({id})\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get extended historical data for Bitcoin\n",
    "btc_data = api.coingecko_get_historical_paginated(\n",
    "    coin_id=\"bitcoin\", \n",
    "    max_days=1000,  # ~3 years\n",
    "    step_days=90    # 90 days per request\n",
    ")\n",
    "\n",
    "print(f\"Bitcoin data shape: {btc_data.shape}\")\n",
    "print(f\"Columns: {list(btc_data.columns)}\")\n",
    "btc_data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot Bitcoin price and volume\n",
    "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))\n",
    "\n",
    "# Price\n",
    "ax1.plot(btc_data.index, btc_data['prices_bitcoin'], linewidth=1.5)\n",
    "ax1.set_title('Bitcoin Price (USD)', fontsize=14, fontweight='bold')\n",
    "ax1.set_ylabel('Price ($)')\n",
    "ax1.grid(True, alpha=0.3)\n",
    "\n",
    "# Volume\n",
    "ax2.plot(btc_data.index, btc_data['total_volumes_bitcoin'], linewidth=1.5, color='orange')\n",
    "ax2.set_title('Bitcoin Trading Volume', fontsize=14, fontweight='bold')\n",
    "ax2.set_ylabel('Volume ($)')\n",
    "ax2.set_xlabel('Date')\n",
    "ax2.grid(True, alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get DVOL (crypto volatility index) data\n",
    "dvol_data = api.deribit_get_dvol(['BTC', 'ETH'], days=365)\n",
    "print(f\"DVOL data shape: {dvol_data.shape}\")\n",
    "dvol_data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot DVOL data\n",
    "fig, ax = plt.subplots(figsize=(15, 6))\n",
    "ax.plot(dvol_data.index, dvol_data['dvol_btc'], label='BTC DVOL', linewidth=1.5)\n",
    "ax.plot(dvol_data.index, dvol_data['dvol_eth'], label='ETH DVOL', linewidth=1.5)\n",
    "ax.set_title('Crypto Volatility Index (DVOL)', fontsize=14, fontweight='bold')\n",
    "ax.set_ylabel('Volatility (%)')\n",
    "ax.set_xlabel('Date')\n",
    "ax.legend()\n",
    "ax.grid(True, alpha=0.3)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get macroeconomic data\n",
    "macro_data = api.fred_get_series()\n",
    "print(f\"Macro data shape: {macro_data.shape}\")\n",
    "print(f\"Columns: {list(macro_data.columns)}\")\n",
    "macro_data.tail(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Complete Data Collection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Collect all data sources\n",
    "print(\"Collecting all data sources...\")\n",
    "unified_data = api.collect_all_data()\n",
    "\n",
    "print(f\"\\nUnified dataset shape: {unified_data.shape}\")\n",
    "print(f\"Date range: {unified_data.index.min()} to {unified_data.index.max()}\")\n",
    "print(f\"Number of features: {len(unified_data.columns)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Display data categories\n",
    "price_cols = [col for col in unified_data.columns if 'prices_' in col]\n",
    "volume_cols = [col for col in unified_data.columns if 'volume' in col.lower()]\n",
    "macro_cols = [col for col in unified_data.columns if any(x in col for x in ['vix', 'yield', 'usd'])]\n",
    "dvol_cols = [col for col in unified_data.columns if 'dvol' in col]\n",
    "\n",
    "print(f\"Price columns ({len(price_cols)}): {price_cols[:5]}...\")\n",
    "print(f\"Volume columns ({len(volume_cols)}): {volume_cols[:5]}...\")\n",
    "print(f\"Macro columns ({len(macro_cols)}): {macro_cols}\")\n",
    "print(f\"DVOL columns ({len(dvol_cols)}): {dvol_cols}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Feature Engineering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compute technical indicators\n",
    "print(\"Computing technical indicators...\")\n",
    "ta_features = api.compute_ta_indicators(unified_data, price_prefix=\"prices_\")\n",
    "\n",
    "print(f\"Technical indicators shape: {ta_features.shape}\")\n",
    "print(f\"Sample TA features: {list(ta_features.columns[:10])}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Combine original data with technical indicators\n",
    "full_dataset = unified_data.join(ta_features, how='left').dropna()\n",
    "print(f\"Full dataset shape: {full_dataset.shape}\")\n",
    "\n",
    "# Create target variable (realized volatility)\n",
    "target_coin = api.TARGET_COIN\n",
    "full_dataset[f'log_returns_{target_coin}'] = (np.log(full_dataset[f'prices_{target_coin}']) - \n",
    "                                              np.log(full_dataset[f'prices_{target_coin}'].shift(1)))\n",
    "full_dataset[f'realized_vol_{target_coin}'] = abs(full_dataset[f'log_returns_{target_coin}'])\n",
    "\n",
    "# Prepare features and target\n",
    "X = full_dataset.diff().dropna()\n",
    "y = X[f'realized_vol_{target_coin}'].shift(-1).dropna()\n",
    "\n",
    "print(f\"Final feature matrix: {X.shape}\")\n",
    "print(f\"Target variable: {y.shape}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot target variable (realized volatility)\n",
    "fig, ax = plt.subplots(figsize=(15, 6))\n",
    "ax.plot(y.index, y.values, linewidth=1.5, alpha=0.8)\n",
    "ax.set_title(f'{target_coin.title()} Realized Volatility (Target Variable)', \n",
    "             fontsize=14, fontweight='bold')\n",
    "ax.set_ylabel('Realized Volatility')\n",
    "ax.set_xlabel('Date')\n",
    "ax.grid(True, alpha=0.3)\n",
    "\n",
    "# Add rolling mean\n",
    "rolling_mean = y.rolling(window=30).mean()\n",
    "ax.plot(rolling_mean.index, rolling_mean.values, \n",
    "        color='red', linewidth=2, label='30-day Moving Average')\n",
    "ax.legend()\n",
    "\n",
    "plt.show()\n",
    "\n",
    "print(f\"Volatility statistics:\")\n",
    "print(f\"Mean: {y.mean():.6f}\")\n",
    "print(f\"Std: {y.std():.6f}\")\n",
    "print(f\"Min: {y.min():.6f}\")\n",
    "print(f\"Max: {y.max():.6f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Model Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train model with default parameters\n",
    "print(\"Training volatility forecasting model...\")\n",
    "results = api.tsxg_multiprocessing(X, y, plot=True)\n",
    "\n",
    "print(\"\\nModel Performance:\")\n",
    "for metric, value in results['evaluation_metrics'].items():\n",
    "    print(f\"{metric}: {value:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature importance analysis\n",
    "model = results['xgb_model']\n",
    "feature_importance = pd.DataFrame({\n",
    "    'feature': results['final_features'].columns,\n",
    "    'importance': model.feature_importances_\n",
    "}).sort_values('importance', ascending=False)\n",
    "\n",
    "print(\"Top 20 Most Important Features:\")\n",
    "print(feature_importance.head(20))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot feature importance\n",
    "fig, ax = plt.subplots(figsize=(12, 8))\n",
    "top_features = feature_importance.head(15)\n",
    "ax.barh(range(len(top_features)), top_features['importance'])\n",
    "ax.set_yticks(range(len(top_features)))\n",
    "ax.set_yticklabels(top_features['feature'], fontsize=10)\n",
    "ax.set_xlabel('Feature Importance')\n",
    "ax.set_title('Top 15 Most Important Features', fontsize=14, fontweight='bold')\n",
    "ax.grid(True, axis='x', alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Hyperparameter Optimization (Optional)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Uncomment to run hyperparameter optimization (takes longer)\n",
    "# print(\"Running hyperparameter optimization...\")\n",
    "# import optuna\n",
    "\n",
    "# study = optuna.create_study(direction='minimize')\n",
    "# objective = api.make_optuna_objective(X, y, plot=False)\n",
    "# study.optimize(objective, n_trials=20)  # Reduce trials for demo\n",
    "\n",
    "# print(\"Best parameters:\", study.best_params)\n",
    "# print(\"Best MASE:\", study.best_value)\n",
    "\n",
    "# # Plot optimization history\n",
    "# optuna.visualization.matplotlib.plot_optimization_history(study)\n",
    "# plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Complete Pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run the complete end-to-end pipeline\n",
    "print(\"Running complete volatility forecasting pipeline...\")\n",
    "\n",
    "# Create a new API instance for demonstration\n",
    "pipeline_api = CryptoVolatilityAPI(\n",
    "    target_coin=\"bitcoin\",  # Try Bitcoin this time\n",
    "    top_n=8,\n",
    "    lookback_days=365*2,\n",
    "    timezone=\"Europe/Madrid\"\n",
    ")\n",
    "\n",
    "# Run pipeline (set optimize_hyperparams=True for production)\n",
    "pipeline_results = pipeline_api.run_volatility_forecast_pipeline(\n",
    "    optimize_hyperparams=False,  # Set to True for hyperparameter tuning\n",
    "    n_trials=10  # Increase for better optimization\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Analyze pipeline results\n",
    "if pipeline_results:\n",
    "    print(\"\\n=== Pipeline Results Summary ===\")\n",
    "    \n",
    "    unified = pipeline_results['unified_data']\n",
    "    model_results = pipeline_results['model_results']\n",
    "    \n",
    "    print(f\"Dataset shape: {unified.shape}\")\n",
    "    print(f\"Features used: {len(model_results['final_features'].columns)}\")\n",
    "    \n",
    "    print(\"\\nModel Performance:\")\n",
    "    for metric, value in model_results['evaluation_metrics'].items():\n",
    "        print(f\"  {metric}: {value:.4f}\")\n",
    "    \n",
    "    # Show prediction vs actual\n",
    "    test_pred = model_results['test_pred']\n",
    "    print(f\"\\nPrediction period: {test_pred.index.min()} to {test_pred.index.max()}\")\n",
    "    print(f\"Number of predictions: {len(test_pred)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Conclusions and Next Steps\n",
    "\n",
    "This notebook demonstrates the complete crypto volatility forecasting toolkit:\n",
    "\n",
    "1. **Data Collection**: Multi-source data from CoinGecko, Binance, Deribit (DVOL), and FRED\n",
    "2. **Feature Engineering**: Technical indicators using TA-Lib\n",
    "3. **Advanced Features**: TSFresh time series feature extraction\n",
    "4. **Modeling**: XGBoost with hyperparameter optimization\n",
    "5. **Evaluation**: Multiple metrics including MASE and R²\n",
    "\n",
    "### Missing Data Sources for Enhanced Accuracy:\n",
    "\n",
    "- **Social sentiment**: Twitter, Reddit, news sentiment\n",
    "- **Options data**: Put/call ratios, implied volatility surfaces\n",
    "- **Futures data**: Contango/backwardation, basis spreads\n",
    "- **Exchange flows**: Whale movements, exchange inflows/outflows\n",
    "- **Network metrics**: Hash rate, active addresses, transaction fees\n",
    "- **Cross-asset correlations**: Stock market, commodities, bonds\n",
    "- **Central bank data**: Interest rate expectations, policy announcements\n",
    "\n",
    "### Potential Improvements:\n",
    "\n",
    "1. **LSTM/Transformer models** for sequence modeling\n",
    "2. **Ensemble methods** combining multiple models\n",
    "3. **Real-time prediction** with streaming data\n",
    "4. **Risk management** integration for DeFi operations\n",
    "5. **Multi-timeframe modeling** (intraday, daily, weekly)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}