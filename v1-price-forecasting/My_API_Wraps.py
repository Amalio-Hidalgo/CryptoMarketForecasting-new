"""Cryptocurrency Data Collection Module

This module provides functions to fetch historical cryptocurrency data.
Supports both Pandas and Dask DataFrame output formats with CoinGecko API.
"""

# Core data processing
import pandas as pd
import dask.dataframe as dd
import numpy as np
import datetime as dt
import requests

# Constants
DEFAULT_TIMEFRAME = 1
DEFAULT_TOP_COINS = 8
DEFAULT_PERIODS = 1
timezone = 'CET'

def CoinGecko_HSPD_Pandas(timeframe=DEFAULT_TIMEFRAME, top_coins=DEFAULT_TOP_COINS, 
                          periods=DEFAULT_PERIODS, api_key=None):
    """Fetch historical crypto data using Pandas
    Args:
        timeframe (int): Historical data period in days
        top_coins (int): Number of top cryptocurrencies to fetch
        periods (int): Resampling frequency multiplier:
            - For timeframe <= 1: periods * 5min intervals
            - For timeframe > 1: periods * 1h intervals
        api_key (str): Optional CoinGecko API key
    Returns:
        pandas.DataFrame: Long format DataFrame with columns [datetime, variable, value, id]
    """
    headers = {
    "accept": "application/json",
    "x-cg-demo-api-key": api_key
    }
    coins = CoinGecko_TopCoinsMC_Pandas(number= top_coins, headers=headers)
    count=0
    for coin in coins:
        response = pd.DataFrame(Coingecko_HSPD_Json(coin=coin, timeframe=timeframe, headers=headers))
        response['datetime'] = response['prices'].map(lambda x: x[0]).apply(pd.to_datetime, unit='ms', utc=True)
        for column in response.columns.drop('datetime'):
            response[column]= response[column].map(lambda x: x[1])
        if timeframe<=1: 
            freq= f'{periods*5}min'
            response = response.set_index('datetime', drop=True).resample(freq).last()
            response.index= response.index.tz_convert(timezone)
        else: 
            freq= f'{periods}h'
            response = response.set_index('datetime', drop=True).resample(freq).last()
            response.index= response.index.tz_convert(timezone)
        if count == 0: 
            output = response
        else: 
            output= output.join(response, rsuffix=' '+ coin, how='left')
        count=count+1
    return output.melt(ignore_index=False).reset_index(drop=False)

def CoinGecko_HSPD_Dask(timeframe=DEFAULT_TIMEFRAME, top_coins=DEFAULT_TOP_COINS, 
                        periods=DEFAULT_PERIODS, api_key=None):
    """Fetch historical crypto data using Dask
    Args:
        timeframe (int): Historical data period in days
        top_coins (int): Number of top cryptocurrencies to fetch
        periods (int): Resampling frequency multiplier:
            - For timeframe <= 1: periods * 5min intervals
            - For timeframe > 1: periods * 1h intervals
        api_key (str): Optional CoinGecko API key
    Returns:
        dask.DataFrame: Long format DataFrame with columns [datetime, variable, value, id]
    """
    headers = {
    "accept": "application/json",
    "x-cg-demo-api-key": api_key
    }
    coins = CoinGecko_TopCoinsMC_Pandas(top_coins, headers=headers)
    count=0
    for coin in coins:
        dd_response= Coingecko_HSPD_Json(coin, headers=headers, timeframe=timeframe)
        dd_response= dd.from_dict(dd_response, npartitions=1)
        dd_response['datetime'] = dd_response['prices'].map(lambda x: float(x.split('[')[1].split(',')[0]), meta=('datetime', 'float64'))
        func = lambda x : pd.to_datetime(x, unit='ms', origin='unix', utc=True)
        dd_response['datetime']= dd_response['datetime'].map(func, meta=('datetime', 'datetime64[ns, UTC]'))
        if timeframe<=1: freq= f'{periods*5}min'
        else: freq= f'{periods}h'
        dd_response= dd_response.set_index('datetime', sorted=False, npartitions=len(dd_response.columns), drop=True).resample(freq).last()
        func= lambda x : float(x.split('[')[1].split(',')[1].split(']')[0])
        for column in dd_response.columns:
            dd_response[column]= dd_response[column].compute().dropna().apply(func)
        if count == 0 : output=dd_response
        else: output= output.join(dd_response, how='left', rsuffix=' '+coin)
        count= count+1
    output= output.reset_index().melt('datetime').sort_values(['variable','datetime'])
    output['datetime']= output['datetime'].dt.tz_convert(timezone)
    # output = output.repartition(npartitions= splits)
    return output

def Coingecko_HSPD_Json(coin, headers, timeframe):
    """Fetch historical price data from CoinGecko API in JSON format
    
    Args:
        coin (str): Cryptocurrency CoinGecko ID (e.g. 'bitcoin', 'ethereum')
        headers (dict): API request headers with authentication
        timeframe (int): Historical data period in days
        
    Note:
        CoinGecko limits:
        - 5min granularity only for 1 day timeframe
        - No hourly data past 89 days
        
    Returns:
        dict: JSON response with price, market cap, and volume data
    """
    start = int(dt.datetime.now().timestamp())
    end = int((dt.datetime.now() - dt.timedelta(days=timeframe)).timestamp())
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart/range?vs_currency=usd&from={end}&to={start}"
    return requests.get(url, headers=headers).json()

def CoinGecko_IDs_Pandas(coins, headers):
    """Map cryptocurrency names to CoinGecko IDs
    
    Args:
        coins (list): List of cryptocurrency names
        headers (dict): API request headers with authentication
    
    Returns:
        pandas.DataFrame: ID mapping for requested coins
    """
    url = "https://api.coingecko.com/api/v3/coins/list"
    response = requests.get(url, headers=headers)
    id_map = pd.DataFrame(response.json()).set_index('name')
    return id_map.loc[coins]

def CoinGecko_TopCoinsMC_Pandas(number, headers):
    """Get top cryptocurrencies by market cap
    
    Args:
        number (int): Number of top coins to retrieve
        headers (dict): API request headers with authentication
    
    Returns:
        numpy.ndarray: Array of CoinGecko IDs for top coins
    """
    url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd"
    response = pd.DataFrame(requests.get(url, headers=headers).json())
    investment_universe = response.head(number)['id'].values
    return investment_universe
