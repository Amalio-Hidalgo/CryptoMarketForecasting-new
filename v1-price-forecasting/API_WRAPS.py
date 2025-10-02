
# API REQUESTS AND JSON READ FUNCTIONS
# -Inputs = Coins, Timeframe

import requests, time 
import pandas as pd
import datetime as dt

# ID Map
def get_ID_map(coins):
    url = "https://api.coingecko.com/api/v3/coins/list"
    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": "CG-r57ENE22zzPUiLmjnflatdf['flatdf['flatdf['y']']']FK7YHw"
    }
    response = requests.get(url, headers=headers)
    id_map = pd.DataFrame(response.json()).set_index('name')
    return id_map.loc[coins]

# Top Coins by Marketcap- Investment Universe
def get_top_mc_coins_ids(number):
    url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd"
    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": "CG-r57ENE22zzPUiLmjnyFK7YHw"
    }
    response = pd.DataFrame(requests.get(url, headers=headers).json())
    investment_universe = response.head(number)['id'].values
    return investment_universe

# Historical Price Data for All Coins Collected Into Flat DataFrame Sharing Resampled Index 
def CoinGecko_HistPrDat(timeframe, coins = get_top_mc_coins_ids(5)):
    start= int(dt.datetime.now().timestamp())
    end= int((dt.datetime.now() - dt.timedelta(days=timeframe)).timestamp())
    headers = {
    "accept": "application/json",
    "x-cg-demo-api-key": "CG-r57ENE22zzPUiLmjnyFK7YHw"
    }
    count=0
    for coin in coins:
        url = f"https://api.coingecko.com/api/v3/coins/"+ coin + f"/market_chart/range?vs_currency=usd&from={end}&to={start}"
        response = pd.DataFrame(requests.get(url, headers=headers).json())
        response['datetime'] = response['prices'].map(lambda x: x[0]).apply(pd.to_datetime, unit='ms', utc=True)
        for column in response.columns.drop('datetime'):
            response[column]= response[column].map(lambda x: x[1])
        if timeframe<=1: response = response.set_index('datetime', drop=True).resample('5min').last()
        else: response = response.set_index('datetime', drop=True)
        if count == 0: output = response
        else: output= output.join(response, rsuffix=': '+ coin, how='left')
        count=count+1
    return output.melt(ignore_index=False).reset_index(drop=False)



