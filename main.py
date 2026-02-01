#========================================================================
#   Project     : Performant Alpha
#   File name   : main.py
#   Author      : Freeman Malit
#   Created date: 2025-09-01
#   Description : Advanced Backtesting and Strategy Evaluation w/ ETFs
#   Purpose     : Fetch and process ETF data for Backtesting
#========================================================================

# importing necessary libraries
import pytz # implement time awareness
import threading
import yfinance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from _alpha1 import Alpha1 as _Alpha1
from _alpha2 import Alpha2 as _Alpha2
from _alpha3 import Alpha3 as _Alpha3
from utils import load_pickle, save_pickle

def get_etf_tickers():
    """
    Function to get popular etf tickers from csv file {source: cnbc}
    Returns:
        list: A list of etf tickers       
    """
    df = pd.read_excel('ETFs.xlsx', sheet_name='etfs')
    etf_tickers = list(df['Symbol'])
    return etf_tickers

def get_history(ticker, period_start, period_end, granularity = "1d", tries = 0):
    """
    Function to fetch OHLCV history for ticker from Yahoo Finance.
    
    Args:
        ticker (str): Ticker symbol (e.g., "AAPL", "SPY", "BRK.B")
        start (str/datetime): Start date
        end (str/datetime): End date
        interval (str): Data interval ("1d", "1h", "5m", "1m", etc.)
        max_retries (int): Retry attempts in case of error

    Returns:
        pd.DataFrame with datetime index (tz-aware, UTC)
    """  
    try:
        # normalize Yahoo symbols like BRK.B -> BRK-B
        ticker = ticker.replace(".", "-")
        df = yfinance.Ticker(ticker).history(
                        start = period_start,
                        end = period_end,
                        interval = granularity,
                        auto_adjust = False, # don't adjust prices
                        raise_errors = True,
                        actions = True
                        ).reset_index()
    except Exception as err:
        if tries < 5:
            return get_history(ticker, period_start, period_end, granularity, tries+1)
        else:
            return pd.DataFrame()
    
    df = df.rename(columns={
        "Date":"datetime",
        "Open":"open",
        "High":"high",
        "Low":"low",
        "Close":"close",
        "Volume":"volume"
    })
    # if dataframe is empty
    if df.empty: 
        return pd.DataFrame()
    
    # make datetime tz-aware (UTC)
    df.datetime = pd.DatetimeIndex(df.datetime.dt.date).tz_localize(pytz.utc)

    df = df.drop(columns=["Dividends", "Stock Splits", 
                          "Adj Close", "Capital Gains"], 
                          errors="ignore")
        
    df = df.set_index("datetime",drop=True)
    return df

def get_histories(tickers, period_starts, period_ends, granularity = "1d"):
    dfs = [None]*len(tickers)
    def _helper(i):
        print(tickers[i])
        df = get_history(
            tickers[i],
            period_starts[i], 
            period_ends[i],
            granularity=granularity
        )
        dfs[i] = df

    # multithreading to speed up data collection
    threads = [threading.Thread(target=_helper,args=(i,)) for i in range(len(tickers))]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads] # wait for all threads to complete
    tickers = [tickers[i] for i in range(len(tickers)) if not dfs[i].empty]
    dfs = [df for df in dfs if not df.empty] # filter out empty dataframes
    return tickers, dfs

def get_ticker_dfs(start,end):
    try:
        # load from pickle if dataset already exists
        tickers, ticker_dfs = load_pickle("etf_dataset.obj")
    except Exception as err:
        tickers = get_etf_tickers()
        starts=[start]*len(tickers)
        ends=[end]*len(tickers)
        tickers,dfs = get_histories(tickers,starts,ends, granularity="1d")
        ticker_dfs = {ticker:df for ticker,df in zip(tickers,dfs)}
        save_pickle("etf_dataset.obj",(tickers,ticker_dfs))

    return tickers, ticker_dfs

if __name__ == "__main__":
    # define period, standardize time zones (UTC)
    period_start = datetime(2010, 1, 1, tzinfo=pytz.utc)
    period_end = datetime.now(pytz.utc)
    etf_tickers, etf_ticker_dfs = get_ticker_dfs(start=period_start,end=period_end)    

    _alpha1 = _Alpha1(etf_tickers, etf_ticker_dfs, start=period_start, end=period_end)
    _alpha2 = _Alpha2(etf_tickers, etf_ticker_dfs, start=period_start, end=period_end)
    _alpha3 = _Alpha3(etf_tickers, etf_ticker_dfs, start=period_start, end=period_end)

    _df1 = _alpha1.run_simulation()
    print(_df1.capital[-1])
    _df2 = _alpha2.run_simulation()
    print(_df2.capital[-1])    
    _df3 = _alpha3.run_simulation()
    print(_df3.capital[-1])

    print(_df1, _df2, _df3)

    # plot cumulative returns
    log_ret = lambda df: np.log((1 + df.capital_ret).cumprod())
    plt.plot(log_ret(_df1), label="alpha1")
    plt.plot(log_ret(_df2), label="alpha2")
    plt.plot(log_ret(_df3), label="alpha3")
    plt.legend()
    plt.show()

    # plot capital evolution
    plt.plot(_df1.capital, label="alpha1")
    plt.plot(_df2.capital, label="alpha2")
    plt.plot(_df3.capital, label="alpha3")
    plt.legend()
    plt.show()

    # plot non-zero returns volatility
    nzr = lambda df:df.capital_ret.loc[df.capital_ret!=0].fillna(0) # non-zero returns
    def plot_vol(r,label=None):
        vol = r.rolling(25).std() * np.sqrt(253)
        plt.plot(vol, label=label)

    plot_vol(nzr(_df1), label="alpha1")
    plot_vol(nzr(_df2), label="alpha2")
    plot_vol(nzr(_df3), label="alpha3")
    plt.legend()
    plt.show()

    # print annualized vol
    print(nzr(_df1).std()*np.sqrt(253), nzr(_df2).std()*np.sqrt(253), nzr(_df3).std()*np.sqrt(253))