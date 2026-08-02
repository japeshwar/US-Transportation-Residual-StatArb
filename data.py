"""
data.py
==============
Dynamic Transportation Factor Laboratory
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

stocks = ['DAL', 'UAL', 'AAL', 'LUV', 'ALK', 'JBLU', 'UNP', 'CSX', 'NSC', 'CP', 'CNI', 'UPS', 'FDX', 'JBHT', 'ODFL', 'KNX', 'SAIA', 'SNDR', 'CHRW', 'EXPD']
benchmark = ['SPY']
macro = ['BZ=F', '^VIX']

start_date = '2015-01-01'
end_date = '2026-07-31'
train_end = '2019-12-31'
test_start = '2022-01-01'

def download_prices(
    tickers: list = None,
    start: str = start_date,
    end: str = end_date,
) -> pd.DataFrame:
    if tickers is None:
        tickers = stocks + benchmark + macro

    print(f"Downloading {len(tickers)} tickers [{start} -> {end}] ...")
    raw = yf.download(tickers, start = start, end = end, auto_adjust = True, progress = False)
    prices = raw['Close'].copy()
    prices = prices.rename(columns={'BZ=F': 'Brent', '^VIX': 'VIX'})
    return prices

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna(how='all')

def clean_data(returns: pd.DataFrame) -> pd.DataFrame:
    returns = returns.ffill(limit=2)
    stocks_cols = [c for c in returns.columns if c in stocks]
    returns = returns.dropna(subset = stocks_cols)

    for col in returns.columns:
        mu = returns[col].mean()
        sigma = returns[col].std()
        returns[col] = returns[col].clip(lower = mu - 4 * sigma, upper = mu + 4 * sigma)
    return returns

def load_data() -> tuple:
    """
    Returns:
        stocks_returns: log returns of the pure US transports
        macro_returns: SPY (log return) + Brent (raw level) + VIX (raw level)
        prices: all adjusted close prices
    """
    prices = download_prices()

    return_cols = stocks + ['SPY']
    returns = compute_log_returns(prices[return_cols])
    returns = clean_data(returns)

    stocks_returns = returns[stocks].copy()

    macro_returns = pd.DataFrame(index = stocks_returns.index)
    macro_returns['SPY'] = returns['SPY']

    macro_returns['Brent'] = prices['Brent'].reindex(stocks_returns.index)
    macro_returns['VIX'] = prices['VIX'].reindex(stocks_returns.index)

    # Forward-fill any tiny gaps in levels
    macro_returns[['Brent', 'VIX']] = macro_returns[['Brent', 'VIX']].ffill(limit = 2)

    print(f"Stocks: {stocks_returns.shape[1]} tickers")
    print(f"Macro columns: {list(macro_returns.columns)}")
    print(f"Dates: {len(stocks_returns)} trading days")
    print(f"Range: {stocks_returns.index[0].date()} -> {stocks_returns.index[-1].date()}")

    return stocks_returns, macro_returns, prices