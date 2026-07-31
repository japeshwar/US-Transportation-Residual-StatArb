"""
data.py
==============
Dynamic Airline Factor Laboratory
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

airlines = ['DAL','UAL','AAL','LUV','ALK','JBLU','CPA']
benchmark = ['SPY','JETS']
macro = ['BZ=F','^VIX']
start_date = '2015-01-01'
end_date = '2026-07-17'
train_end = '2019-12-31'
test_start = '2022-01-01'

def download_prices(
    tickers: list=airlines + benchmark + macro,
    start:str= start_date,
    end: str = end_date,
) -> pd.DataFrame:
    """
    Purpose
    _______
    Download split-and-dividend adjusted closing prices for all tickers.

    Imputs
    _______
    tickers: list of string -- ticker symbols
    start: string -- start date "YYYY-MM-DD"
    end: string -- end date "YYYY-MM-DD"
    
    Outputs
    _______
    pd.DataFrame (T x N) -- adjusted closing prices, DatetimeIndex rows. BZ=F renamed to Brent; ^VIX renamed to VIX

    Assumptions
    _______
    - Adjusted prices account for splits and dividends
    - Brent front-month futures (BZ=F) may contain roll discontinuities, treat like a log-return series not a price.
"""
    print(f"Downloading {len(tickers)} tickers [{start} -> {end}] ...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw['Close'].copy()
    prices = prices.rename(columns={'BZ=F':'Brent','^VIX':'VIX'})
    return prices

# Return Computation

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose
    _______
    Compute continuously compounded daily returns from price levels.

    Inputs
    _______
    prices: pd.DataFrame (T x N) -- adjusted closing prices

    Outputs
    _______
    pd.DataFrame (t-1 X N) -- log returns r_t = ln(P_t / P_{t-1})

    Math Explanation
    _______
    r_t = ln(P_t) - ln(P_{t-1})

    Log returns are preferred over simple returns because:
    1) Time additivity: r_{1-->t} = Σ r_t (simple returns do not add)
    2) Approximate symmetry: gains and losses are symmetric in log space
    3) No theoretical lower bound of -100%

    Assumptions
    _______
    - Prices are strictly positive (guaranteed by adjusted close date)
    - The first row (NaN by construction) is dropped
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna(how='all')
def clean_data(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose
    -------
    Remove data-quality issues: short gaps, insufficient coverage,
    and extreme outliers that likely reflect data errors.

    Inputs
    ------
    returns : pd.DataFrame (T × N)  — raw log returns (may contain NaNs)

    Outputs
    -------
    pd.DataFrame  — cleaned log returns aligned to common trading dates

    Cleaning Steps
    --------------
    (1) Forward-fill gaps of ≤ 2 consecutive days.
        Rationale: market holidays and early closes produce legitimate
        short gaps. Forward-filling (P_t = P_{t-1} → r_t = 0) is the
        standard treatment for illiquid or closed periods.

    (2) Drop rows where any airline return is missing.
        Rationale: the rolling covariance matrix requires a complete
        return panel. A single missing airline contaminates N-1 covariance
        estimates.

    (3) Clip outliers at ±5 standard deviations per series.
        Rationale: under normality, |r| > 5σ has probability ≈ 3×10⁻⁷
        (roughly once per 13,000 years of daily data). Values beyond this
        almost certainly reflect data errors, not genuine price moves.

    Assumptions
    -----------
    - Outlier clipping uses full-sample mean and std (acceptable for
      cleaning only; the factor model itself uses rolling statistics).
    """
    returns = returns.ffill(limit=2)
    airline_cols = [c for c in returns.columns if c in airlines]
    returns = returns.dropna(subset=airline_cols)

    for col in returns.columns:
        mu = returns[col].mean()
        sigma = returns[col].std()
        returns[col] = returns[col].clip(
        lower=mu - 4* sigma,
        upper=mu + 4 * sigma,
        )
    return returns
def load_data() -> tuple:
    """
    Purpose
    _______
    Master entry point: download → compute returns → clean → split.

    Outputs
    _______
    airline_returns: pd.DataFrame (T × 6)   — daily log returns, airlines
    macro_returns: pd.DataFrame (T × 4)   — SPX, SPY, BRENT, VIX
    prices: pd.DataFrame (T × N)   — raw adjusted closing prices

    Assumptions
    _______
    - VIX is a volatility level, not a return series. macro_returns["VIX"]
      contains Δ VIX (log change in index level), used only as a diagnostic.
    - The two DataFrames are aligned to a common DatetimeIndex.
    """

    prices = download_prices()
    returns = compute_log_returns(prices)
    returns = clean_data(returns)

    airline_cols = [c for c in airlines if c in returns.columns]
    macro_cols = [c for c in ['SPY','JETS','Brent','VIX']
                    if c in returns.columns]

    airline_returns = returns[airline_cols].copy()
    macro_returns = returns[macro_cols].copy()

    common_idx = airline_returns.index.intersection(macro_returns.index)
    airline_returns = airline_returns.loc[common_idx]
    macro_returns = macro_returns.loc[common_idx]

    print(f"Airlines: {airline_returns.shape[1]} tickers")
    print(f"Macro: {macro_returns.shape[1]} tickers")
    print(f"Dates: {len(airline_returns)} trading days")
    print(f"Range: {airline_returns.index[0].date()} "
        f"-> {airline_returns.index[-1].date()}")
    print(f"Training: {airline_returns.index[0].date()} -> {train_end}")
    print(f"Validation: {test_start} -> {airline_returns.index[-1].date()}")

    return airline_returns, macro_returns, prices