"""
models.py
==============
Dynamic Transportation Factor Laboratory
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt

lookback = 90 # Default rolling covariance window (trading days)
zscore_window = 45 # Rolling z-score window (trading days)
n_components = 3 # Default # of PCs

window_candidates = [45, 60, 90, 120, 180, 240]

def compute_simple_residuals(
    stock_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    window: int = 90
) -> pd.DataFrame:
    """
    Rolling residual of each stock against SPY + IYT.
    
    residual_t = r_stock_t - beta1 * r_SPY_t - beta2 * r_IYT_t
    where betas are estimated on the previous 'window' days.
    """
    residuals = pd.DataFrame(index = stock_returns.index, columns = stock_returns.columns, dtype = float)
    
    factors = factor_returns[['SPY', 'IYT']].dropna()
    
    for ticker in stock_returns.columns:
        y = stock_returns[ticker].dropna()
        common_idx = y.index.intersection(factors.index)
        y = y.loc[common_idx]
        X = factors.loc[common_idx]
        
        for i in range(window, len(common_idx)):
            y_win = y.iloc[i-window:i]
            X_win = X.iloc[i-window:i]
            
            # Design matrix with intercept
            X_design = np.column_stack([np.ones(len(X_win)), X_win.values])
            
            try:
                beta = np.linalg.lstsq(X_design, y_win.values, rcond = None)[0]
                predicted = beta[0] + beta[1] * X.iloc[i]['SPY'] + beta[2] * X.iloc[i]['IYT']
                residuals.loc[common_idx[i], ticker] = y.iloc[i] - predicted
            except Exception:
                residuals.loc[common_idx[i], ticker] = np.nan
                
    return residuals

# Rolling Z-scores (we will be keeping this as the S-scores didn't work at all, so we redacted it)

def compute_rolling_zscore(
    residuals: pd.DataFrame,
    window: int = zscore_window,
) -> pd.DataFrame:
    mu = residuals.rolling(window = window, min_periods = 10).mean()
    sig = residuals.rolling(window = window, min_periods = 10).std().replace(0, np.nan)
    return (residuals - mu) / sig