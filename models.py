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
    
    residual_t = r_stock_t - β1 * r_SPY_t - β2 * r_IYT_t
    where betas are estimated on the previous `window` days.
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

def rolling_covariances(
    returns: pd.DataFrame,
    window: int = lookback,
    use_ledoit_wolf: bool = True,
) -> Dict[pd.Timestamp, np.ndarray]:
    """
    Purpose
    _______
    Compute annualized covariance matrix for each trading date.
    Default: Ledoit-Wolf shrinkage to reduce estimation error.
 
    Inputs
    _______
    returns: pd.DataFrame (T × N) — daily log returns
    window: int — lookback in trading days
    use_ledoit_wolf: bool — True = LW shrinkage, False = sample covariance
 
    Outputs
    _______
    dict {pd.Timestamp → np.ndarray (N × N)}
    Key t uses returns [t-window, t-1]. Day t excluded — no look-ahead.
    """
    
    cov_matrices: Dict[pd.Timestamp, np.ndarray] = {}
    dates = returns.index

    for i in range(window, len(dates)):
        # window is [i-window, i-1]: excludes current day i
        window_returns = returns.iloc[i-window:i]
        if use_ledoit_wolf:
            lw = LedoitWolf(assume_centered = False).fit(window_returns.values)
            cov = lw.covariance_ * 252
        else:
            cov = window_returns.cov().values * 252
        cov = (cov + cov.T) / 2
        cov_matrices[dates[i]] = cov
        
    return cov_matrices
    
# Principal Component Analysis via Eigendecomposition
def run_pca(
    cov_matrix: np.ndarray,
    n_components: int = n_components,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Purpose
    _______
    Eigendecompose the covariance matrix. Return top 3 components.
    
    Sign Convention
    _______
    Largest absolute loading constrained positive per eigenvector.
    Prevents sign flips across rolling windows.
    """
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[idx], 0.0)
    eigenvectors = eigenvectors[:, idx]
 
    for j in range(eigenvectors.shape[1]):
        dom = np.argmax(np.abs(eigenvectors[:, j]))
        if eigenvectors[dom, j] < 0:
            eigenvectors[:, j] *= -1
 
    total_var = eigenvalues.sum()
    expl = eigenvalues / total_var if total_var > 0 else np.zeros_like(eigenvalues)
 
    return (eigenvalues[:n_components], eigenvectors[:, :n_components], expl[:n_components])
    
# Factor Model + Residual

def compute_factor_model(
    returns: pd.DataFrame,
    cov_matrices: Dict[pd.Timestamp, np.ndarray],
    n_components: int = n_components
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Purpose
    _______
    Decompose returns into systematic (factor) and idiosyncratic (residual)
    components using rolling PCA projection.
    """
    
    tickers = returns.columns.tolist()
    k = n_components
    pc_cols = [f"PC{i+1}" for i in range(k)]
 
    sys_rows = []; res_rows = [];
    eval_rows = []; evec_rows = [];
    expl_rows = []; dates_out = []
 
    for date in returns.index:
        if date not in cov_matrices:
            continue
 
        r_t = returns.loc[date].values
        cov = cov_matrices[date]
 
        evals, evecs, expl = run_pca(cov, k)
        P = evecs @ evecs.T # Transpose
        r_hat = P @ r_t
        eps = r_t - r_hat # Epsilon
 
        dates_out.append(date)
        sys_rows.append(r_hat)
        res_rows.append(eps)
        eval_rows.append(evals)
        expl_rows.append(expl)
        evec_rows.append(evecs.T.flatten())
 
    evec_cols = [f"PC{i+1}_{t}" for i in range(k) for t in tickers]
 
    systematic = pd.DataFrame(sys_rows, index = dates_out, columns = tickers)
    residuals = pd.DataFrame(res_rows, index = dates_out, columns = tickers)
    eigenvalue_df = pd.DataFrame(eval_rows, index = dates_out, columns = pc_cols)
    expl_var_df = pd.DataFrame(expl_rows, index = dates_out, columns = pc_cols)
    eigenvector_df = pd.DataFrame(evec_rows, index = dates_out, columns = evec_cols)
 
    return systematic, residuals, eigenvalue_df, eigenvector_df, expl_var_df
    
# This portion celebrates Factor Returns

def compute_factor_returns(
    returns: pd.DataFrame,
    cov_matrices: Dict[pd.Timestamp, np.ndarray],
    n_components: int = n_components,
) -> pd.DataFrame:
    """
    Factor portfolio returns
    """
    pc_cols = [f"PC{i + 1}" for i in range(n_components)]
    rows, dates = [], []

    for date in returns.index:
        if date not in cov_matrices: continue
        _, evecs, _ = run_pca(cov_matrices[date], n_components)
        rows.append(evecs.T @ returns.loc[date].values)
        dates.append(date)
    return pd.DataFrame(rows, index = dates, columns = pc_cols)

# Rolling Z-scores (we will be keeping this as the S-scores didn't work at all, so we redacted it)

def compute_rolling_zscore(
    residuals: pd.DataFrame,
    window: int = zscore_window,
) -> pd.DataFrame:
    mu = residuals.rolling(window = window, min_periods = 10).mean()
    sig = residuals.rolling(window = window, min_periods = 10).std().replace(0, np.nan)
    return (residuals - mu) / sig
    
# PC1 Concentration

def compute_pc1_concentration(
    cov_matrices: Dict[pd.Timestamp, np.ndarray],
    n_total: int = None,
) -> pd.Series:
    """
    PC1 concentration = λ₁ / Σλᵢ per date.
    Used for continuous exposure scaling in backtest.
    """
    records = {}
    for date, cov in cov_matrices.items():
        evals = np.maximum(np.linalg.eigvalsh(cov), 0.0)
        total = evals.sum()
        records[date] = evals[-1] / total if total > 0 else np.nan
    return pd.Series(records, name = "pc1_concentration")
 
# Brent Lead-Lag Diagnostic
 
def compute_brent_leadlag(
    residuals: pd.DataFrame,
    macro_returns: pd.DataFrame,
    max_lag: int = 21,
) -> pd.DataFrame:
    """
    Purpose
    _______
    Test whether lagged Brent crude returns predict transportation residuals.
    Identifies potential leading indicator to improve signal timing.
 
    Trading application:
        On days of large Brent moves, pre-position before z-score crosses
        the threshold. This reduces entry lag and improves timing.
 
    Note: This is a diagnostic for the TRAINING period.
    Only use as a signal modification if correlation is:
        - Consistent in direction across multiple lags
        - Magnitude > 0.10 (economically meaningful)
        - Stable across sub-periods
 
    Assumptions
    _______
    - 'Brent' column must exist in macro_returns
    - Correlations computed on aligned dates only
    """
    if 'Brent' not in macro_returns.columns:
        print("Warning: 'Brent' not found in macro_returns.")
        print(f"Available columns: {macro_returns.columns.tolist()}")
        return pd.DataFrame()
 
    brent = macro_returns['Brent']
    common = residuals.index.intersection(brent.index)
    res = residuals.loc[common]
    br = brent.loc[common]
 
    rows = []
    for lag in range(1, max_lag + 1):
        row = {'lag': lag}
        for col in res.columns:
            aligned = pd.concat([br.shift(lag), res[col]], axis = 1).dropna()
            if len(aligned) > 30:
                row[col] = round(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]), 4)
            else:
                row[col] = np.nan
        rows.append(row)
 
    df = pd.DataFrame(rows).set_index('lag')
    print("\n=== Brent Lead-Lag Correlations ===")
    print("(negative = Brent rise predicts stocks underperformance)")
    print(df.round(4).to_string())
    return df