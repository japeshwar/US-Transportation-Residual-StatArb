"""
models.py
==============
Dynamic Airline Factor Laboratory
_______
Redesigned per research roadmap:
  1. Ledoit-Wolf shrinkage covariance (replaces sample covariance)
  2. OU parameter estimation — half-life, mean-reversion speed, sigma
  3. OU S-score signal (replaces plain rolling z-score)
  4. Brent crude lead-lag diagnostic
  5. PC1 concentration (continuous, not binary)
 
All look-ahead bias controls preserved from v1.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt

lookback = 90 # Default rolling covariance window (trading days)
zscore_window = 20 # Rolling z-score window (trading days)
n_components = 3 # Default # of PCs
max_half_life = 15 # Days -- skip stocks with slower mean reversion

window_candidates = [45,60,90,120,180, 240]

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
    use_ledoit_wolf: bool — True = LW shrinkage,
                            False = sample covariance
 
    Outputs
    _______
    dict {pd.Timestamp → np.ndarray (N × N)}
    Key t uses returns [t-window, t-1]. Day t excluded — no look-ahead.
 
    Why Ledoit-Wolf Shrinkage
    _______
    The sample covariance matrix Σ_sample = (1/(T-1)) XᵀX has estimation
    error that grows with N/T. For N=6 assets and T=90 days, N/T = 0.067,
    which seems small but off-diagonal covariance estimates are still noisy.
 
    Ledoit-Wolf finds the optimal linear combination:
        Σ_LW = (1-α)·Σ_sample + α·μ·I
 
    where α (shrinkage coefficient) and μ (target) are analytically derived
    to minimize the expected Frobenius distance between Σ_LW and the true Σ.
 
    In practice: Σ_LW has smaller off-diagonal elements (less spurious
    correlation), making eigenvectors more stable across rolling windows
    and portfolio weights less extreme.
 
    Look-Ahead Prevention
    _______
    Window iloc[i-window : i] uses only returns up to day i-1.
    Day i return is excluded from every matrix it helps estimate.
    """
    
    cov_matrices: Dict[pd.Timestamp, np.ndarray] = {}
    dates = returns.index

    for i in range(window, len(dates)):
        # window is [i-window, i-1]: excludes current day i
        window_returns = returns.iloc[i-window:i]
        if use_ledoit_wolf:
            lw = LedoitWolf(assume_centered=False).fit(window_returns.values)
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
    Eigendecompose the covariance matrix. Return top-k components.
 
    Inputs
    _______
    cov_matrix: np.ndarray (N × N)
    n_components : int
 
    Outputs
    _______
    eigenvalues: np.ndarray (k,)
    eigenvectors: np.ndarray (N × k)
    explained_variance: np.ndarray (k,)
 
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
 
    return (
        eigenvalues[:n_components],
        eigenvectors[:, :n_components],
        expl[:n_components],
    )
    
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
 
    Outputs
    _______
    systematic: pd.DataFrame (T × N)  — r̂ = P r
    residuals: pd.DataFrame (T × N)  — ε = r − r̂
    eigenvalue_df: pd.DataFrame (T × k)
    eigenvector_df: pd.DataFrame (T × Nk)
    expl_var_df: pd.DataFrame (T × k)
    """
    
    tickers = returns.columns.tolist()
    k = n_components
    pc_cols = [f"PC{i+1}" for i in range(k)]
 
    sys_rows  = []; res_rows  = []
    eval_rows = []; evec_rows = []; expl_rows = []
    dates_out = []
 
    for date in returns.index:
        if date not in cov_matrices:
            continue
 
        r_t = returns.loc[date].values
        cov = cov_matrices[date]
 
        evals, evecs, expl = run_pca(cov, k)
        P = evecs @ evecs.T
        r_hat = P @ r_t
        eps = r_t - r_hat
 
        dates_out.append(date)
        sys_rows.append(r_hat)
        res_rows.append(eps)
        eval_rows.append(evals)
        expl_rows.append(expl)
        evec_rows.append(evecs.T.flatten())
 
    evec_cols = [f"PC{i+1}_{t}" for i in range(k) for t in tickers]
 
    systematic = pd.DataFrame(sys_rows, index=dates_out, columns=tickers)
    residuals = pd.DataFrame(res_rows, index=dates_out, columns=tickers)
    eigenvalue_df = pd.DataFrame(eval_rows, index=dates_out, columns=pc_cols)
    expl_var_df = pd.DataFrame(expl_rows, index=dates_out, columns=pc_cols)
    eigenvector_df = pd.DataFrame(evec_rows, index=dates_out, columns=evec_cols)
 
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
    return pd.DataFrame(rows, index=dates, columns=pc_cols)

# Rolling Z-scores (we will be keeping this as the S-scores didn't work at all)

def compute_rolling_zscore(
    residuals: pd.DataFrame,
    window: int = zscore_window,
) -> pd.DataFrame:
    mu  = residuals.rolling(window=window, min_periods=10).mean()
    sig = residuals.rolling(window=window, min_periods=10).std().replace(0, np.nan)
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
    return pd.Series(records, name="pc1_concentration")
 
# Brent Lead-Lag Diagnostic
 
def compute_brent_leadlag(
    residuals: pd.DataFrame,
    macro_returns: pd.DataFrame,
    max_lag: int = 21,
) -> pd.DataFrame:
    """
    Purpose
    _______
    Test whether lagged Brent crude returns predict airline residuals.
    Identifies potential leading indicator to improve signal timing.
 
    Inputs
    _______
    residuals: pd.DataFrame — airline PCA residuals
    macro_returns: pd.DataFrame — must contain 'Brent' column
    max_lag: int — test lags 1 through max_lag
 
    Outputs
    _______
    pd.DataFrame — correlation of Brent(t-lag) with residual(t) for each lag and each airline
 
    Interpretation
    _______
    If Brent(t-1) has correlation −0.30 with DAL residual(t):
        A 1-day rise in Brent predicts a NEGATIVE residual for DAL
        (oil cost shock not yet priced → airlines underperform factor model)
 
    Trading application:
        On days of large Brent moves, pre-position before z-score crosses
        the threshold. This reduces entry lag and improves timing.
 
    Note: This is a diagnostic for the TRAINING period.
    Only use as a signal modification if correlation is:
        - Consistent in direction across multiple lags
        - Magnitude > 0.10 (economically meaningful)
        - Stable across sub-periods
 
    Assumptions
    -----------
    - 'Brent' column must exist in macro_returns (case-sensitive)
    - Correlations computed on aligned dates only
    """
    if 'Brent' not in macro_returns.columns:
        print("Warning: 'Brent' not found in macro_returns.")
        print(f"Available columns: {macro_returns.columns.tolist()}")
        return pd.DataFrame()
 
    brent  = macro_returns['Brent']
    common = residuals.index.intersection(brent.index)
    res = residuals.loc[common]
    br = brent.loc[common]
 
    rows = []
    for lag in range(1, max_lag + 1):
        row = {'lag': lag}
        for col in res.columns:
            aligned = pd.concat([br.shift(lag), res[col]], axis=1).dropna()
            if len(aligned) > 30:
                row[col] = round(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]), 4)
            else:
                row[col] = np.nan
        rows.append(row)
 
    df = pd.DataFrame(rows).set_index('lag')
    print("\n=== Brent Lead-Lag Correlations ===")
    print("(negative = Brent rise predicts airline underperformance)")
    print(df.round(4).to_string())
    return df