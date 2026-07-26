"""
Dynamic Airline Factor Laboratory
_______
Rolling covariance, rolling PCA via eigendecomposition, factor model,
residual (mispricing) computation, and rolling z-score generation.

Signal Timing — No Look-Ahead Bias
_______
The central timing discipline of this module:

    Day t-W  ───────────────────  Day t-1  |  Day t
    [rolling covariance estimation window] |  r_t arrives
                                           |  residual computed
                                           |  z-score computed
                                           |  SIGNAL formed
                                           |  Day t+1: position executed

Concretely: the covariance matrix "dated" t uses returns [t-W, t-1].
Day t's return is NEVER inside its own covariance estimation window.
The residual at day t is computed using eigenvectors from [t-W, t-1].
Positions are then shifted one day before backtesting (executed at t+1).

Mathematical Framework
_______
Let  R ∈ ℝ^{T × N}  be the matrix of daily log returns for N airlines.

(1) COVARIANCE  (at date t, using W days ending at t-1)
        Σ_t = 1/(W-1) · (R_w - μ_w)ᵀ (R_w - μ_w) × 252

(2) EIGENDECOMPOSITION
        Σ_t = Q Λ Qᵀ
        Q ∈ ℝ^{N×N}: columns are orthonormal eigenvectors (principal components)
        Λ ∈ ℝ^{N×N}: diagonal eigenvalue matrix, λ₁ ≥ λ₂ ≥ … ≥ λ_N

        Explained variance ratio:  EVR_i = λ_i / Σ_j λ_j

(3) SYSTEMATIC COMPONENT  (factor model projection)
        f_t    = Q_kᵀ r_t           ∈ ℝ^k   (factor returns)
        r̂_t   = Q_k Q_kᵀ r_t       ∈ ℝ^N   (systematic component)
        where Q_k = first k columns of Q (top k eigenvectors)
        P = Q_k Q_kᵀ is the rank-k projection matrix onto factor space.

(4) RESIDUAL  (relative mispricing)
        ε_t = (I − P) r_t           ∈ ℝ^N
        The component of returns orthogonal to all k common factors.
        Under the mean-reversion hypothesis, |ε_t| is stationary.

(5) Z-SCORE
        z_{i,t} = (ε_{i,t} − μ_{ε,i}) / σ_{ε,i}
        where μ and σ are rolling 30-day mean and std of residuals.

Framing Note
_______
The residual ε_t is NOT an "expected return." It is a relative mispricing:
the portion of airline i's return on day t that cannot be attributed to the
k common airline risk factors identified by PCA. The trading hypothesis is
that unusually large residuals (|z| > 2) tend to mean-revert, implying
that airline i is temporarily mispriced relative to its peers.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

lookback = 90 # Default rolling covariance window (trading days)
zscore_window = 20 # Rolling z-score window (trading days)
n_components = 3 # Default # of PCs
 
window_candidates = [45,60,90,120,180]

def rolling_covariances(
    returns: pd.DataFrame,
    window: int = lookback,
) -> Dict[pd.Timestamp, np.ndarray]:
    """
    Purpose
    ___
    Compute an annualized sample covariance matrix for each trading date
    using a rolling lookback window that EXCLUDES the current day.

    Inputs
    _______
    returns : pd.DataFrame (T × N)  — daily log returns
    window  : int                   — rolling window length in trading days

    Outputs
    _______
    dict  {pd.Timestamp → np.ndarray (N × N)}
    The matrix stored at date t uses returns from [t-window, t-1].
    The earliest key in the dictionary is returns.index[window].

    Mathematical Explanation
    _______
    Sample covariance with Bessel's correction (divides by W-1, unbiased):

        Σ_t = 1/(W-1) · Σ_{s=t-W}^{t-1} (r_s − μ_w)(r_s − μ_w)ᵀ × 252

    Annualization by 252 assumes i.i.d. daily returns — standard in
    equity factor models.

    Symmetry enforcement:  Σ ← (Σ + Σᵀ) / 2
    Floating-point operations can introduce tiny asymmetries (order 1e-16).
    Enforcing symmetry guarantees np.linalg.eigh receives valid input.

    Look-Ahead Bias Prevention
    _______
    Window: returns.iloc[i - window : i]  →  [t-W, t-1], excludes day t.
    The covariance matrix stored at dates[i] is fully known before
    day i's return is observed. This is the critical timing fix.

    Assumptions
    _______
    - First available matrix is at index `window` (day 0 through window-1
      are consumed by the first estimation window).
    - Bessel correction applied (pandas .cov() default).
    - Annualization factor: 252 trading days per year.
    """
    cov_matrices: Dict[pd.Timestamp, np.ndarray] = {}
    dates = returns.index

    for i in range(window, len(dates)):
        # window is [i-window, i-1]: excludes current day i
        window_returns = returns.iloc[i-window:i]
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
    Decompose a symmetric covariance matrix into its principal components
    via eigendecomposition. Return the top k eigenvalues, eigenvectors,
    and explained variance ratios.

    Inputs
    _______
    cov_matrix   : np.ndarray (N × N)  — symmetric PSD covariance matrix
    n_components : int                 — number of top components to retain

    Outputs
    _______
    eigenvalues        : np.ndarray (k,)    — top k eigenvalues, descending
    eigenvectors       : np.ndarray (N × k) — columns are unit eigenvectors
    explained_variance : np.ndarray (k,)    — fraction of total variance per PC

    Mathematical Explanation
    _______
    The eigenvalue equation for symmetric Σ:

        Σ v_i = λ_i v_i

    λ_i  = variance of the data projected onto direction v_i
    v_i  = the direction (principal component) itself

    np.linalg.eigh is used (not .eig) because:
    - Specialised for symmetric/Hermitian matrices
    - Guarantees real eigenvalues (no spurious imaginary parts)
    - ~2× faster than the general eig routine
    - Returns eigenvalues in ascending order → reversed below

    Explained variance:  EVR_i = λ_i / trace(Σ)
    trace(Σ) = sum of all eigenvalues = total portfolio variance.

    Component Selection Guidance (for 6-stock airline universe)
    _______
    k = 1: PC1 alone typically explains 50–65% of variance.
           Represents the broad airline / equity market factor.
           Cleanest interpretation; residuals are less precise.

    k = 2: PC1 + PC2 typically explains 65–80% of variance.
           PC2 separates legacy carriers (DAL, UAL, AAL) from
           low-cost carriers (LUV, ALK, JBLU) or captures
           short-haul vs long-haul exposure.
           Recommended default for a 6-stock universe.

    k = 3: PC1–PC3 may explain 80–90% but the third component
           can be unstable with only 6 stocks. Test empirically.

    Sign Convention
    _______
    Eigenvectors have inherent sign ambiguity: if v is an eigenvector,
    so is -v. Without a convention, PC loadings flip sign across rolling
    windows, making time-series analysis of factor exposures meaningless.

    Convention: the element with the largest absolute value in each
    eigenvector is constrained to be positive. This anchors sign to the
    dominant loading and ensures consistency across rolling estimates.

    Assumptions
    _______
    - Input must be symmetric PSD (enforced in rolling_covariance).
    - Eigenvalues below zero (floating-point noise) are clipped to zero.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort descending: greatest eigenvalue first
    
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Boot the noise of very small eigenvalues
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Dominant loading is positive
    for j in range(eigenvectors.shape[1]):
        dominant = np.argmax(np.abs(eigenvectors[:, j]))
        if eigenvectors[dominant, j] < 0:
            eigenvectors[:, j] *= -1

    # Explain the variance
    total_var = eigenvalues.sum()
    explained_variance = (eigenvalues / total_var
        if total_var > 0
    else np.zeros_like(eigenvalues))

    return(eigenvalues[:n_components],eigenvectors[:, :n_components],explained_variance[:n_components])
# Factor Model + Residual

def compute_factor_model(
    returns: pd.DataFrame,
    cov_matrices: Dict[pd.Timestamp, np.ndarray],
    n_components: int = n_components
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Purpose
    _______
    Apply the rolling PCA factor model to decompose each day's airline
    returns into a systematic (factor-driven) component and an idiosyncratic
    residual that represents relative mispricing.

    Inputs
    _______
    returns      : pd.DataFrame (T × N)           — daily log returns
    cov_matrices : dict {Timestamp → ndarray}      — rolling covariance matrices
                   Key t uses returns [t-W, t-1]: no look-ahead.
    n_components : int                             — number of PCs to retain

    Outputs
    _______
    systematic      : pd.DataFrame (T' × N)  — r̂_t = P r_t
    residuals       : pd.DataFrame (T' × N)  — ε_t  = r_t − r̂_t
    eigenvalue_df   : pd.DataFrame (T' × k)  — eigenvalue time series
    eigenvector_df  : pd.DataFrame (T' × Nk) — flattened eigenvectors
    expl_var_df     : pd.DataFrame (T' × k)  — explained variance ratios

    Mathematical Explanation
    _______
    At each date t where a covariance matrix exists:

    Covariance Σ_t  ← estimated from returns [t-W, t-1]  (no look-ahead)
    Eigenvectors Q_k ← top k columns of Q in Σ_t = Q Λ Qᵀ

    Projection matrix:
        P_t = Q_k Q_kᵀ                   (N×N, rank-k orthogonal projection)

    P_t has the properties:  P² = P  (idempotent),  Pᵀ = P  (symmetric)
    It maps any vector to its closest point in the k-dimensional factor space.

    Systematic component:
        r̂_t = P_t r_t = Q_k (Q_kᵀ r_t)  ∈ ℝ^N

    Residual (relative mispricing):
        ε_t = r_t − r̂_t = (I − P_t) r_t  ∈ ℝ^N

    Key properties of ε_t:
    (1) Orthogonal to every column of Q_k:  Q_kᵀ ε_t = 0
    (2) E[ε_t] ≈ 0  (by factor model construction)
    (3) If ε_t exhibits mean reversion, it is tradeable as a spread

    Framing Note
    _______
    r̂_t is labeled "systematic component" NOT "expected return."
    It is the part of today's return explained by common airline factors.
    ε_t is the mispricing: airline i moved more or less than its factor
    loadings predicted. This mispricing is the trading signal.

    Timing Note
    _______
    cov_matrices[t] uses returns up to t-1 only. Using it to project r_t
    (same-day return) is clean. The signal is formed at end of day t.
    Execution happens on day t+1 (position is shifted in backtest.py).

    Assumptions
    _______
    - Only dates present in cov_matrices are included in output.
    - Factor loadings change every day as the rolling window advances.
    - n_components = 3 is the recommended default for 6 airlines.
    """
    tickers = returns.columns.tolist()
    k = n_components
    pc_cols = [f"PC{i+1}" for i in range(k)]

    systematic_rows = []
    residual_rows = []
    eigenval_rows = []
    eigenvec_rows = []
    explvar_rows = []
    valid_dates = []

    for date in returns.index:
        if date not in cov_matrices:
            continue
            
        r_t = returns.loc[date].values
        cov = cov_matrices[date]

        eigenvalues, eigenvectors, explained = run_pca(cov, k)

        P = eigenvectors @ eigenvectors.T
        r_hat = P @ r_t
        epsilon = r_t - r_hat
        valid_dates.append(date)
        systematic_rows.append(r_hat)
        residual_rows.append(epsilon)
        eigenval_rows.append(eigenvalues)
        explvar_rows.append(explained)
        eigenvec_rows.append(eigenvectors.T.flatten())

    evec_cols = [f"PC{i + 1}_{t}" for i in range(k) for t in tickers]

    systematic = pd.DataFrame(systematic_rows, index = valid_dates, columns = tickers)
    residuals = pd.DataFrame(residual_rows, index = valid_dates, columns = tickers)
    eigenvalue_df = pd.DataFrame(eigenval_rows, index = valid_dates, columns = pc_cols)
    expl_var_df = pd.DataFrame(explvar_rows, index = valid_dates, columns = pc_cols)
    eigenvector_df = pd.DataFrame(eigenvec_rows, index = valid_dates, columns = evec_cols)

    return systematic, residuals, eigenvalue_df, eigenvector_df, expl_var_df
# This portion celebrates Factor Returns

def compute_factor_returns(
    returns: pd.DataFrame,
    cov_matrices: Dict[pd.Timestamp, np.ndarray],
    n_components: int = n_components,
) -> pd.DataFrame:
    """
    Purpose
    _______
    Compute the return of each principal component factor portfolio
    (the projection of airline returns onto each PC) at each date.

    Inputs
    _______
    returns      : pd.DataFrame (T × N)   — airline log returns
    cov_matrices : dict {Timestamp → ndarray}
    n_components : int

    Outputs
    _______
    pd.DataFrame (T' × k)  — factor returns  f_{i,t} = v_iᵀ r_t

    Mathematical Explanation
    _______
    The i-th factor return at time t:

        f_{i,t} = v_iᵀ r_t = Σ_j v_{ij} · r_{j,t}

    This is the return of a notional long-short portfolio whose weights
    equal the i-th eigenvector. Because eigenvectors are unit vectors
    (||v_i|| = 1), gross exposure = 1 by construction.

    Interpretation:
    - PC1 factor return ≈ equal-weighted airline index return
    - PC2 factor return ≈ legacy vs low-cost carrier spread
    """
    pc_cols = [f"PC{i + 1}" for i in range(n_components)]
    rows = []
    dates = []

    for date in returns.index:
        if date not in cov_matrices:
            continue

        r_t = returns.loc[date].values
        cov = cov_matrices[date]
        _, evecs, _ = run_pca(cov, n_components)
        rows.append(evecs.T @ r_t)
        dates.append(date)

    return pd.DataFrame(rows, index = dates, columns = pc_cols)
# Rolling Z-scores

def compute_rolling_zscore(
    residuals: pd.DataFrame,
    window: int = zscore_window,
) -> pd.DataFrame:
    """
    Purpose
    _______
    Standardize residuals relative to their recent rolling history,
    producing z-scores that drive the long/short trading signals.

    Inputs
    _______
    residuals : pd.DataFrame (T × N)  — daily mispricing residuals ε_t
    window    : int                   — rolling window for mean/std

    Outputs
    _______
    pd.DataFrame (T × N)  — rolling z-scores z_{i,t}

    Mathematical Explanation
    _______
    For airline i at time t, using a rolling window of length W_z:

        μ_{i,t}  = mean(ε_{i,s},  s ∈ [t-W_z+1, t])
        σ_{i,t}  = std(ε_{i,s},   s ∈ [t-W_z+1, t])
        z_{i,t}  = (ε_{i,t} − μ_{i,t}) / σ_{i,t}

    Signal interpretation:
        z >  +2  :  airline i returned more than common factors predict
                    → relatively overvalued among peers → SHORT
        z <  -2  :  airline i returned less than common factors predict
                    → relatively undervalued among peers → LONG
        |z| < 0.5:  mispricing has reverted → EXIT position

    The ±2 threshold corresponds to the 95th percentile of a standard
    normal (two-tailed 5%). Selected to balance signal frequency against
    statistical significance. Alternative thresholds (±1.5, ±2.5) can be
    tested on the training period only.

    Assumptions
    _______
    - Minimum 10 observations required before computing std (min_periods=10).
    - Rolling std of zero (flat residuals) replaced with NaN.
    - The 30-day z-score window is shorter than the 90-day covariance window
      by design: z-scores must reflect recent deviations, not long-run means.
    """
    roll_mean = residuals.rolling(window=window, min_periods = 10).mean()
    roll_std = residuals.rolling(window=window, min_periods = 10).std()
    roll_std = roll_std.replace(0.0, np.nan)
    return (residuals - roll_mean) / roll_std
# PC1 Concentration

def compute_pc1_concentration(
    cov_matrices: Dict[pd.Timestamp, np.ndarray],
    n_total: int = None,
) -> pd.Series:
    """
    Purpose
    _______
    Track the fraction of total variance explained by PC1 over time.
    Used as a diagnostic to monitor correlation regimes, NOT as a
    trading rule trigger.

    Inputs
    _______
    cov_matrices : dict {Timestamp → ndarray}  — rolling covariance matrices
    n_total      : int                          — total number of stocks (N)

    Outputs
    _______
    pd.Series  — PC1 explained variance ratio at each date

    Mathematical Explanation
    _______
    Concentration = λ₁ / Σ_i λ_i = λ₁ / trace(Σ)

    Interpretation:
    - Concentration ≈ 0.30–0.45 : normal regime, airlines moving semi-independently
    - Concentration ≈ 0.55–0.70 : elevated correlation, common factor dominant
    - Concentration ≈ 0.70+     : stress regime, all airlines highly correlated

    Research Use
    _______
    Plot alongside strategy daily PnL. Examine whether high concentration
    (all airlines move together) coincides with poor strategy performance
    (fewer diversifiable mispricings). If a consistent relationship emerges,
    consider a continuous regime adjustment out-of-sample — not a threshold.

    Avoid creating a trading rule (e.g., "suspend trading if concentration > 0.6")
    purely because it improves in-sample backtest performance. Validate the
    economic logic first, then test the rule on the validation period.

    Assumptions
    _______
    - Uses full N eigenvalues (not just top k) for correct denominator.
    """
    records = {}

    for date, cov in cov_matrices.items():
        evals = np.linalg.eigvalsh(cov)
        evals = np.maximum(evals, 0.0)
        total = evals.sum()
        concentration = evals[-1] / total if total > 0 else np.nan
        records[date] = concentration
    return pd.Series(records, name = "pc1_concentration")
def run_window_sensitivity(
    returns: pd.DataFrame,
    windows: List[int] = window_candidates,
    n_components: int = n_components,
) -> dict:
    """
    Purpose
    _______
    Compute residuals for each candidate window length so that downstream
    backtesting can evaluate which window produces the most stable
    out-of-sample results.

    Inputs
    _______
    returns      : pd.DataFrame (T × N)  — airline log returns (training period)
    windows      : list of int            — candidate window lengths to test
    n_components : int                    — number of PCs

    Outputs
    _______
    dict  {window_length → (residuals_df, zscores_df)}
    Each entry is the full residual and z-score output for that window.

    Usage
    _______
    Call this function on the TRAINING period only. Choose the window with
    the best training-period Sharpe ratio or most stable z-score statistics.
    Hold that window fixed and apply it to the validation period without
    further adjustment.

    Assumptions
    _______
    - Candidate windows are [60, 90, 120, 180] by default.
    - Testing should be limited to these four economically reasonable
      alternatives. Testing dozens of windows and reporting the best
      result constitutes data-snooping (multiple testing problem).
    - The selected window is a structural choice with economic motivation,
      not an optimized parameter.
    """
    results = {}

    for w in windows:
        print(f" Window {w:3d}d...", end = "  ")
        cov = rolling_covariances(returns, window = w)
        _, residuals, _, _, _ = compute_factor_model(returns, cov, n_components)
        zscores = compute_rolling_zscore(residuals).dropna()
        results[w] = (residuals, zscores)
        print("done")

    return results
