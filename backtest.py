# Backtesting Strategy
"""
Dynamic Airline Factor Laboratory
_______
Trading engine separated into two conceptual stages:

    STAGE 1 — PORTFOLIO CONSTRUCTION
        Converts z-scores into directional signals and dollar-neutral weights.
        Functions: generate_signals() & construct_portfolio()

    STAGE 2 — EXECUTION & PERFORMANCE
        Applies transaction costs, computes PnL, evaluates performance
        including benchmark regression against JETS and SPY.
        Functions: compute_portfolio_returns(), compute_performance(),
                   benchmark_regression(), macro_regime_diagnostics()

Strategy Logic
_______
    Entry Long  : z < -2.0  (airline underperformed factor model → buy)
    Entry Short : z > +2.0  (airline outperformed factor model → sell)
    Hold        : -2 ≤ z ≤ -0.5  or  +0.5 ≤ z ≤ +2  (within bounds)
    Exit        : |z| < 0.5  (mispricing has reverted)

Position sizing: equal-weight within longs and shorts, dollar-neutral.
Execution: positions formed at end of day t, executed at open of day t+1.
           (Implemented via a 1-day shift on the position DataFrame.)
Transaction costs: 15 basis points one-way on turnover.
"""
import numpy as np
import pandas as pd
import sys, os
from statsmodels.tsa.stattools import adfuller
import scipy.stats as stats
from typing import Tuple
# Strategic Parameters
entry_threshold = 2 # Z score to enter a position
exit_threshold = 0.5 # Z score must fall beneath this to exit a position
concentration_close = 0.80 # PC1 > this --> close all positions (crisis mode regime hedge)
concentration_reopen = 0.65 # PC1 > this --> allow new positions again
transaction_cost = 0.0015 # 15 basis points
trading_days = 252
top_n = 2 # max positions per side (long or short)
# Portfolio Construction
def generate_signals(
    zscores: pd.DataFrame,
    entry:   float = entry_threshold,
    exit_:   float = exit_threshold,
    top_n:   int   = top_n,
) -> pd.DataFrame:
    """
    Convert z-scores to signals: +1 long, -1 short, 0 flat.
    Only trades top_n most mispriced stocks per side.
    """
    signals = pd.DataFrame(
        0.0, index=zscores.index, columns=zscores.columns
    )

    # Initialize prev BEFORE the loop — this was the bug
    # On iteration 0, new = prev.copy() needs prev to already exist
    prev = pd.Series(0.0, index=zscores.columns)

    for i, date in enumerate(zscores.index):
        z   = zscores.loc[date]
        new = prev.copy()   # safe now — prev always initialized

        # ── Step 1: Exit positions that have reverted ──────────────────────
        new[np.abs(z) < exit] = 0.0

        # ── Step 2: Prune positions no longer in top_n ────────────────────
        current_longs  = new[new ==  1.0].index.tolist()
        current_shorts = new[new == -1.0].index.tolist()

        if len(current_longs) > top_n:
            keep = z[current_longs].nsmallest(top_n).index
            for ticker in current_longs:
                if ticker not in keep:
                    new[ticker] = 0.0

        if len(current_shorts) > top_n:
            keep = z[current_shorts].nlargest(top_n).index
            for ticker in current_shorts:
                if ticker not in keep:
                    new[ticker] = 0.0

        # ── Step 3: Enter new top_n signals ───────────────────────────────
        long_universe  = z[(z < -entry) & (new != 1.0)]
        short_universe = z[(z >  entry) & (new != -1.0)]

        if len(long_universe) > 0:
            slots = top_n - (new == 1.0).sum()
            if slots > 0:
                for ticker in long_universe.nsmallest(top_n).index[:slots]:
                    new[ticker] = 1.0

        if len(short_universe) > 0:
            slots = top_n - (new == -1.0).sum()
            if slots > 0:
                for ticker in short_universe.nlargest(top_n).index[:slots]:
                    new[ticker] = -1.0

        signals.loc[date] = new
        prev = new.copy()

    return signals

def apply_concentration_filter(
    signals:       pd.DataFrame,
    concentration: pd.Series,
    close_above:   float = concentration_close,
    reopen_below:  float = concentration_reopen,
) -> pd.DataFrame:
    """
    Purpose
    -------
    Zero out all positions when PC1 concentration exceeds the crisis threshold.
    Prevents trading during stress regimes when all airlines move in lockstep
    and there is no idiosyncratic spread to capture.
 
    Inputs
    ------
    signals       : pd.DataFrame (T × N)  — raw signals from generate_signals()
    concentration : pd.Series (T,)         — PC1 explained variance ratio per day
                                             output of compute_pc1_concentration()
                                             in models.py
    close_above   : float                  — concentration above this → close all
    reopen_below  : float                  — concentration below this → re-enable
 
    Outputs
    -------
    pd.DataFrame (T × N)  — filtered signals
 
    Mathematical Explanation
    ------------------------
    PC1 concentration = λ₁ / Σᵢ λᵢ
 
    When concentration > 0.80:
        The first principal component (≈ market/sector factor) explains
        more than 80% of total variance. All airlines are nearly perfectly
        correlated. Residuals from the factor model are essentially zero
        or noise. No genuine relative mispricing exists to trade.
 
    Hysteresis band (close_above=0.80, reopen_below=0.65):
        Without hysteresis, concentration oscillating near 0.80 would
        cause the strategy to toggle on/off every day, generating turnover.
        We close when concentration > 0.80 and only reopen when it drops
        back below 0.65.
 
    Empirically:
        Normal market:  concentration ≈ 0.30–0.50
        Elevated:       concentration ≈ 0.50–0.70
        Stress:         concentration ≈ 0.70–0.85
        Crisis (COVID): concentration ≈ 0.85–0.95
 
    Assumptions
    -----------
    - Concentration is computed from the SAME rolling covariance that
      generates the residuals — consistent timing, no look-ahead.
    - Dates without concentration data (early rows) are treated as safe
      (no filter applied). Conservative choice: prefer being in the market
      when we lack information rather than sitting out.
    """
    filtered   = signals.copy()
    blocked    = False   # tracks whether we are currently in a blocked regime
    common = signals.index.intersection(concentration.index)
    for date in common:
        c = concentration.loc[date]
 
        # Engage crisis filter
        if c > close_above:
            blocked = True
 
        # Release crisis filter only when concentration fully recovers
        if blocked and c < reopen_below:
            blocked = False
 
        # Zero all positions while blocked
        if blocked:
            filtered.loc[date] = 0.0
 
    return filtered

def construct_portfolio(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose
    _______
    Convert directional signal into dollar-neutral portfolio weights

    Inputs
    _______
    signals: pd.DataFrame (T x N) - signals in {-1.0, 0.0, +1.0)

    Outputs
    _______
    pd.DataFrame (T x N) - portfolio weights

    Math Explanation
    _______
    Dollar-neutral construction:

    1) Identify long positions (s_i = +1) and short positions (s_i = -1)
    2) Assign equal weight within longs: w_i = +1 / (2 x n_long)
    3) Assign equal weight within shorts: w_i = -1 / (2 x n_short)

    Properties:
    - Gross exposure = 1.0 (each side 50%, total 100%)
    - Net exposure = 0.0 (dollar-neutral, no directional market bet)
    - Equal weighting avoids concentration in individual names
    - Weights are rescaled each rebalancing period

    Market Neutrality Note
    _______
    Dollar-neutrality (Σ w_i = 0) does not guarantee beta neutrality.
    The portfolio may still carry residual beta to JETS/SPY depending
    on which airlines are long vs short. The benchmark regression in
    compute_performance() quantifies this residual exposure.

    Assumptions
    _______
    - If only lnogs or only shorts exist on a given date, the portfolio is one-sided. Uncommon under the +/- 2σ threshold.
    - Weights of zero on flat positions (s = 0)
    """

    positions = signals.copy().astype(float)

    for date in positions.index:
        row = positions.loc[date]
        n_long = (row > 0).sum()
        n_short = (row < 0).sum()

        if n_long > 0:
            positions.loc[date, row > 0] = 1.0 / (2 * n_long)
        if n_short > 0:
            positions.loc[date, row < 0] = -1.0 / (2 * n_short)

    return positions
# Execution & Performance

def compute_portfolio_returns(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    cost: float = transaction_cost,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Purpose
    _______
    Shift positions by one day (execute @ t+1), compute gross returns,
    apply transaction costs, and return net daily PnL.

    Inputs
    _______
    positions: pd.DataFrame (T x N) - portfolio weights (from construct_portfolio)
    returns: pd.DataFrame (T x N) - daily log returns for airlines
    cost: float - one-way transactions cost (0.0015 = 15 bps)

    Outputs
    _______
    net_returns: pd.Series (T,) - daily net portfolio returns
    trades_df: pd.DataFrame - trade log: gross returns, turnover, costs

    Math Explanation
    _______
    Execution Lag (Critical):
    positions_executed = positions.shift(1)

    Signals at end of day t -> execution at open of day t+1
    This one-day shift is the correct implementation of "no look ahead bias"

    Gross return on day t:
        R_t = Σ_i w_{i,t-1} · r_{i,t} = wᵀ r

    Turnover on day t:
        TO_t = Σ_i |w_{i,t} − w_{i,t-1}| / 2

    Dividing by 2 avoids double-counting: selling $1 and buying $1 of the
    same name is one unit of turnover, not two.

    Transaction cost drag:
        cost_t = TO_t × cost_per_unit

    Net return:
        R_net,t = R_t − cost_t

    Assumptions
    -----------
    - Costs of 15 bps are realistic for institutional trading in liquid
      U.S. airline equities (mid-cap, average daily volume $200M+).
    - No market impact assumed (appropriate for a small research portfolio).
    - Positions are shifted before cost computation; the initial position
      establishment cost is captured on the first day of the position.
    """
    common = positions.index.intersection(returns.index)
    pos = positions.loc[common].copy()
    ret = returns.loc[common, positions.columns].copy()
    # Execute on t+1: shift positions by one day
    pos_executed = pos.shift(1).fillna(0.0)
    # Gross daily returns
    gross_ret = (pos_executed * ret).sum(axis = 1)
    # Turnover: half the sum of absolute weight changes
    turnover = pos_executed.diff().abs().sum(axis=1) / 2
    # First day
    turnover.iloc[0] = pos_executed.iloc[0].abs().sum() / 2
    # Transaction cost drag
    cost_drag = turnover * cost
    # Net return
    net_ret = gross_ret - cost_drag

    trades_df = pd.DataFrame({
    "gross_return": gross_ret,
    "turnover": turnover,
    "transaction_cost": cost_drag,
    "net_returns": net_ret,
    "n_longs": (pos_executed > 0).sum(axis=1),
    "n_shorts": (pos_executed < 0).sum(axis=1),
    })

    return net_ret, trades_df

def compute_mean_reversion_stats(
    residuals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Purpose
    _______
    For each airline's residual series, estimate the speed and
    statistical significance of mean reversion. This is the core
    diagnostic for whether the z-score signal has predictive power.
 
    Inputs
    _______
    residuals : pd.DataFrame (T × N)  — PCA residuals (from models.py)
                Run this on the TRAINING period only.
 
    Outputs
    _______
    pd.DataFrame  — one row per airline:
        half_life    : days for residual to decay by half (AR1 estimate)
        ar1_coef     : AR(1) coefficient φ ∈ (0,1)
        adf_pvalue   : Augmented Dickey-Fuller test p-value
        adf_stat     : ADF test statistic
        tradeable    : bool — half_life<20 AND adf_pvalue<0.05
 
    Mathematical Explanation
    _______
    AR(1) Model:
        ε_t = a + φ · ε_{t-1} + η_t
 
    φ = AR(1) coefficient, estimated by OLS
        φ ∈ (0,1)  → mean reverting (ε_t pulls toward mean)
        φ = 1      → random walk (no mean reversion, DO NOT TRADE)
        φ > 1      → explosive (impossible after proper factor removal)
 
    Half-life from AR(1):
        The half-life is the number of days required for a deviation
        from the mean to reduce by half:
 
        half_life = ln(2) / -ln(φ)
 
        φ = 0.97  → half_life = 22.7 days  (slow, marginal)
        φ = 0.93  → half_life =  9.5 days  (good, tradeable)
        φ = 0.87  → half_life =  4.9 days  (fast, strong signal)
 
    Relationship to z-score window:
        The z-score rolling window should be LARGER than the half-life,
        otherwise the denominator (rolling std) captures the reversion
        itself, shrinking the z-score prematurely.
        Rule of thumb: z_window ≈ 3-5 × half_life
 
        If half_life ≈ 5 days → z_window = 15-25 days (NOT 30)
        If half_life ≈ 10 days → z_window = 30-50 days
 
    ADF Test:
        Formal hypothesis test for stationarity.
        H₀: series has a unit root (random walk, no mean reversion)
        H₁: series is stationary (mean reverting)
 
        p < 0.05  → reject H₀ → stationary → TRADEABLE
        p > 0.10  → fail to reject H₀ → possible random walk → CAUTION
 
    Diagnostic Decision Table:
        half_life < 10 AND p < 0.05 → STRONG signal, trade with confidence
        half_life < 20 AND p < 0.05 → MODERATE signal, trade normally
        half_life < 20 AND p > 0.05 → WEAK signal, reduce position size
        half_life > 20 OR  p > 0.10 → NO EDGE, do not trade this name
 
    Assumptions
    _______
    - ADF test uses maxlag=5 (captures autocorrelation up to 1 week)
    - Constant included in ADF regression (allows non-zero mean residual)
    - Minimum 50 observations required for reliable ADF test
    """
    results = []
 
    for ticker in residuals.columns:
        series = residuals[ticker].dropna()
        if len(series) < 50:
            results.append({
                "ticker":    ticker,
                "half_life": np.nan,
                "ar1_coef":  np.nan,
                "adf_pvalue": np.nan,
                "adf_stat":  np.nan,
                "tradeable": False,
            })
            continue
 
        # ── AR(1) half-life ────────────────────────────────────────────────
        y = series.iloc[1:].values
        x = series.iloc[:-1].values
        # OLS: y = a + φ·x + noise → φ = Cov(x,y)/Var(x)
        phi = np.cov(x, y)[0, 1] / np.var(x) if np.var(x) > 0 else np.nan
 
        if phi is not None and 0 < phi < 1:
            half_life = np.log(2) / (-np.log(phi))
        else:
            half_life = np.nan
 
        # ── ADF test ───────────────────────────────────────────────────────
        try:
            adf_result = adfuller(series, maxlag=5, regression='c',
                                   autolag='AIC')
            adf_stat   = adf_result[0]
            adf_pvalue = adf_result[1]
        except Exception:
            adf_stat   = np.nan
            adf_pvalue = np.nan
 
        # ── Tradeable determination ────────────────────────────────────────
        tradeable = (
            half_life is not None
            and not np.isnan(half_life)
            and half_life < 20
            and adf_pvalue is not None
            and not np.isnan(adf_pvalue)
            and adf_pvalue < 0.05
        )
 
        results.append({
            "ticker":     ticker,
            "half_life":  round(half_life, 2) if not np.isnan(half_life) else np.nan,
            "ar1_coef":   round(phi,       4) if phi and not np.isnan(phi) else np.nan,
            "adf_pvalue": round(adf_pvalue, 4) if not np.isnan(adf_pvalue) else np.nan,
            "adf_stat":   round(adf_stat,  4) if not np.isnan(adf_stat) else np.nan,
            "tradeable":  tradeable,
        })
 
    df = pd.DataFrame(results).set_index("ticker")
 
    print("\n=== Mean Reversion Diagnostics ===")
    print(f"{'Ticker':<8} {'Half-Life':>10} {'AR1(φ)':>8} "
          f"{'ADF p':>8} {'Tradeable':>10}")
    print("─" * 50)
    for t, row in df.iterrows():
        hl  = f"{row['half_life']:.1f}d" if not pd.isna(row['half_life']) else "N/A"
        phi = f"{row['ar1_coef']:.3f}"   if not pd.isna(row['ar1_coef'])  else "N/A"
        p   = f"{row['adf_pvalue']:.4f}" if not pd.isna(row['adf_pvalue']) else "N/A"
        tr  = "✓ YES" if row['tradeable'] else "✗ NO"
        print(f"{t:<8} {hl:>10} {phi:>8} {p:>8} {tr:>10}")
 
    print(f"\nZ-score window recommendation:")
    valid_hl = df['half_life'].dropna()
    if len(valid_hl) > 0:
        median_hl = valid_hl.median()
        rec_min   = int(median_hl * 3)
        rec_max   = int(median_hl * 5)
        print(f"  Median half-life: {median_hl:.1f} days")
        print(f"  Recommended z-score window: {rec_min}–{rec_max} days")
        print(f"  (Current: 30 days — adjust models.ZSCORE_WINDOW accordingly)")
 
    return df

def benchmark_regression(
    net_returns: pd.Series,
    macro_returns: pd.DataFrame,
) -> pd.Series:
    """
    Purpose
    _______
    Regress strat net returns against JETS (primary) and SPY (secondary)
    to decompose performance into alpha and beta components.

    Inputs
    _______
    net_returns: pd.Series - daily net strat returns
    macro_returns: pd.DataFrame - includes JETS, SPY daily returns

    Outputs
    _______
    pd.Series - regression statistics:
        jets_alpha: daily alpha vs JETS (annualized)
        jets_alpha_tstat: t-stat on alpha ( |t| > 2 -> statistically significant)
        jets_beta: beta exposure to JETS ETF
        jets_r2: R-squared of strategy vs JETS
        jets_corr: correlation of strategy with JETS
        spy_alpha: daily alpha vs SPY (annaulized ofc)
        spy_beta: beta exposure to SPY

    Math Explanation
    _______
    OLS regression (JETS as X, strategy return as Y):

        R_strategy,t = α + β · R_JETS,t + ε_t

    Alpha (α): return attributable to the strategy's skill, not market exposure.
               α_annual = α_daily × 252

    Beta (β):  sensitivity of strategy returns to JETS movements.
               A market-neutral strategy should have β ≈ 0.

    Alpha t-statistic:  t_α = α / SE(α)
               |t_α| > 2  →  alpha is statistically significant at ~5% level.
               This is the primary evidence that the strategy earns
               relative-value returns rather than airline sector beta.

    R-squared: fraction of strategy variance explained by JETS.
               R² ≈ 0 is desirable (strategy is uncorrelated with benchmark).

    Interpretation Guide
    --------------------
    Desirable result for a stat arb strategy:
        β  ≈  0.0 to 0.1   (near market-neutral)
        α  >  0             (positive excess return)
        |t_α| > 2           (statistically meaningful alpha)
        R²  <  0.10         (strategy uncorrelated with JETS)

    Assumptions
    -----------
    - JETS is the primary benchmark (airline sector ETF).
    - SPY is included as secondary (broad market context).
    - Returns aligned on common dates; dates without both series are dropped.
    """
    result = {}

    for bench in ["JETS","SPY"]:
        if bench not in macro_returns.columns:
            continue

        common = net_returns.index.intersection(macro_returns.index)
        y = net_returns.loc[common].values
        x = macro_returns.loc[common, bench].values

        # Remove any NaN pairs
        mask = ~(np.isnan(y) | np.isnan(x))
        y, x = y[mask], x[mask]

        if len(y) < 30:
            continue

        #OLS: adding the constant for ~alpha~
        X = np.column_stack([np.ones(len(x)),x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        alpha_daily, beta_coef = beta[0], beta[1]

        # Residual STD & standard error of alpha
        y_hat = X @ beta
        resid = y - y_hat
        s2 = resid.var(ddof=2)
        se_mat = s2 * np.linalg.inv(X.T @ X)
        se_alpha = np.sqrt(se_mat[0, 0])

        t_stat = alpha_daily / se_alpha if se_alpha > 0 else np.nan
        ss_tot = ((y - y.mean()) ** 2).sum()
        ss_res = (resid ** 2).sum()
        r2 = 1 - (resid**2).sum() / ss_tot if ss_tot > 0 else np.nan
        corr = np.corrcoef(y, x)[0, 1]

        b = bench.lower()
        result[f"{b}_alpha_annual"] = round(alpha_daily * trading_days, 4)
        result[f"{b}_alpha_tstat"] = round(t_stat, 3)
        result[f"{b}_beta"] = round(beta_coef, 4)
        result[f"{b}_r2"] = round(r2, 4)
        result[f"{b}_correlation"] = round(corr, 4)
    return pd.Series(result)

def compute_performance(
    net_returns: pd.Series,
    trades_df: pd.DataFrame,
    macro_returns: pd.DataFrame,
    rf: float = 0.0,
) -> pd.Series:
    """
    Purpose
    _______
    Compute comprehensive performance statistics for the strategy,
    including returns, risk metrics, and benchmark attribution.

    Inputs
    _______
    net_returns   : pd.Series      — daily net portfolio returns
    trades_df     : pd.DataFrame   — trade log (from compute_portfolio_returns)
    macro_returns : pd.DataFrame   — for benchmark regression
    rf            : float          — annualized risk-free rate (default 0)

    Outputs
    _______
    pd.Series  — performance statistics

    Mathematical Explanation
    _______
    Annualized Return:
        μ_ann = E[R_t] × 252

    Annualized Volatility:
        σ_ann = std(R_t) × √252

    Sharpe Ratio:
        SR = (μ_ann − rf) / σ_ann
        Assumes i.i.d. daily returns (standard simplification).

    Maximum Drawdown:
        MDD = min_t { (CumRet_t − max_{s≤t} CumRet_s) / max_{s≤t} CumRet_s }
        Measures the largest peak-to-trough decline in cumulative wealth.

    Calmar Ratio:
        Calmar = μ_ann / |MDD|
        Return per unit of maximum drawdown risk.

    Hit Rate:
        Fraction of trading days (non-zero return) where the strategy profited.
        Measures directional accuracy of signals.

    Turnover (annualized):
        Average daily turnover × 252.
        Measures portfolio churn; lower is better for cost management.

    Assumptions
    _______
    - rf = 0 is a simplification appropriate for relative strategy analysis.
    - Sharpe ratio uses daily returns scaled by √252 (standard convention).
    - Benchmark regression results appended from benchmark_regression().
    """
    r = net_returns.dropna()

    if len(r) == 0:
        return pd.Series({"error": "No returns computed"})

    # Return & Risk
    ann_return = r.mean() * trading_days
    ann_vol = r.std() * np.sqrt(trading_days)
    sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else np.nan

    # Drawdown
    cum_ret = (1 + r).cumprod()
    rolling_max = cum_ret.cummax()
    drawdown = (cum_ret - rolling_max) / rolling_max
    max_dd = drawdown.min()
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan

    # Hit Rate
    active = r[r != 0]
    hit_rate = (active > 0).sum() / len(active) if len(active) > 0 else np.nan

    # Turnover
    ann_turnover = trades_df["turnover"].mean() * trading_days

    # Average Holding Period
    is_active = (trades_df["n_longs"] + trades_df["n_shorts"] > 0).astype(int)
    run_ids = (is_active != is_active.shift()).cumsum()
    avg_holding = is_active.groupby(run_ids).sum().mean()

    # Core Stats
    core = pd.Series({
        "annualized_returns": round(ann_return, 4),
        "annualized_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "calmar_ratio": round(calmar, 3),
        "hit_rate": round(hit_rate, 4),
        "avg_holding_days": round(avg_holding, 1),
        "ann_turnover": round(ann_turnover, 3),
        "total_trading_days": len(r),
        "active_trading_days": len(active),
    })

    # Benchmark Regression
    bench = benchmark_regression(net_returns, macro_returns)

    return pd.concat([core, bench])

def macro_regime_diagnostics(
    net_returns: pd.Series,
    macro_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Purpose
    _______
    Examine how strategy performance varies across VIX and oil return
    quarts. Used as a diagnostic to understand macro sensitivity,
    not as a basis for trading rules.

    Inputs
    _______
    net_returns: pd.Series - daily net strategy returns
    macro_returns: pd.DataFrame - includes VIX and Brent

    Outputs
    _______
    pd.DataFrame - Sharpe ratio by VIX quart and by Brent return quart

    Interpretation
    _______
    - If Sharpe drops as VIX rises then strategy underperforms in stressful regimes.
    (Good to consider a continuous vol adjustment in this instance)
    - If Sharpe is stable across VIX, then strat is robust to regime changes.
    (Like water, put it in a cup it becomes the cup, vice versa)
    - If strategy underperforms when Brent returns are extreme, rapid oil price
    changes create systematic airline moves that swamp idiosyncratic mispricings.
    (Consider conditioning on |absolute value: Brent| < threshold.

    Important
    _______
    - Distinguish between high oil levels (abs price) and rapid oil changes (daily returns).
    - Airline equity mispricings are more likely driven by unexpected oil-price changes than by absolute levels.
    - Abstain from creating threshold-based rules (i.e., "pause when VIX climbs > 27") based solely on backtest.

    Assumptions
    _______
    - Quartiles computed over the entire available history (for diagnostics)
    - Sharpe within quartile computed with > 30 observations
    """
    common = net_returns.index.intersection(macro_returns.index)
    r = net_returns.loc[common]
    macro = macro_returns.loc[common]

    records = {}

    for macro_var in ["VIX","Brent"]:
        if macro_var not in macro.columns:
            print(f"  Warning: {macro_var} not in macro_returns columns")
            print(f"  Available: {macro_returns.columns.tolist()}")
            continue
        var_series = macro[macro_var].dropna()
        quartiles = pd.qcut(var_series, q = 4, labels = ["Q1","Q2","Q3","Q4"])

        sharpes = {}
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            mask = quartiles == q
            subset = r[mask]
            if len(subset) < 30:
                sharpes[q] = np.nan
                continue
            ann_r = subset.mean() * trading_days
            ann_v = subset.std() * np.sqrt(trading_days)
            sharpes[q] = round(ann_r / ann_v, 3) if ann_v > 0 else np.nan

        records[macro_var] = sharpes

    return pd.DataFrame(records).T
