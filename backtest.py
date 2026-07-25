# Backtesting Strategy
"""
Dynamic Airline Factor Laboratory
_______
Trading engine separated into two conceptual stages following the mentor's
advice:

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
import scipy.stats as stats
from typing import Tuple
# Strategic Parameters
entry_threshold = 2 # Z score to enter a position
exit_threshold = 0.75 # Z score must fall beneath this to exit a position
transaction_cost = 0.0015 # 15 basis points
trading_days = 252
# Portfolio Construction
def generate_signals(
    zscores: pd.DataFrame,
    entry: float = entry_threshold,
    exit: float = exit_threshold,
) -> pd.DataFrame:

    """
    Purpose
    _______
    Converting the rolling z-scores into raw directional signals using a band. Signal formed @ the end of day t, executed on t+1

    Inputs
    _______
    zscores: pd.DataFrame (T x N) - rolling z-scores of mispriced residuals
    entry: float - z-score magnitude to enter
    enxt: float - z-score magnitude to exit

    Outputs
    _______
    pd.DataFrame (T x N) - raw signals: +1 long, -1 short, 0 flat

    Mathematical Explanation
    _______
    The signal rule implements a band to avoid excessive trading:
     s_{i,t} = +1   if z_{i,t} < -entry (enter long)
        s_{i,t} = -1   if z_{i,t} > +entry (enter short)
        s_{i,t} = 0    if |z_{i,t}| < exit (exit position)
        s_{i,t} = s_{i,t-1} otherwise (hold)

    The outer band (|z| > 2) triggers entry; the inner band (|z| < 0.5) triggers exit. Between the two bands, the prior position is continued.

    Assumptions
    _______
    - Signal generates daily and rebalanced once thresholds are crossed
    - The carry-forward logic here handles intra-month position persistence
    - This loop cannot be trivially vectorized: yesterday's influences today's, by intention.
    """
    signals = pd.DataFrame(0, index=zscores.index, columns=zscores.columns,dtype=float)

    for i, date in enumerate(zscores.index):
        z = zscores.iloc[i]

        if i == 0:
            signals.iloc[0] = np.where(z < -entry, 1.0,
                                       np.where(z > entry, -1.0, 0.0))
            continue
        prev = signals.iloc[i - 1].copy()
        new = prev.copy()

        # Exiting when the mispricing reverts
        new[np.abs(z) < exit] = 0.0

        # Entry: new anomaly found
        new[z < -entry] = 1.0
        new[z > entry] = -1.0

        signals.iloc[i] = new
    return signals
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
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
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
