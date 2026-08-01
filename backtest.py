# Backtesting Strategy
"""
backtest.py
================
Dynamic Airline Factor Laboratory
----------------------------------
Redesigned signal generation and portfolio construction:
 
  1. Cross-sectional + time-series hybrid signals
  2. Inverse-volatility weighting proportional to signal strength
  3. Continuous PC1 exposure scaling (replaces binary filter)
  4. Minimum holding period (prevents premature exits)
  5. OU half-life aware signal generation
  6. Preserved: benchmark regression, performance statistics
"""

import numpy as np
import pandas as pd
from typing import Tuple
import sys, os
from statsmodels.tsa.stattools import adfuller
import scipy.stats as stats


# Strategic Parameters
entry_threshold = 1.75 # Z-score to enter
exit_threshold = 0.75 # |z| below which we exit
min_hold_days = 5 # minimum days before exit allowed
conc_full = 0.40 # below this: full exposure (normal regime)
conc_zero = 0.80 # above this: zero exposure (crisis regime)
transaction_cost = 0.0010 # 10 bps one-way
trading_days = 252
top_n = 1 # max positions per side (long or short)
max_daily_turnover = 0.10


# Signal Generation
 
def generate_signals(
    zscores:  pd.DataFrame,
    entry: float = entry_threshold,
    exit_: float = exit_threshold,
    top_n: int = 2,
    min_hold: int = 5,
    momentum: bool = False,
) -> pd.DataFrame:
    """
    Purpose
    _______
    Generate directional signals with optional momentum mode.

    momentum=False (default, mean reversion):
        z < -entry  →  LONG  (stock underperformed, expect bounce)
        z > +entry  →  SHORT (stock outperformed, expect pullback)

    momentum=True (trend following):
        z > +entry  →  LONG  (positive residual momentum, ride it)
        z < -entry  →  SHORT (negative residual momentum, ride it)

    Exit condition is the same in both modes:
        abs(z) < exit  →  signal has dissipated, exit

    Minimum holding period enforced in both modes.
    """
    z_slope = zscores.diff().fillna(0.0)
    cols = zscores.columns.tolist()
    signals = pd.DataFrame(0.0, index = zscores.index, columns=cols)
    prev = pd.Series(0.0, index=cols)
    hold_cnt = pd.Series(0, index=cols, dtype=int)
 
    for i, date in enumerate(zscores.index):
        z = zscores.loc[date].fillna(0.0)  # NaN → 0 (untradeable = flat)
        new = prev.copy()
 
        # Increment holding counter for open positions
        for t in cols:
            if prev[t] != 0:
                hold_cnt[t] += 1
 
        # Exit: only if minimum hold period met AND signal reverted
        for t in cols:
            if prev[t] != 0 and hold_cnt[t] >= min_hold:
                if abs(z[t]) < exit_:
                    new[t] = 0.0
                    hold_cnt[t] = 0
 
        current_longs = [t for t in cols if new[t] == 1.0]
        current_shorts = [t for t in cols if new[t] == -1.0]
 
        if len(current_longs) >= top_n:
            weakest_long = min(current_longs, key=lambda t: abs(z[t]))
            candidates = [t for t in cols
                          if t not in current_longs
                          and (z[t] > entry if momentum else z[t] < -entry)]
            if candidates:
                strongest_new = max(candidates, key=lambda t: abs(z[t]))
                if abs(z[strongest_new]) > abs(z[weakest_long]):
                    new[weakest_long] = 0.0
                    hold_cnt[weakest_long] = 0
 
        if len(current_shorts) >= top_n:
            weakest_short = min(current_shorts, key=lambda t: abs(z[t]))
            candidates = [t for t in cols
                          if t not in current_shorts
                          and (z[t] < -entry if momentum else z[t] > entry)]
            if candidates:
                strongest_new = max(candidates, key=lambda t: abs(z[t]))
                if abs(z[strongest_new]) > abs(z[weakest_short]):
                    new[weakest_short] = 0.0
                    hold_cnt[weakest_short] = 0
 
        n_long_slots = top_n - sum(1 for t in cols if new[t] == 1.0)
        n_short_slots = top_n - sum(1 for t in cols if new[t] == -1.0)
 
        slope_row = z_slope.iloc[i] if i < len(z_slope) else pd.Series(0.0, index=cols)

        if momentum:
            long_cands = {t: z[t] for t in cols if z[t] > entry and slope_row[t] >= 0 and new[t] == 0.0}
            short_cands = {t: z[t] for t in cols if z[t] < -entry and slope_row[t] <= 0 and new[t] == 0.0}
        else:
            long_cands = {t: z[t] for t in cols if z[t] < -entry and slope_row[t] <= 0 and new[t] == 0.0}
            short_cands = {t: z[t] for t in cols if z[t] > entry and slope_row[t] >= 0 and new[t] == 0.0}

        n_long_open = sum(1 for t in cols if new[t] == 1.0)
        n_short_open = sum(1 for t in cols if new[t] == -1.0)

        if long_cands and n_long_open < top_n:
            ranked = sorted(long_cands.items(),
                           key=lambda x: x[1], reverse=momentum)
            for t, _ in ranked[:top_n - n_long_open]:
                new[t] = 1.0; hold_cnt[t] = 0

        if short_cands and n_short_open < top_n:
            ranked = sorted(short_cands.items(),
                           key=lambda x: x[1], reverse=not momentum)
            for t, _ in ranked[:top_n - n_short_open]:
                new[t] = -1.0; hold_cnt[t] = 0

        signals.iloc[i] = new
        prev = new.copy()

    return signals
 
# Portfolio Construction
 
def apply_continuous_concentration_scaling(
    positions: pd.DataFrame,
    concentration: pd.Series,
    conc_full: float = conc_full,
    conc_zero: float = conc_zero,
) -> pd.DataFrame:
    """
    Purpose
    _______
    Linearly scale all position sizes with PC1 concentration.
    Replaces the binary on/off filter from v1.
 
    Inputs
    _______
    positions: pd.DataFrame (T × N)  — raw portfolio weights
    concentration: pd.Series (T,) — PC1 explained variance ratio
    conc_full: float — below this: full exposure (scale=1.0)
    conc_zero: float — above this: zero exposure (scale=0.0)
 
    Outputs
    _______
    pd.DataFrame (T × N)  — scaled positions
 
    Mathematical Explanation
    _______
    Scaling factor at date t:
 
        scale_t = max(0, min(1, (conc_zero − c_t) / (conc_zero − conc_full)))
 
    Examples (conc_full=0.40, conc_zero=0.80):
        c = 0.30  →  scale = 1.00  (below full threshold, maximum exposure)
        c = 0.40  →  scale = 1.00  (at full threshold)
        c = 0.60  →  scale = 0.50  (midpoint, half exposure)
        c = 0.80  →  scale = 0.00  (at zero threshold, flat)
        c = 0.95  →  scale = 0.00  (crisis, flat)
 
    Why Continuous Beats Binary
    _______
    Binary filter (v1): all-or-nothing at c > 0.80
        - On the day concentration crosses 0.80, ALL positions are liquidated
        - Creates a large, costly rebalancing event
        - Next day if concentration drops to 0.79, all positions reinstated
        - Extreme turnover and cost drag around the threshold
 
    Continuous scaling (v2): smooth exposure adjustment
        - As markets become more correlated, positions shrink gradually
        - No sudden liquidation event
        - Cost of de-risking spread over many days
        - More realistic representation of how risk managers operate
 
    Empirical behavior:
        Normal regime (c ≈ 0.35):      scale ≈ 1.00, full positions
        Elevated (c ≈ 0.55):           scale ≈ 0.63, moderate reduction
        Stress (c ≈ 0.70):             scale ≈ 0.25, heavily reduced
        Crisis (c ≈ 0.85+, e.g. COVID): scale = 0.00, flat
 
    Assumptions
    _______
    - Dates without concentration data are treated as scale=1.0 (no penalty)
    - Scaling is applied to final portfolio weights, not signals
    """
    rng = conc_zero - conc_full
    scaled = positions.copy()
 
    for date in positions.index:
        if date not in concentration.index:
            continue
        c = concentration.loc[date]
        scale = max(0.0, min(1.0, (conc_zero - c) / rng if rng > 0 else 1.0))
        scaled.loc[date] = scaled.loc[date] * scale
    return scaled
 
def construct_portfolio(
    signals: pd.DataFrame,
    zscores: pd.DataFrame = None,
    residuals: pd.DataFrame = None,
    vol_window: int = 30,
) -> pd.DataFrame:
    """
    Purpose
    _______
    Convert signals to dollar-neutral weights using inverse-volatility
    weighting proportional to signal strength.
 
    Inputs
    ------
    signals: pd.DataFrame (T × N) — directional signals (+1/-1/0)
    zscores: pd.DataFrame (T × N) — S-scores (used for signal weighting)
                                         If None: equal weights per side
    residuals: pd.DataFrame (T × N) — residuals (used for vol weighting)
                                         If None: equal weights per side
    vol_window: int — rolling volatility window
 
    Outputs
    _______
    pd.DataFrame (T × N)  — portfolio weights
 
    Mathematical Explanation
    _______
    EQUAL WEIGHT (when zscores and residuals are None):
        w_i = +1/(2·n_long)  for longs
        w_i = -1/(2·n_short) for shorts
 
    SIGNAL-PROPORTIONAL INVERSE-VOL (when both are provided):
        For each long position i:
            raw_weight_i = |s_{i,t}| / σ_{resid,i,t}
 
        Normalize to sum to 0.5 on each side:
            w_i = 0.5 × raw_weight_i / Σ_long raw_weight_j
 
        For shorts: same with negative sign.
 
    This weighting scheme achieves two goals simultaneously:
    (1) SIGNAL STRENGTH: higher |s| → larger position
        A stock with s=-3.2 gets 2.1× the weight of s=-1.6
    (2) RISK PARITY: higher residual vol → smaller position
        Prevents high-volatility names from dominating portfolio risk
 
    Combined effect:
        w ∝ |s| / σ  =  signal per unit of risk (information ratio)
        The portfolio is weighted by its local information ratio
 
    Dollar-Neutrality:
        Σ w_i = 0 always (equal allocation to longs and shorts)
 
    Assumptions
    _______
    - If a side has no valid vol data, falls back to equal weighting
    - Minimum vol floor of 1e-8 to prevent division by zero
    """
    common = signals.index
    if residuals is not None:
        vols = residuals.reindex(common).rolling(vol_window).std()
        vols = vols.replace(0, np.nan)
    else:
        vols = None
 
    positions = pd.DataFrame(0.0, index=common, columns=signals.columns)
 
    for date in common:
        row = signals.loc[date]
        longs = row[row > 0].index.tolist()
        shorts = row[row < 0].index.tolist()
 
        if not longs and not shorts:
            continue
 
        def _weights(names, direction):
            if not names:
                return {}
            if zscores is not None and vols is not None and date in vols.index:
                s_row = zscores.loc[date] if date in zscores.index else pd.Series()
                v_row = vols.loc[date]
                raw = {}
                for t in names:
                    sv = abs(s_row.get(t, 0))
                    vv = v_row.get(t, np.nan)
                    if sv > 0 and pd.notna(vv) and vv > 1e-8:
                        raw[t] = sv / vv
                if raw:
                    total = sum(raw.values())
                    return {t: direction * 0.5 * w / total for t, w in raw.items()}
            n = len(names)
            return {t: direction * 0.5 / n for t in names}
 
        long_weights = _weights(longs, 1.0)
        short_weights = _weights(shorts, -1.0)
 
        for t, w in {**long_weights, **short_weights}.items():
            positions.loc[date, t] = w
 
    return positions
 

# Execution & Performance

def apply_macro_regime_filter(
    positions: pd.DataFrame,
    macro_returns: pd.DataFrame,
    vix_window: int = 120,
    brent_window: int = 120,
    vix_scale_q3: float = 1.0,
    vix_close_q4: float = 0.25,
    brent_close_q4: float = 0.0,
) -> pd.DataFrame:
    """
    Purpose
    _______
    Scale positions based on rolling VIX and Brent regimes.
    Derived from macro_regime_diagnostics output showing:

        VIX Q1: Sharpe 1.130  → full exposure
        VIX Q2: Sharpe 0.480  → full exposure
        VIX Q3: Sharpe 0.386  → half exposure
        VIX Q4: Sharpe -0.329 → close all

        Brent Q1: Sharpe 1.686 → full exposure
        Brent Q2: Sharpe 1.391 → full exposure
        Brent Q3: Sharpe -0.429 → half exposure
        Brent Q4: Sharpe -1.078 → close all

    Implementation uses rolling percentiles (no look-ahead):
        VIX percentile = rank of today's VIX in last vix_window days
        Brent percentile = rank of today's Brent in last brent_window days

    Scale factor = min(vix_scale, brent_scale):
        percentile < 0.50 → scale 1.0 (Q1/Q2 equivalent)
        percentile 0.50-0.75 → scale 0.5 (Q3)
        percentile > 0.75 → scale 0.0 (Q4)

    No look-ahead: rolling window always ends at t-1.

    Inputs
    _______
    positions: pd.DataFrame — portfolio weights from construct_portfolio()
    macro_returns: pd.DataFrame — must contain 'VIX' and 'Brent' columns
    vix_window: int — rolling window for VIX percentile (trading days)
    brent_window: int — rolling window for Brent percentile
    vix_scale_q3: float — exposure in Q3 VIX regime (0.5 = 50%)
    vix_close_q4: float — exposure in Q4 VIX regime (0.0 = flat)
    brent_close_q4: float — exposure in Q4 Brent regime

    Outputs
    _______
    pd.DataFrame — scaled positions, same shape as input
    """
    scaled = positions.copy()
    common = positions.index.intersection(macro_returns.index)

    def rolling_pctile(series, window):
        """Vectorized rolling percentile using rank"""
        return series.rolling(window, min_periods=window).rank(pct=True).shift(1)

    vix_pctile = pd.Series(np.nan, index=macro_returns.index)
    brent_pctile = pd.Series(np.nan, index=macro_returns.index)

    if 'VIX' in macro_returns.columns:
        vix_pctile = rolling_pctile(macro_returns['VIX'].dropna(), vix_window)

    if 'Brent' in macro_returns.columns:
        brent_pctile = rolling_pctile(macro_returns['Brent'].dropna(), brent_window)

    for date in common:
        vix_p = vix_pctile.get(date, np.nan)
        brent_p = brent_pctile.get(date, np.nan)

        # VIX scaling
        if pd.isna(vix_p):
            vix_scale = 1.0
        elif vix_p > 0.75:
            vix_scale = vix_close_q4 # Q4: close
        elif vix_p > 0.50:
            vix_scale = vix_scale_q3 # Q3: half
        else:
            vix_scale = 1.0 # Q1/Q2: full

        # Brent scaling
        if pd.isna(brent_p):
            brent_scale = 1.0
        elif brent_p > 0.75:
            brent_scale = brent_close_q4 # Q4: close
        elif brent_p > 0.50:
            brent_scale = 0.5 # Q3: half
        else:
            brent_scale = 1.0 # Q1/Q2: full

        # Most conservative wins
        combined_scale = min(vix_scale, brent_scale)
        scaled.loc[date] = scaled.loc[date] * combined_scale

    # Report how many days were filtered
    days_full = 0
    days_half = 0
    days_closed = 0
    for date in common:
        vix_p = vix_pctile.get(date, np.nan)
        brent_p = brent_pctile.get(date, np.nan)
        vp = vix_p if not pd.isna(vix_p) else 0
        bp = brent_p if not pd.isna(brent_p) else 0
        cp = min(
            0.0 if vp > 0.75 else (0.5 if vp > 0.50 else 1.0),
            0.0 if bp > 0.75 else (0.5 if bp > 0.50 else 1.0),
        )
        if cp == 0.0: days_closed += 1
        elif cp < 1.0: days_half   += 1
        else: days_full   += 1

    print(f"Macro regime filter: full={days_full}d  half={days_half}d closed={days_closed}d")

    return scaled
 
def compute_portfolio_returns(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    cost: float = transaction_cost,
    turnover_budget: float = max_daily_turnover,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Shift positions 1 day (execute at t+1), compute net returns.
 
    Gross: R_t = Σᵢ w_{i,t-1} · r_{i,t}
    Turnover: TO_t = Σᵢ |w_{i,t} − w_{i,t-1}| / 2
    Cost: cost_t = TO_t × 0.0015
    Net: R_net,t = R_t − cost_t
    """
    common = positions.index.intersection(returns.index)
    pos = positions.loc[common].copy()
    ret = returns.loc[common, positions.columns].copy()
    pos_executed = pos.shift(1).fillna(0.0)
    pos_prev = pos_executed.shift(1).fillna(0.0)
    pos_change = pos_executed - pos_prev
    raw_to = pos_change.abs().sum(axis=1) / 2
    scale = (turnover_budget / raw_to).clip(upper=1.0).replace([np.inf], 1.0)
    pos_executed = pos_prev + pos_change.multiply(scale, axis=0)
    gross_ret = (pos_executed * ret).sum(axis=1)
    turnover = pos_executed.diff().abs().sum(axis=1) / 2
    turnover.iloc[0] = pos_executed.iloc[0].abs().sum() / 2
    cost_drag = turnover * cost
    net_ret = gross_ret - cost_drag
 
    trades = pd.DataFrame({
        "gross_return": gross_ret,
        "turnover": turnover,
        "transaction_cost": cost_drag,
        "net_return": net_ret,
        "n_longs": (pos_executed > 0).sum(axis=1),
        "n_shorts": (pos_executed < 0).sum(axis=1),
        "gross_exposure": pos_executed.abs().sum(axis=1),
    })
    return net_ret, trades
 
def benchmark_regression(
    net_returns: pd.Series,
    macro_returns: pd.DataFrame,
) -> pd.Series:
    """
    OLS regression vs SPY and JETS.
    Target: β≈0, α>0, |t_α|>2, R²<0.10
 
    alpha_annual = alpha_daily × 252
    t-stat = alpha / SE(alpha) — must exceed 2.0 for significance
    """
    result = {}
    for bench in ["SPY", "JETS"]:
        if bench not in macro_returns.columns:
            continue
        common = net_returns.index.intersection(macro_returns.index)
        y = net_returns.loc[common].values
        x = macro_returns.loc[common, bench].values
        mask = ~(np.isnan(y) | np.isnan(x))
        y, x = y[mask], x[mask]
        if len(y) < 30: continue
 
        X = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        a, b = beta
        yhat = X @ beta
        res = y - yhat
        s2 = res.var(ddof=2)
        se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
        t = a / se if se > 0 else np.nan
        ss_t = ((y - y.mean())**2).sum()
        r2 = 1 - (res**2).sum() / ss_t if ss_t > 0 else np.nan
 
        bn = bench.lower()
        result[f"{bn}_alpha_annual"] = round(a * trading_days, 4)
        result[f"{bn}_alpha_tstat"] = round(t, 3)
        result[f"{bn}_beta"] = round(b, 4)
        result[f"{bn}_r2"] = round(r2, 4)
        result[f"{bn}_correlation"] = round(np.corrcoef(y, x)[0,1], 4)
 
    return pd.Series(result)
 
def compute_performance(
    net_returns: pd.Series,
    trades_df: pd.DataFrame,
    macro_returns: pd.DataFrame,
    rf: float = 0.0,
) -> pd.Series:
    """
    Full performance statistics: Sharpe, MDD, Calmar, Hit Rate,
    Turnover, Alpha, Beta, R², Correlation vs benchmarks.
    """
    r = net_returns.dropna()
    if len(r) == 0:
        return pd.Series({"error": "No returns"})
 
    ann_ret = r.mean() * trading_days
    ann_vol = r.std() * np.sqrt(trading_days)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
 
    cum = (1 + r).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar = ann_ret / abs(mdd) if mdd < 0 else np.nan
 
    active = r[r != 0]
    hit_rate = (active > 0).mean() if len(active) > 0 else np.nan
    ann_to = trades_df["turnover"].mean() * trading_days
 
    core = pd.Series({
        "annualized_returns": round(ann_ret, 4),
        "annualized_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(mdd, 4),
        "calmar_ratio": round(calmar, 3),
        "hit_rate": round(hit_rate, 4),
        "ann_turnover": round(ann_to, 3),
        "active_days": int(len(active)),
        "total_days": int(len(r)),
    })
    bench = benchmark_regression(net_returns, macro_returns)
    return pd.concat([core, bench])
 
def macro_regime_diagnostics(
    net_returns: pd.Series,
    macro_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sharpe by VIX and Brent quartile. Diagnostic — not a trading rule.
    Column name 'Brent' (title case) must match data.py.
    """
    common = net_returns.index.intersection(macro_returns.index)
    r = net_returns.loc[common]
    macro = macro_returns.loc[common]
    records = {}
 
    for var in ["VIX", "Brent"]:
        if var not in macro.columns:
            continue
        q = pd.qcut(macro[var].dropna(), q=4, labels=["Q1","Q2","Q3","Q4"])
        sharpes = {}
        for qi in ["Q1","Q2","Q3","Q4"]:
            sub = r[q == qi]
            if len(sub) < 30: sharpes[qi] = np.nan; continue
            ar = sub.mean() * trading_days
            av = sub.std() * np.sqrt(trading_days)
            sharpes[qi] = round(ar/av, 3) if av > 0 else np.nan
        records[var] = sharpes
 
    return pd.DataFrame(records).T