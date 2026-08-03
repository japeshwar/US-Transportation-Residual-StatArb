# Backtesting Strategy
"""
backtest.py
================
Dynamic Transportation Factor Laboratory
"""

import numpy as np
import pandas as pd
from typing import Tuple
import sys, os
from statsmodels.tsa.stattools import adfuller
import scipy.stats as stats


# Strategic Parameters
entry_threshold = 4.30
exit_threshold = 0.5
min_hold_days = 5
top_n = 5
max_daily_turnover = 0.06
transaction_cost = 0.0010
trading_days = 252


# Signal Generation
 
def generate_signals(
    zscores: pd.DataFrame,
    entry: float = entry_threshold,
    exit_: float = exit_threshold,
    top_n: int = top_n,
    min_hold: int = min_hold_days,
    momentum: bool = False,
) -> pd.DataFrame:
    """
    Clean mean-reversion / momentum signal generator.
    """
    cols = zscores.columns.tolist()
    signals = pd.DataFrame(0.0, index=zscores.index, columns=cols)
    
    prev = pd.Series(0.0, index=cols)
    hold_cnt = pd.Series(0, index=cols, dtype=int)
    
    for i, date in enumerate(zscores.index):
        z = zscores.loc[date].fillna(0.0)
        new = prev.copy()
        
        # Update holding counters
        for t in cols:
            if prev[t] != 0:
                hold_cnt[t] += 1
        
        # Exit logic
        for t in cols:
            if prev[t] != 0 and hold_cnt[t] >= min_hold:
                if abs(z[t]) < exit_:
                    new[t] = 0.0
                    hold_cnt[t] = 0
        
        # Entry logic
        if momentum:
            long_cands  = {t: z[t] for t in cols if z[t] > entry and new[t] == 0.0}
            short_cands = {t: z[t] for t in cols if z[t] < -entry and new[t] == 0.0}
        else:
            long_cands  = {t: z[t] for t in cols if z[t] < -entry and new[t] == 0.0}
            short_cands = {t: z[t] for t in cols if z[t] > entry and new[t] == 0.0}
        
        # Fill available slots
        n_long_open  = sum(1 for t in cols if new[t] == 1.0)
        n_short_open = sum(1 for t in cols if new[t] == -1.0)
        
        if long_cands and n_long_open < top_n:
            ranked = sorted(long_cands.items(), key=lambda x: x[1]) # most negative first
            for t, _ in ranked[:top_n - n_long_open]:
                new[t] = 1.0
                hold_cnt[t] = 0
                
        if short_cands and n_short_open < top_n:
            ranked = sorted(short_cands.items(), key=lambda x: x[1], reverse=True) # most positive first
            for t, _ in ranked[:top_n - n_short_open]:
                new[t] = -1.0
                hold_cnt[t] = 0
        
        signals.iloc[i] = new
        prev = new.copy()
    
    return signals
 
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
 
    Assumptions
    _______
    - If a side has no valid vol data, falls back to equal weighting
    """
    common = signals.index
    if residuals is not None:
        vols = residuals.reindex(common).rolling(vol_window).std()
        vols = vols.replace(0, np.nan)
    else:
        vols = None
 
    positions = pd.DataFrame(0.0, index = common, columns = signals.columns)
 
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
 
def apply_volatility_targeting(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    target_vol: float = 0.12, # 12% annualized target
    vol_window: int = 30,
    max_leverage: float = 2.0
) -> pd.DataFrame:
    """
    Scale the entire portfolio so realized residual volatility stays near target.
    """
    # Portfolio returns before targeting
    port_ret = (positions.shift(1) * returns).sum(axis=1)
    
    realized_vol = port_ret.rolling(vol_window).std() * np.sqrt(252)
    leverage = (target_vol / realized_vol).clip(upper = max_leverage).fillna(1.0)
    leverage = leverage.shift(1)
    
    scaled = positions.multiply(leverage, axis=0)
    return scaled

# Execution & Performance

def apply_macro_regime_filter(
    positions: pd.DataFrame,
    macro_returns: pd.DataFrame,
    vix_window: int = 90,
    brent_window: int = 90,
) -> pd.DataFrame:
    """
    Purpose
    _______
    Scale positions based on rolling VIX and Brent regimes.

    Outputs
    _______
    pd.DataFrame — scaled positions, same shape as input
    """
    scaled = positions.copy()
    common = positions.index.intersection(macro_returns.index)

    def rolling_pctile(series, window):
        return series.rolling(window, min_periods = window // 2).rank(pct = True).shift(1)

    vix_pct = rolling_pctile(macro_returns['VIX'], vix_window) if 'VIX' in macro_returns.columns else pd.Series(0.5, index = macro_returns.index)
    brent_pct = rolling_pctile(macro_returns['Brent'], brent_window) if 'Brent' in macro_returns.columns else pd.Series(0.5, index = macro_returns.index)

    for date in common:
        v = vix_pct.get(date, 0.5)
        b = brent_pct.get(date, 0.5)
        
        # Continuous scale: 1.0 at low stress → 0.0 at extreme stress
        vix_scale = np.clip(1.5 - 1.5 * v, 0.0, 1.0) # full size until ~33rd pct, then linear down
        brent_scale = np.clip(1.4 - 1.4 * b, 0.0, 1.0)
        
        scale = min(vix_scale, brent_scale)
        scaled.loc[date] = scaled.loc[date] * scale

    return scaled
 
def compute_portfolio_returns(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    cost: float = transaction_cost,
    turnover_budget: float = max_daily_turnover,
) -> Tuple[pd.Series, pd.DataFrame]:
    common = positions.index.intersection(returns.index)
    pos = positions.loc[common].copy()
    ret = returns.loc[common, positions.columns].copy()
    pos_executed = pos.shift(1).fillna(0.0)
    pos_prev = pos_executed.shift(1).fillna(0.0)
    pos_change = pos_executed - pos_prev
    raw_to = pos_change.abs().sum(axis = 1) / 2
    scale = (turnover_budget / raw_to).clip(upper = 1.0).replace([np.inf], 1.0)
    pos_executed = pos_prev + pos_change.multiply(scale, axis = 0)
    gross_ret = (pos_executed * ret).sum(axis = 1)
    turnover = pos_executed.diff().abs().sum(axis = 1) / 2
    turnover.iloc[0] = pos_executed.iloc[0].abs().sum() / 2
    cost_drag = turnover * cost
    net_ret = gross_ret - cost_drag
 
    trades = pd.DataFrame({
        "gross_return": gross_ret,
        "turnover": turnover,
        "transaction_cost": cost_drag,
        "net_return": net_ret,
        "n_longs": (pos_executed > 0).sum(axis = 1),
        "n_shorts": (pos_executed < 0).sum(axis = 1),
        "gross_exposure": pos_executed.abs().sum(axis = 1),
    })
    return net_ret, trades
 
def benchmark_regression(
    net_returns: pd.Series,
    macro_returns: pd.DataFrame,
) -> pd.Series:
    result = {}
    for bench in ["SPY", "IYT"]:
        if bench not in macro_returns.columns:
            continue
        common = net_returns.index.intersection(macro_returns.index)
        y = net_returns.loc[common].values
        x = macro_returns.loc[common, bench].values
        mask = ~(np.isnan(y) | np.isnan(x))
        y, x = y[mask], x[mask]
        if len(y) < 30: continue
 
        X = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(X, y, rcond = None)[0]
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
    common = net_returns.index.intersection(macro_returns.index)
    r = net_returns.loc[common]
    macro = macro_returns.loc[common]
    records = {}
 
    for var in ["VIX", "Brent"]:
        if var not in macro.columns:
            continue
        q = pd.qcut(macro[var].dropna(), q = 4, labels = ["Q1","Q2","Q3","Q4"])
        sharpes = {}
        for qi in ["Q1","Q2","Q3","Q4"]:
            sub = r[q == qi]
            if len(sub) < 30: sharpes[qi] = np.nan; continue
            ar = sub.mean() * trading_days
            av = sub.std() * np.sqrt(trading_days)
            sharpes[qi] = round(ar/av, 3) if av > 0 else np.nan
        records[var] = sharpes
 
    return pd.DataFrame(records).T