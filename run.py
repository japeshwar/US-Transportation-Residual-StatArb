"""
run.py
===========
Dynamic Transportation Factor Laboratory
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import data as dt
import models as md
import backtest as bt

output_dir = Path("../Outputs")

def _section(title: str) -> None:
    print(f"\n{'═'*62}\n {title}\n{'═'*62}")

def _print_perf(perf: pd.Series, label: str) -> None:
    pct = {"annualized_return","annualized_volatility","max_drawdown",
           "hit_rate","ann_turnover","spy_alpha_annual"}
    print(f"\n  ── {label} ──")
    for k, v in perf.items():
        if isinstance(v, float):
            fmt = f"{v:.2%}" if k in pct else f"{v:.4f}"
        else:
            fmt = str(v)
        print(f" {k:<35s}: {fmt}")

def main() -> dict:
    output_dir.mkdir(exist_ok=True)
 
    # Data
    _section("1 Data Loading")
    stocks_returns, macro_returns, prices = dt.load_data()
 
    # Train / Test Split
    _section("2 Train / Test Split")
    train_stocks = stocks_returns[stocks_returns.index <= dt.train_end]
    test_stocks = stocks_returns[stocks_returns.index >= dt.test_start]
    train_macro = macro_returns[macro_returns.index <= dt.train_end]
    test_macro = macro_returns[macro_returns.index >= dt.test_start]
    print(f" Training: {train_stocks.index[0].date()} → {train_stocks.index[-1].date()}"
          f" ({len(train_stocks)} days)")
    print(f" Validation: {test_stocks.index[0].date()} → {test_stocks.index[-1].date()}"
          f" ({len(test_stocks)} days)")
 
    # Training Diagnostics
    _section("3 Training Diagnostics")
 
    # Use LW covariance for diagnostics
    cov_diag = md.rolling_covariances(train_stocks, window = md.lookback,
                                      use_ledoit_wolf = True)
    _, resid_diag, _, _, expl_diag = md.compute_factor_model(
        train_stocks, cov_diag, md.n_components
    )
 
    print(f"\n Avg explained variance (training):")
    for col, val in expl_diag.mean().items():
        print(f" {col}: {val:.1%}")
 
    brent_lag = md.compute_brent_leadlag(resid_diag, train_macro, max_lag = 21)
    if len(brent_lag) > 0:
        brent_lag.to_csv(output_dir / "brent_leadlag.csv")

    zscore_window = 45
    print(f"Z-score window: {zscore_window} days")

    # Window Selection on Training Period
    _section("4 Window Sensitivity (Training Period)")
    print(f" Candidates: {md.window_candidates} days")
 
    best_sr = -np.inf
    best_win = md.lookback
 
    for w in md.window_candidates:
        print(f" Window {w:3d}d ... ", end="", flush=True)
        cov_w = md.rolling_covariances(train_stocks, window = w, use_ledoit_wolf = True)
        
        sys_w, res_w, eval_w, evec_w, expl_w = md.compute_factor_model(train_stocks, cov_w, md.n_components)
        zs_w = md.compute_rolling_zscore(res_w, window = zscore_window).dropna()
        conc_w = md.compute_pc1_concentration(cov_w)
        
        if len(zs_w) == 0:
            print("no signal"); continue
 
        sig_w = bt.generate_signals(zs_w, top_n = bt.top_n, exit_ = bt.exit_threshold,
                                      min_hold = bt.min_hold_days, momentum = True)
        pos_w = bt.construct_portfolio(sig_w, zscores = zs_w, residuals = res_w)
        pos_ws = bt.apply_continuous_concentration_scaling(pos_w, conc_w)
        net_w, trades_w = bt.compute_portfolio_returns(pos_ws, train_stocks)
        net_w = net_w.dropna()
 
        if len(net_w) < 30:
            print("insufficient data"); continue
 
        ar = net_w.mean() * bt.trading_days
        av = net_w.std() * np.sqrt(bt.trading_days)
        sr = ar / av if av > 0 else np.nan
        to = trades_w['turnover'].mean() * bt.trading_days
        print(f"Sharpe: {sr:+.3f} | Ann.Ret: {ar:.2%} | Turnover: {to:.2f}")
 
        if pd.notna(sr) and sr > best_sr:
            best_sr  = sr
            best_win = w
 
    print(f"\n Best window: {best_win} days (training Sharpe: {best_sr:+.3f})")
 
    # Full Pipeline — Training
    _section("5 Full Pipeline — Training Period")
    cov_tr  = md.rolling_covariances(train_stocks, window = best_win, use_ledoit_wolf = True)
    sys_tr, res_tr, eval_tr, evec_tr, expl_tr = md.compute_factor_model(
        train_stocks, cov_tr, md.n_components
    )
    conc_tr = md.compute_pc1_concentration(cov_tr)
    fr_tr = md.compute_factor_returns(train_stocks, cov_tr, md.n_components)
    zs_tr = md.compute_rolling_zscore(res_tr, window = zscore_window).dropna()
 
    sig_tr = bt.generate_signals(zs_tr, top_n = bt.top_n, exit_ = bt.exit_threshold, min_hold = bt.min_hold_days, momentum = True)
    pos_tr = bt.construct_portfolio(sig_tr, zscores = zs_tr, residuals = res_tr)
    pos_trs = bt.apply_continuous_concentration_scaling(pos_tr, conc_tr)
    pos_trs = bt.apply_macro_regime_filter(pos_trs, train_macro)
    net_tr, trades_tr = bt.compute_portfolio_returns(pos_trs, train_stocks)
    perf_tr = bt.compute_performance(net_tr, trades_tr, train_macro)
    _print_perf(perf_tr, f"Training (window = {best_win}d)")
 
    # Full Pipeline — Validation
    _section("6 Full Pipeline — Validation (Out-of-Sample)")
    cov_te = md.rolling_covariances(test_stocks, window = best_win, use_ledoit_wolf = True)
    sys_te, res_te, eval_te, evec_te, expl_te = md.compute_factor_model(
        test_stocks, cov_te, md.n_components
    )
    conc_te = md.compute_pc1_concentration(cov_te)
    fr_te = md.compute_factor_returns(test_stocks, cov_te, md.n_components)
    zs_te = md.compute_rolling_zscore(res_te, window = zscore_window).dropna()
 
    sig_te = bt.generate_signals(zs_te, top_n = bt.top_n, exit_ = bt.exit_threshold, min_hold = bt.min_hold_days, momentum = True)
    pos_te = bt.construct_portfolio(sig_te, zscores = zs_te, residuals = res_te)
    pos_tes = bt.apply_continuous_concentration_scaling(pos_te, conc_te)
    pos_tes = bt.apply_macro_regime_filter(pos_tes, test_macro)
    net_te, trades_te = bt.compute_portfolio_returns(pos_tes, test_stocks)
    perf_te = bt.compute_performance(net_te, trades_te, test_macro)
    _print_perf(perf_te, "Validation (Out-of-Sample)")
 
    # Macro Diagnostics
    _section("7 Macro Regime Diagnostics — Validation")
    diag = bt.macro_regime_diagnostics(net_te, test_macro)
    print(diag.to_string())
 
    # Plots
    _section("8 Plots")
 
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))
 
    # Cumulative returns
    (1 + net_tr).cumprod().plot(ax = axes[0], color = 'steelblue',
        label = f"Training SR = {perf_tr['sharpe_ratio']:.2f}", linewidth = 1.5)
    (1 + net_te).cumprod().plot(ax = axes[0], color = 'darkgreen',
        label = f"Validation SR = {perf_te['sharpe_ratio']:.2f}", linewidth = 1.5)
    axes[0].axhline(1.0, color = 'black', linestyle = '--', linewidth = 0.8)
    axes[0].axvline(pd.Timestamp(dt.test_start), color = 'red',
                    linestyle = ':', linewidth = 1.0, label = 'Train/Test Split')
    axes[0].set_title("Cumulative Returns — Net of 10 bps")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend()
 
    # PC1 concentration
    conc_all = pd.concat([conc_tr, conc_te])
    conc_all.plot(ax = axes[1], color = 'tomato', linewidth = 1.0, alpha = 0.8)
    axes[1].axhline(bt.conc_zero, color = 'red', linestyle = '--',
                    label = f"Zero exposure ({bt.conc_zero:.0%})")
    axes[1].axhline(bt.conc_full, color = 'orange', linestyle = '--',
                    label = f"Full exposure ({bt.conc_full:.0%})")
    axes[1].set_title("PC1 Concentration — Continuous Exposure Scaling")
    axes[1].set_ylabel("λ₁ / Σλᵢ")
    axes[1].legend()
 
    # Drawdown
    cum_te = (1 + net_te.dropna()).cumprod()
    dd_te = (cum_te - cum_te.cummax()) / cum_te.cummax()
    dd_te.plot(ax = axes[2], color = 'crimson', linewidth = 1.0)
    axes[2].fill_between(dd_te.index, dd_te, 0, alpha = 0.3, color = 'crimson')
    axes[2].set_title(f"Validation Drawdown (MDD: {perf_te['max_drawdown']:.1%})")
    axes[2].set_ylabel("Drawdown")
 
    plt.tight_layout()
    plt.savefig(output_dir / "strategy_overview.png", dpi = 150)
    plt.close()
    print(f"Saved strategy_overview.png")
 
    # Save CSVs
    _section("9 Saving Outputs")
    perf_both = pd.concat(
        [perf_tr.rename("training"), perf_te.rename("validation")], axis = 1
    )
    prices.to_csv(output_dir / "prices.csv")
    stocks_returns.to_csv(output_dir / "stocks_returns.csv")
    macro_returns.to_csv(output_dir / "macro_returns.csv")
    perf_both.to_csv(output_dir / "performance.csv")
    pos_tes.to_csv(output_dir / "signals.csv")
    trades_te.to_csv(output_dir / "trades.csv")
    res_te.to_csv(output_dir / "residuals.csv")
    zs_te.to_csv(output_dir / "zscores.csv")
    fr_te.to_csv(output_dir / "factor_returns.csv")
    eval_te.to_csv(output_dir / "eigenvalues.csv")
    evec_te.to_csv(output_dir / "eigenvectors.csv")
    expl_te.to_csv(output_dir / "explained_variance.csv")
    conc_all.to_csv(output_dir / "pc1_concentration.csv")
    net_te.to_frame("net_return").to_csv(output_dir / "net_returns.csv")
    print(f"All outputs saved → {output_dir.resolve()}/")
 
    _section("Pipeline Complete")
    return dict(
        perf_train = perf_tr, perf_test = perf_te,
        net_train = net_tr, net_test = net_te,
        best_window = best_win,
        zscore_window = zscore_window,
    )
 
if __name__ == "__main__":
    results = main()