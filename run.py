"""
run.py
===========
Dynamic Transportation Factor Laboratory
Simple Residual vs SPY + IYT
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
    print(f"\n ── {label} ──")
    for k, v in perf.items():
        if isinstance(v, float):
            if "return" in k or "drawdown" in k or "turnover" in k or "rate" in k:
                fmt = f"{v:.2%}"
            else:
                fmt = f"{v:.4f}"
        else:
            fmt = str(v)
        print(f" {k:<35s}: {fmt}")

def main() -> dict:
    output_dir.mkdir(exist_ok = True)

    # =========================================================
    # 1. Data
    # =========================================================
    _section("1 Data Loading")
    stocks_returns, macro_returns, prices = dt.load_data()

    # =========================================================
    # 2. Train / Test Split
    # =========================================================
    _section("2 Train / Test Split")
    train_stocks = stocks_returns[stocks_returns.index <= dt.train_end]
    test_stocks = stocks_returns[stocks_returns.index >= dt.test_start]
    train_macro = macro_returns[macro_returns.index <= dt.train_end]
    test_macro = macro_returns[macro_returns.index >= dt.test_start]

    print(f" Training: {train_stocks.index[0].date()} → {train_stocks.index[-1].date()} ({len(train_stocks)} days)")
    print(f" Validation: {test_stocks.index[0].date()} → {test_stocks.index[-1].date()} ({len(test_stocks)} days)")

    zscore_window = 45
    residual_window = 90

    # =========================================================
    # 3. Full Pipeline — Training
    # =========================================================
    _section("3 Full Pipeline — Training Period")

    factor_returns_train = train_macro[['SPY', 'IYT']]
    residuals_train = md.compute_simple_residuals(train_stocks, factor_returns_train, window = residual_window)
    zscores_train = md.compute_rolling_zscore(residuals_train, window = zscore_window).dropna()

    signals_train = bt.generate_signals(
        zscores_train,
        top_n = bt.top_n,
        exit_ = bt.exit_threshold,
        min_hold = bt.min_hold_days,
        momentum = False
    )

    positions_train = bt.construct_portfolio(signals_train, zscores = zscores_train, residuals = residuals_train)
    positions_train = bt.apply_macro_regime_filter(positions_train, train_macro)

    net_train, trades_train = bt.compute_portfolio_returns(positions_train, train_stocks)
    perf_train = bt.compute_performance(net_train, trades_train, train_macro)
    _print_perf(perf_train, "Training")

    # =========================================================
    # 4. Full Pipeline — Validation
    # =========================================================
    _section("4 Full Pipeline — Validation (Out-of-Sample)")

    factor_returns_test = test_macro[['SPY', 'IYT']]
    residuals_test = md.compute_simple_residuals(test_stocks, factor_returns_test, window = residual_window)
    zscores_test = md.compute_rolling_zscore(residuals_test, window = zscore_window).dropna()

    signals_test = bt.generate_signals(
        zscores_test,
        top_n = bt.top_n,
        exit_ = bt.exit_threshold,
        min_hold = bt.min_hold_days,
        momentum = False
    )

    positions_test = bt.construct_portfolio(signals_test, zscores = zscores_test, residuals = residuals_test)
    positions_test = bt.apply_macro_regime_filter(positions_test, test_macro)

    net_test, trades_test = bt.compute_portfolio_returns(positions_test, test_stocks)
    perf_test = bt.compute_performance(net_test, trades_test, test_macro)
    _print_perf(perf_test, "Validation (Out-of-Sample)")

    # =========================================================
    # 5. Plots
    # =========================================================
    _section("5 Plots")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    (1 + net_train).cumprod().plot(ax = axes[0], color = 'steelblue',
                                   label = f"Training SR = {perf_train['sharpe_ratio']:.2f}", linewidth = 1.5)
    (1 + net_test).cumprod().plot(ax = axes[0], color='darkgreen',
                                  label=f"Validation SR = {perf_test['sharpe_ratio']:.2f}", linewidth = 1.5)
    axes[0].axhline(1.0, color='black', linestyle='--', linewidth = 0.8)
    axes[0].axvline(pd.Timestamp(dt.test_start), color = 'red', linestyle = ':', label = 'Train/Test Split')
    axes[0].set_title("Cumulative Returns — Net of Costs")
    axes[0].legend()
    axes[0].grid(alpha = 0.3)

    cum_test = (1 + net_test.dropna()).cumprod()
    dd_test = (cum_test - cum_test.cummax()) / cum_test.cummax()
    dd_test.plot(ax = axes[1], color='crimson', linewidth = 1.0)
    axes[1].fill_between(dd_test.index, dd_test, 0, alpha=0.3, color='crimson')
    axes[1].set_title(f"Validation Drawdown (MDD: {perf_test['max_drawdown']:.1%})")
    axes[1].grid(alpha = 0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "strategy_overview.png", dpi=150)
    plt.close()
    print("Saved strategy_overview.png")

    # =========================================================
    # 6. Save Outputs
    # =========================================================
    _section("6 Saving Outputs")

    perf_both = pd.concat([perf_train.rename("Training"), perf_test.rename("Validation")], axis = 1)

    prices.to_csv(output_dir / "prices.csv")
    stocks_returns.to_csv(output_dir / "stocks_returns.csv")
    macro_returns.to_csv(output_dir / "macro_returns.csv")
    perf_both.to_csv(output_dir / "performance.csv")
    positions_test.to_csv(output_dir / "positions.csv")
    trades_test.to_csv(output_dir / "trades.csv")
    residuals_test.to_csv(output_dir / "residuals.csv")
    zscores_test.to_csv(output_dir / "zscores.csv")
    net_test.to_frame("net_return").to_csv(output_dir / "net_returns.csv")

    print(f"All outputs saved → {output_dir.resolve()}/")

    _section("Pipeline Complete")

    return {
        "perf_train": perf_train,
        "perf_test": perf_test,
        "net_train": net_train,
        "net_test": net_test,
        "zscore_window": zscore_window,
    }

if __name__ == "__main__":
    results = main()