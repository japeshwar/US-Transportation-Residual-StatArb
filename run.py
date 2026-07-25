"""
Dynamic Airline Factor Laboratory
_______
Orchestrates the full research pipeline from raw data to output CSVs.

Pipeline
_______
    [1]  Download and clean market data
    [2]  Split into training (2015–2019) and validation (2020–2024) periods
    [3]  Window sensitivity analysis on training period (60, 90, 120, 180 days)
    [4]  Select optimal window; hold fixed for validation
    [5]  Run full factor model on selected window
    [6]  Report variance explained by PC1–PCk (diagnose n_components)
    [7]  Generate signals → reconstruct portfolio
    [8]  Compute PnL with 15 bps transaction costs
    [9]  Evaluate performance: Sharpe, drawdown, alpha/beta vs JETS and SPY
    [10] Macro regime diagnostics (VIX, Brent quartiles)
    [11] Write all output CSVs to outputs/

Train / Test Discipline
_______
    Training  : 2015-01-01 → 2019-12-31
        - Window length selected here (60, 90, 120, 180 days)
        - n_components tested here (1, 2, 3)
        - No other parameters tuned

    Validation: 2022-01-01 → 2026-07-17
        - Selected window applied unchanged
        - All performance statistics reported from this period
"""
import warnings
warnings.filterwarnings("ignore")
import sys, os
sys.path.insert(0, os.path.abspath('..'))
import numpy as np
import pandas as pd
from pathlib import Path

import data as dt
import models as md
import backtest as bt

output_dir = Path("../Outputs")
# Reporting helpers

def _section(title: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"\ {title}")
    print(f"\{'=' * width}")
def _print_performance(perf: pd.Series, label: str) -> None:
    pct_keys = {"annualized_returns", "annualized_volatility",
               "max_drawdown", "hit_rate", "ann_turnover",
               "jets_alpha_annual", "spy_alpha_annual"}
    print(f"\n - {label} -")
    for k, v in perf.items():
        if isinstance(v, float):
            fmt = f"{v:.2%}" if k in pct_keys else f"{v:.4f}"
        else:
            fmt = str(v)
        print(f" {k:<35s}: {fmt}")
def _select_window(
    train_returns: pd.Series,
    n_components: int,
) -> int:
    """
    Run window sensitivity on the training period.
    Select the window with the highest in-sample Sharpe ratio.
    Report all candidates so the choice is transparent.
    """
    _section("Window Sensitivity - Training Period")
    train_ret = train_returns

    print(f"\n Testing windows: {md.window_candidates} days")
    print(f" n_components = {n_components}")
    print(f" Period: {train_ret.index[0].date()} -> {train_ret.index[-1].date()}\n")

    sensitivity = md.run_window_sensitivity(train_ret, md.window_candidates, n_components)

    sharpes = {}
    for w, (residuals, zscores) in sensitivity.items():
        signals = bt.generate_signals(zscores)
        positions = bt.construct_portfolio(signals)
        pc1 = md.compute_pc1_concentration(cov)
        pc1 = pc1.reindex(positions.index)
        positions.loc[pc1 > 0.80] = 0.0
        
        # Need macro returns aligned to training period for eval; simple sharpe
        common = positions.index.intersection(train_ret.index)
        pos_x = positions.loc[common]
        ret_x = train_ret.loc[common]
        pos_exec = pos_x.shift(1).fillna(0.0)
        gross = (pos_exec * ret_x).sum(axis=1)
        turnover = pos_exec.diff().abs().sum(axis=1) / 2
        net = gross - turnover * bt.transaction_cost
        net = net.dropna()

        ann_r = net.mean() * bt.trading_days
        ann_v = net.std() * np.sqrt(bt.trading_days)
        sr = ann_r / ann_v if ann_v > 0 else np.nan
        sharpes[w] = sr

        print(f" Window {w:3d}d -> Sharpe: {sr:+.3f}"
            f" | Ann. Return: {ann_r: .2%}"
            f" | Ann. Vol {ann_v: .2%}")
    best_window = max(sharpes,key=lambda k: sharpes[k] if not np.isnan(sharpes[k]) else -99)
    print(f"\n -> Selected window: {best_window} days (best training Sharpe)")
    return best_window
def _report_variance_explained(expl_var_df: pd.DataFrame, period_label: str) -> None:
    """
    Print average variance explained by each PC.
    """
    avg_expl = expl_var_df.mean()
    cumsum = avg_expl.cumsum()
    print(f"\n Explained Variance ({period_label}):")
    for col, val in avg_expl.items():
        i = int(col.replace("PC", "")) - 1
        print(f" {col}: {val:.1%} (cumulative: {cumsum.iloc[i]:.1%})")
    print(f"\n Note: if PC1 alone explains > 65%, consider n_components = 1.")
    print(f" If PC3 explains <5%, it may be noise - use n_components = 2.")
# Main Pipeline

def main() -> dict:
    output_dir.mkdir(exist_ok=True)

    # 01 Data
    _section("Data Loading")
    airline_returns, macro_returns, prices = dt.load_data()

    # 02 Train / Test Split
    train_mask = airline_returns.index <= dt.train_end
    test_mask = airline_returns.index >= dt.test_start

    train_airlines = airline_returns[train_mask]
    test_airlines = airline_returns[test_mask]
    train_macro = macro_returns[train_mask]
    test_macro = macro_returns[test_mask]

    _section("Train / Test Split")
    print(f" Training: {train_airlines.index[0].date()} -> "
          f"{train_airlines.index[-1].date()} ({len(train_airlines)} days)")
    print(f" Validation: {test_airlines.index[0].date()} -> "
          f"{test_airlines.index[-1].date()} ({len(test_airlines)} days)")

    # 03 Window Selection on Training Period
    best_window = _select_window(train_airlines, md.n_components)

    # 04 Full Factor Model on Training Period
    _section("Factor Model - Training Period")
    cov_train = md.rolling_covariances(train_airlines, window = best_window)
    sys_train, resid_train, evals_train, evecs_train, expl_train = \
    md.compute_factor_model(train_airlines, cov_train, md.n_components)
    factor_ret_train = md.compute_factor_returns(train_airlines, cov_train, md.n_components)
    conc_train = md.compute_pc1_concentration(cov_train)

    _report_variance_explained(expl_train, "Training")

    zscores_train = md.compute_rolling_zscore(resid_train).dropna()
    sig_train = bt.generate_signals(zscores_train)
    pos_train = bt.construct_portfolio(sig_train)
    net_train, trades_train = bt.compute_portfolio_returns(pos_train, train_airlines)
    perf_train = bt.compute_performance(net_train, trades_train, train_macro)

    _print_performance(perf_train, f"Training Results (window = {best_window}d)")

    # 05 Full Factor Model on Validation Period
    _section("Factor Model - Validation Period (Out-of-Sample)")
    cov_test = md.rolling_covariances(test_airlines, window = best_window)
    sys_test, resid_test, evals_test, evecs_test, expl_test = \
    md.compute_factor_model(test_airlines, cov_test, md.n_components)
    factor_ret_test = md.compute_factor_returns(test_airlines, cov_test, md.n_components)
    conc_test = md.compute_pc1_concentration(cov_test)

    _report_variance_explained(expl_test, "Validation")

    zscores_test = md.compute_rolling_zscore(resid_test).dropna()
    sig_test = bt.generate_signals(zscores_test)
    pos_test = bt.construct_portfolio(sig_test)
    net_test, trades_test = bt.compute_portfolio_returns(pos_test, test_airlines)
    perf_test = bt.compute_performance(net_test, trades_test, test_macro)

    _print_performance(perf_test, "Validation Results (Out-of-Sample)")

    # 06 Macro Regime Diagnostics
    _section("Macro Regime Diagnostics")
    print("\n [Training Period]")
    diag_train = bt.macro_regime_diagnostics(net_train, train_macro)
    print(diag_train.to_string())

    print("\n [Validation Period]")
    diag_test = bt.macro_regime_diagnostics(net_test, test_macro)
    print(diag_test.to_string())
    print("\n Note: These are diagnostics, not trading rules.")
    print(" If Sharpe deteriorates with VIX, consider")
    print(" a continuous vol-scaling adjustments (not a hard threshold).")

    # 07 PC1 Concentration Diagnostic
    _section("PC1 Concentration - Diagnostic")
    print(f"\n Training Period:")
    print(f" Mean Concentration: {conc_train.mean():.3f}")
    print(f" STD Concentration: {conc_train.std():.3f}")
    print(f" Max Concentration: {conc_train.max():.3f} (stress signal)")
    print(f"\n Validation Period:")
    print(f" Mean Concentration: {conc_test.mean():.3f}")
    print(f" STD Concentration: {conc_test.std():.3f}")
    print(f" Max Concentration: {conc_test.max():.3f}")
    print(f"\n Interpretation: Concentration > 0.65 indicates a stress")
    print(f" regime where all airlines correlate strongly (i.e., COVID).")
    print(f" Use as diagnostic first; test as regime filter out-of-sample.")

    # 08 Save Outputs
    _section("Saving Outputs")

    # Performance: training + validation
    perf_combined = pd.concat(
        [perf_train.rename("training"), perf_test.rename("validation")],
        axis = 1
    )
    perf_combined.to_csv(output_dir / "performance.csv")

    # Validation period outputs
    sig_test.to_csv(output_dir / "signals.csv")
    trades_test.to_csv(output_dir / "trades.csv")
    resid_test.to_csv(output_dir / "residuals.csv")
    factor_ret_test.to_csv(output_dir / "factor_returns.csv")
    evals_test.to_csv(output_dir / "eigenvalues.csv")
    evecs_test.to_csv(output_dir / "eigenvectors.csv")
    expl_test.to_csv(output_dir / "explained_variance.csv")

    # Diagnostics
    conc_combined = pd.concat([conc_train, conc_test]).rename("pc1_concentration")
    conc_combined.to_csv(output_dir / "pc1_concentration.csv")

    diag_combined = pd.concat(
        [diag_train.add_suffix("_train"), diag_test.add_suffix("_test")],
        axis=1
    )
    diag_combined.to_csv(output_dir / "macro_diagnostics.csv")

    print(f"\n All outputs saved to {output_dir.resolve()}/")

    _section("Pipeline Complete")

    return {
       "perf_train": perf_train,
        "perf_test": perf_test,
        "net_train": net_train,
        "net_test": net_test,
        "residuals_train": resid_train,
        "residuals_test": resid_test,
        "zscores_test": zscores_test,
        "eigenvalues_test":  evals_test,
        "expl_var_test": expl_test,
        "best_window": best_window,
        "diag_train": diag_train,
        "diag_test": diag_test,
    } 
if __name__ == "__main__":
    results = main()