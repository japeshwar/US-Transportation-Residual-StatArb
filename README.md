# US-Transportation-Residual-StatArb

**US Transportation Residual Statistical Arbitrage**

Dollar-neutral residual mean-reversion strategy on a universe of 20 liquid U.S. transportation stocks spanning airlines, railroads, trucking, and logistics.

## Strategy Overview

- **Universe**: 20 U.S. transportation equities  
- **Residualization**: Rolling regression of each stock against SPY + IYT (iShares U.S. Transportation ETF)  
- **Signal**: Extreme residual z-score mean-reversion (entry threshold = 4.3)  
- **Risk Controls**: Position limits (`top_n = 5`), minimum holding period, turnover budget, and macro regime filter  
- **Portfolio**: Dollar-neutral, inverse-volatility weighted within each side

## Performance (Validation Period)

| Metric                    | Validation (2022–2026) |
|---------------------------|------------------------|
| Annualized Return         | 14.07%                 |
| Annualized Volatility     | 12.38%                 |
| **Sharpe Ratio**          | **1.14**               |
| Maximum Drawdown          | –10.53%                |
| Calmar Ratio              | 1.34                   |
| Annual Turnover           | ~11.09×                |
| SPY Alpha (t-stat)        | 2.21                   |

Training period: 2015–2019  
Validation period: 2022–2026  

## Project Structure
data.py - Data download & residual preparation
models.py - Simple residual & z-score functions
backtest.py - Signal generation, portfolio construction, performance
run.py - Full train/validation pipeline
Notebooks - Research & diagnostic notebooks
Outputs - Saved results and charts
