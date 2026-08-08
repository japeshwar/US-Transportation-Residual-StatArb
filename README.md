# US-Transportation-Residual-StatArb

**US Transportation Residual Statistical Arbitrage**

Dollar-neutral residual mean-reversion strategy on a universe of 20 liquid U.S. transportation stocks (airlines, railroads, trucking, and logistics).

## Strategy Overview

- **Universe**: 20 U.S. transportation equities  
- **Residualization**: Rolling regression of each stock against SPY + IYT (iShares U.S. Transportation ETF)  
- **Signal**: Extreme residual z-score mean-reversion (entry threshold = 4.3)  
- **Risk Controls**: Position limits (`top_n = 5`), minimum holding period, turnover budget, and macro regime filter  
- **Portfolio**: Dollar-neutral, inverse-volatility weighted within each side

## Performance (Out-of-Sample: 2022–2026)

| Metric                | Unlevered (Research Result) | 4× Levered (Implementation) |
|-----------------------|-----------------------------|-----------------------------|
| Annualized Return     | 3.50%                       | 14.02%                      |
| Annualized Volatility | 3.07%                       | 12.27%                      |
| **Sharpe Ratio**      | **1.14**                    | **1.14**                    |
| Maximum Drawdown      | –2.67%                      | –10.37%                     |

- Training period: 2015–2019 (parameters selected only on this period)  
- Validation period: 2022–2026 (fully held out)

## Key Implementation Details

- Expanding-window outlier clipping (no look-ahead bias)
- Transaction costs applied on full absolute position changes (no `/2` under-counting)
- Both unlevered and leveraged results are reported for transparency

## Project Structure
