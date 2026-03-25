# Architecture

## Project Structure

```
Task2Solution/
  main_analysis.py              Entry point. Full pipeline.
  config.py                     Paths and risk-free rate.
  Options_Trading_Analysis.ipynb  Notebook version of main_analysis.py.
  data/
    loader.py                   Load CME parquet/CSV with column mapping.
    preprocessor.py             Implied vol, Greeks, feature engineering.
    synthetic_generator.py      Baseline data generator (fallback when parquet absent).
    oi_weighted_futures.csv     Futures data (included).
  models/
    black_scholes.py            BS pricing. Greeks to 3rd order.
    garch_models.py             GARCH(1,1), GJR, EGARCH, CGARCH.
    markov_switching.py         Hamilton (1989) two-state regime model.
  utils/
    numba_kernels.py            JIT-compiled BS, IV bisection, RV.
    statistics.py               Jarque-Bera, KS, Ljung-Box, variance ratio.
    metrics.py                  Sharpe, Sortino, Calmar, VaR, CVaR.
    performance.py              CPU detection, timing decorator.
    plotting.py                 Chart helpers.
  output/
    charts/                     10 PNG charts (generated on run).
    data/                       CSV exports (signals, trades, hedge data).
    reports/                    LaTeX source and compiled PDF.
```

## Pipeline

```
Load data -> EDA (6 findings) -> Signal construction -> Walk-forward backtest
                                       |                        |
                                  GARCH filter            Hold-to-expiry
                                  Regime filter           Stop-loss (3x)
                                  VRP confirmation        Cut-loss (0.3)
                                  Jump avoidance
```

## Models

| Model | Role | Output |
|-------|------|--------|
| EGARCH(1,1) | Primary entry filter | Conditional vol |
| Hamilton MS | Safety filter | P(high-vol regime) |
| Black-Scholes | Pricing and Greeks | Premium, delta, gamma, theta, vega |
| Poisson jump count | Tail risk filter | 30-day jump count |

## Signal Logic

SHORT strangle when all four hold:
1. GARCH vol > expanding median (vol elevated, will revert)
2. Regime P(high-vol) < 0.5 (market is calm)
3. VRP z-score < 0 (IV and RV compressing together)
4. Trailing 30-day jump count < 2 (no jump clustering)

LONG strangle when all three hold:
1. Month in June, July, August (EIA injection season)
2. GARCH vol < expanding median (premium is cheap)
3. Trailing 30-day jump count < 2

Exit: hold to expiry (DTE <= 5). SHORT stop at 3x premium. LONG cut at 30%.

## Strategy 2: Vol-Targeted Defensive Allocation

Core mechanism: `delta = target_vol / rv20` (0.25 / realised vol).

| Market state | Delta | Rationale |
|---|---|---|
| Calm (no stress) | 1.0–1.5 | Leverage in low-vol trending markets |
| Stress + uptrend | ≤ 0.50 | Rapid exposure cut during vol spikes |
| Stress + downtrend | ≤ 0.25 | Near-flat, preserve capital |

Crisis-onset protective puts (15% OTM, 45-day) bought when stress flag rises.
Closed when stress ends. This limits theta bleed to crisis periods only.

The vol-targeting generates mild leverage in calm periods (~70% of days),
which more than compensates for reduced participation during stress (~30%).
Result: S2 cumulative P&L sits above B&H for ~90% of trading days.
