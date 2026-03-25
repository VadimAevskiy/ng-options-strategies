"""
Strategy Performance Metrics
=============================
Hedge-fund-standard performance reporting.
Sharpe and risk metrics computed on raw daily PnL, not portfolio returns.
This is correct for strategies where notional capital is not well-defined
(e.g. options strategies denominated in cents/MMBtu).
"""
import numpy as np
import pandas as pd


ANN_FACTOR = 252


def compute_metrics(pnl_series: "pd.Series", name: str = "Strategy") -> dict:
    """
    Compute performance metrics from a daily PnL series.

    Sharpe is computed directly on daily PnL:
        Sharpe = mean(pnl) * sqrt(252) / std(pnl)
    This avoids the distortion caused by dividing small PnL by large notional.

    Parameters
    ----------
    pnl_series : pd.Series
        Daily P&L (can be sparse with zeros on non-trade days).

    Returns
    -------
    dict : Metrics suitable for IC reporting.
    """
    pnl = pnl_series.dropna()
    if len(pnl) < 2:
        return _empty_metrics()

    cum_pnl = pnl.cumsum()
    n_days = len(pnl)
    total_pnl = cum_pnl.iloc[-1]

    # Sharpe on raw PnL (standard for absolute-return strategies)
    mean_pnl = pnl.mean()
    std_pnl = pnl.std()
    sharpe = mean_pnl / std_pnl * np.sqrt(ANN_FACTOR) if std_pnl > 0 else 0

    # Sortino: penalise only downside
    downside = pnl[pnl < 0]
    down_std = downside.std() if len(downside) > 1 else 1e-10
    sortino = mean_pnl / down_std * np.sqrt(ANN_FACTOR) if down_std > 0 else 0

    # Max drawdown (on cumulative PnL, not returns)
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_dd = drawdown.min()
    calmar = (mean_pnl * ANN_FACTOR) / abs(max_dd) if max_dd != 0 else 0

    # Win rate and profit factor (on non-zero days)
    nonzero = pnl[pnl != 0]
    winning = nonzero[nonzero > 0]
    losing = nonzero[nonzero < 0]
    win_rate = len(winning) / len(nonzero) if len(nonzero) > 0 else 0
    profit_factor = winning.sum() / abs(losing.sum()) if losing.sum() != 0 else np.inf

    # VaR / CVaR on non-zero PnL (trade-level risk)
    if len(nonzero) > 10:
        trade_var = np.percentile(nonzero, 5)
        trade_cvar = nonzero[nonzero <= trade_var].mean() if (nonzero <= trade_var).any() else trade_var
    else:
        trade_var = np.percentile(pnl, 5) if len(pnl) > 10 else 0
        trade_cvar = pnl[pnl <= trade_var].mean() if (pnl <= trade_var).any() else trade_var

    ann_pnl = mean_pnl * ANN_FACTOR

    return {
        "Total P&L": f"{total_pnl:+.1f}",
        "Ann. P&L": f"{ann_pnl:+.1f}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Sortino Ratio": f"{sortino:.2f}",
        "Max Drawdown": f"{max_dd:+.1f}",
        "Calmar Ratio": f"{calmar:.2f}",
        "Win Rate": f"{win_rate:.0%}",
        "Profit Factor": f"{profit_factor:.2f}",
        "Trade VaR (95%)": f"{trade_var:.2f}",
        "Trade CVaR (95%)": f"{trade_cvar:.2f}",
        "Trading Days": n_days,
        # Raw values for sorting and comparison
        "_sharpe": sharpe,
        "_ann_pnl": ann_pnl,
        "_max_dd": max_dd,
        "_total_pnl": total_pnl,
    }


def _empty_metrics():
    return {k: "N/A" for k in [
        "Total P&L", "Ann. P&L", "Sharpe Ratio",
        "Sortino Ratio", "Max Drawdown", "Calmar Ratio", "Win Rate",
        "Profit Factor", "Trade VaR (95%)",
        "Trade CVaR (95%)", "Trading Days",
    ]}


def compute_benchmark(futures_returns, label="Buy & Hold Futures"):
    """Compute buy-and-hold futures benchmark metrics."""
    cum = (1 + futures_returns).cumprod()
    equity = cum * 1_000_000
    daily_pnl = equity.diff().fillna(0)
    m = compute_metrics(daily_pnl)
    m["Strategy"] = label
    return m


def comparison_table(results_dict, benchmark=None):
    """
    Build side-by-side comparison table.
    results_dict = {"Strategy Name": metrics_dict, ...}
    """
    rows = []
    if benchmark:
        rows.append({"Strategy": "Benchmark", **benchmark})
    for name, m in results_dict.items():
        rows.append({"Strategy": name, **m})

    df = pd.DataFrame(rows).set_index("Strategy")
    # Drop internal raw columns
    display_cols = [c for c in df.columns if not c.startswith("_")]
    return df[display_cols]


def format_trade_summary(trades_df):
    """Format trade log into summary statistics."""
    if trades_df is None or len(trades_df) == 0:
        return "  No trades executed."

    lines = []
    n = len(trades_df)
    wins = (trades_df["pnl"] > 0).sum()
    losses = (trades_df["pnl"] < 0).sum()
    flat = n - wins - losses

    lines.append(f"  Total trades:    {n}")
    lines.append(f"  Winners/Losers:  {wins}/{losses} (flat: {flat})")
    lines.append(f"  Win rate:        {wins/max(n,1):.0%}")
    lines.append(f"  Avg P&L/trade:   {trades_df['pnl'].mean():.2f}")
    lines.append(f"  Median P&L:      {trades_df['pnl'].median():.2f}")
    lines.append(f"  Best trade:      {trades_df['pnl'].max():.2f}")
    lines.append(f"  Worst trade:     {trades_df['pnl'].min():.2f}")
    lines.append(f"  Total P&L:       {trades_df['pnl'].sum():.2f}")

    if "days" in trades_df.columns:
        lines.append(f"  Avg hold period: {trades_df['days'].mean():.0f} days")

    if "dir" in trades_df.columns:
        for d in trades_df["dir"].unique():
            sub = trades_df[trades_df["dir"] == d]
            wr = (sub["pnl"] > 0).mean()
            lines.append(f"    {d.upper():6s}: {len(sub)} trades, WR={wr:.0%}, "
                        f"avg={sub['pnl'].mean():.2f}, total={sub['pnl'].sum():.2f}")

    return "\n".join(lines)


def print_comparison(table_df, title="PERFORMANCE COMPARISON"):
    """Pretty-print comparison table."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    # Transpose for readability
    print(table_df.T.to_string())
    print(f"{'='*80}\n")
