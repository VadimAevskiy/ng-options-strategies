"""
Plotting utilities
==================
Consistent, publication-quality charts.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Plot defaults
PLOT_STYLE = "ggplot"
FIG_SIZE = (14, 7)
DPI = 150

try:
    plt.style.use(PLOT_STYLE)
except Exception:
    plt.style.use("ggplot")

COLORS = sns.color_palette("muted", 10)


def save_fig(fig, name, dpi=DPI):
    """Save figure to output directory and display in Jupyter."""
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            plt.show()
            return path
    except ImportError:
        pass
    plt.close(fig)
    return path


# ── Time-series plot ───────────────────────────────────────────────────────
def plot_timeseries(df, columns, title="", ylabel="", figsize=FIG_SIZE, save_name=None):
    fig, ax = plt.subplots(figsize=figsize)
    for i, col in enumerate(columns):
        ax.plot(df.index, df[col], label=col, color=COLORS[i % len(COLORS)], linewidth=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    if save_name:
        save_fig(fig, save_name)
    return fig, ax


# ── Return distribution ───────────────────────────────────────────────────
def plot_return_distribution(returns, title="Return Distribution", save_name=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Histogram + KDE
    axes[0].hist(returns.dropna(), bins=80, density=True, alpha=0.6, color=COLORS[0], edgecolor="white")
    x = np.linspace(returns.min(), returns.max(), 300)
    from scipy.stats import norm
    mu, sigma = returns.mean(), returns.std()
    axes[0].plot(x, norm.pdf(x, mu, sigma), "r-", lw=2, label=f"N({mu:.4f}, {sigma:.4f})")
    axes[0].set_title("Histogram + Normal fit")
    axes[0].legend()

    # QQ-plot
    from scipy.stats import probplot
    probplot(returns.dropna(), dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot (Normal)")
    axes[1].get_lines()[0].set_markerfacecolor(COLORS[0])
    axes[1].get_lines()[0].set_markersize(3)

    # Box plot
    axes[2].boxplot(returns.dropna(), vert=True, widths=0.5,
                    patch_artist=True, boxprops=dict(facecolor=COLORS[0], alpha=0.5))
    axes[2].set_title("Box Plot")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    if save_name:
        save_fig(fig, save_name)
    return fig


# ── Volatility surface / smile ────────────────────────────────────────────
def plot_vol_surface(strikes, expiries, iv_matrix, title="Implied Volatility Surface",
                     save_name=None):
    """3D implied volatility surface."""
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection="3d")
    K, T = np.meshgrid(strikes, expiries)
    ax.plot_surface(K, T, iv_matrix, cmap="viridis", alpha=0.8, edgecolor="k", linewidth=0.2)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Days to Expiry")
    ax.set_zlabel("Implied Vol")
    ax.set_title(title)
    if save_name:
        save_fig(fig, save_name)
    return fig, ax


def plot_vol_smile(strikes, ivs, spot=None, title="Volatility Smile", save_name=None):
    """2D vol smile."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(strikes, ivs, "o-", color=COLORS[0], markersize=4)
    if spot is not None:
        ax.axvline(spot, color="red", linestyle="--", alpha=0.7, label=f"Spot={spot:.1f}")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Volatility")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    if save_name:
        save_fig(fig, save_name)
    return fig, ax


# ── Strategy PnL ──────────────────────────────────────────────────────────
def plot_strategy_pnl(pnl_series, title="Strategy Cumulative P&L", save_name=None):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    cum_pnl = pnl_series.cumsum()
    axes[0].plot(cum_pnl.index, cum_pnl.values, color=COLORS[0], linewidth=1.2)
    axes[0].fill_between(cum_pnl.index, 0, cum_pnl.values,
                         where=cum_pnl.values >= 0, color=COLORS[2], alpha=0.3)
    axes[0].fill_between(cum_pnl.index, 0, cum_pnl.values,
                         where=cum_pnl.values < 0, color=COLORS[3], alpha=0.3)
    axes[0].set_title(title, fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Cumulative P&L")
    axes[0].axhline(0, color="grey", linewidth=0.5)

    # Drawdown
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    axes[1].fill_between(drawdown.index, 0, drawdown.values, color=COLORS[3], alpha=0.5)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")

    fig.tight_layout()
    if save_name:
        save_fig(fig, save_name)
    return fig


# ── Greeks over time ──────────────────────────────────────────────────────
def plot_greeks_timeseries(dates, greeks_dict, title="Portfolio Greeks", save_name=None):
    """Plot portfolio-level Greeks over time. greeks_dict = {name: array}."""
    n = len(greeks_dict)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for i, (name, values) in enumerate(greeks_dict.items()):
        axes[i].plot(dates, values, color=COLORS[i], linewidth=0.8)
        axes[i].axhline(0, color="grey", linewidth=0.5)
        axes[i].set_ylabel(name, fontsize=11, fontweight="bold")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.autofmt_xdate()
    if save_name:
        save_fig(fig, save_name)
    return fig


# ── Correlation heatmap ───────────────────────────────────────────────────
def plot_correlation_matrix(df, title="Correlation Matrix", save_name=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    if save_name:
        save_fig(fig, save_name)
    return fig


# ── Regime plot ───────────────────────────────────────────────────────────
def plot_regimes(dates, series, regime_probs, title="Regime Detection", save_name=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(dates, series, color=COLORS[0], linewidth=0.7)
    ax1.set_ylabel("Value")
    ax1.set_title(title, fontsize=14, fontweight="bold")

    ax2.fill_between(dates, 0, regime_probs, color=COLORS[3], alpha=0.6, label="P(High-vol regime)")
    ax2.set_ylabel("Regime probability")
    ax2.legend()
    ax2.set_ylim(0, 1)
    fig.tight_layout()
    if save_name:
        save_fig(fig, save_name)
    return fig
