"""
Data preprocessor
==================
Feature engineering, IV computation, and data cleaning for the options pipeline.
"""
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RISK_FREE_RATE
from utils.numba_kernels import implied_vol_vectorised, bs_greeks_vectorised


def compute_implied_vols(options_df, r=RISK_FREE_RATE):
    """
    Compute implied volatility for each option row using mid_price and underlying.
    Adds 'iv' column in-place (annualised).
    """
    df = options_df.copy()
    mask = (df["mid_price"] > 0) & (df["time_to_expiry"] > 0) & (df["underlying"] > 0)
    subset = df.loc[mask]

    if len(subset) == 0:
        df["iv"] = np.nan
        return df

    prices = subset["mid_price"].values.astype(np.float64)
    S = subset["underlying"].values.astype(np.float64)
    K = subset["strike"].values.astype(np.float64)
    T = subset["time_to_expiry"].values.astype(np.float64)
    is_call = (subset["option_type"] == "C").values

    ivs = implied_vol_vectorised(prices, S, K, T, r, is_call)
    df.loc[mask, "iv"] = ivs

    # Filter unreasonable IVs
    df.loc[(df["iv"] < 0.01) | (df["iv"] > 5.0), "iv"] = np.nan

    return df


def compute_greeks(options_df, r=RISK_FREE_RATE, vol_col="iv"):
    """
    Compute BS Greeks for all options.
    Adds columns: bs_price, delta, gamma, vega, theta, rho.
    """
    df = options_df.copy()
    mask = (df[vol_col] > 0) & (df["time_to_expiry"] > 0) & (df["underlying"] > 0)
    subset = df.loc[mask]

    if len(subset) == 0:
        for c in ["bs_price", "delta", "gamma", "vega", "theta", "rho"]:
            df[c] = np.nan
        return df

    S = subset["underlying"].values.astype(np.float64)
    K = subset["strike"].values.astype(np.float64)
    T = subset["time_to_expiry"].values.astype(np.float64)
    sigma = subset[vol_col].values.astype(np.float64)
    is_call = (subset["option_type"] == "C").values

    greeks = bs_greeks_vectorised(S, K, T, r, sigma, is_call)

    for i, col in enumerate(["bs_price", "delta", "gamma", "vega", "theta", "rho"]):
        df.loc[mask, col] = greeks[:, i]

    return df


def build_vol_surface(options_df, date, min_oi=10, min_volume=1):
    """
    Extract volatility surface for a given date.
    Returns DataFrame with columns: strike, dte, moneyness, iv, option_type.
    """
    df = options_df[options_df["date"] == pd.Timestamp(date)].copy()

    # Filter for liquid options
    if "open_interest" in df.columns:
        df = df[df["open_interest"] >= min_oi]
    if "volume" in df.columns:
        df = df[df["volume"] >= min_volume]

    df = df[df["iv"].notna() & (df["iv"] > 0)]

    return df[["strike", "dte", "time_to_expiry", "moneyness", "iv",
               "option_type", "underlying"]].sort_values(["dte", "strike"])


def compute_term_structure(options_df, date, moneyness_range=(0.95, 1.05)):
    """
    ATM implied vol term structure for a given date.
    Returns DataFrame indexed by dte with ATM IV.
    """
    surf = build_vol_surface(options_df, date)
    atm = surf[(surf["moneyness"] >= moneyness_range[0]) &
               (surf["moneyness"] <= moneyness_range[1])]

    if len(atm) == 0:
        return pd.DataFrame()

    term = atm.groupby("dte")["iv"].mean().reset_index()
    term.columns = ["dte", "atm_iv"]
    return term.sort_values("dte")


def compute_skew_metrics(options_df, date):
    """
    Compute vol skew metrics for a given date:
      - 25-delta skew (put IV - call IV at 25-delta)
      - Risk reversal
      - Butterfly spread
    """
    df = options_df[(options_df["date"] == pd.Timestamp(date)) & (options_df["iv"].notna())].copy()

    if len(df) == 0 or "delta" not in df.columns:
        return {}

    results = {}
    for dte_group, grp in df.groupby("dte"):
        calls = grp[grp["option_type"] == "C"]
        puts = grp[grp["option_type"] == "P"]

        # Find 25-delta options
        c25 = calls.iloc[(calls["delta"] - 0.25).abs().argsort()[:1]]
        p25 = puts.iloc[(puts["delta"] + 0.25).abs().argsort()[:1]]
        atm_c = calls.iloc[(calls["delta"] - 0.50).abs().argsort()[:1]]

        if len(c25) > 0 and len(p25) > 0 and len(atm_c) > 0:
            rr = p25["iv"].values[0] - c25["iv"].values[0]  # Risk reversal
            bf = 0.5 * (p25["iv"].values[0] + c25["iv"].values[0]) - atm_c["iv"].values[0]
            results[dte_group] = dict(
                risk_reversal=rr,
                butterfly=bf,
                put_25d_iv=p25["iv"].values[0],
                call_25d_iv=c25["iv"].values[0],
                atm_iv=atm_c["iv"].values[0],
            )

    return results


def prepare_backtest_data(options_df, futures_df):
    """
    Prepare a unified dataset for backtesting.
    Returns daily DataFrame with:
      - Futures columns (Close, returns, realised vol)
      - Aggregated options metrics (ATM IV, skew, term structure slope)
    """
    # Futures features
    fut = futures_df.copy()
    fut["rv_5d"] = fut["log_return"].rolling(5).std() * np.sqrt(252)
    fut["rv_10d"] = fut["log_return"].rolling(10).std() * np.sqrt(252)
    fut["rv_20d"] = fut["log_return"].rolling(20).std() * np.sqrt(252)
    fut["rv_60d"] = fut["log_return"].rolling(60).std() * np.sqrt(252)
    fut["rv_ratio"] = fut["rv_5d"] / fut["rv_60d"]  # Vol-of-vol proxy

    # Daily aggregated options features
    opt = options_df.copy()
    daily_opt = []

    for date, grp in opt.groupby("date"):
        row = {"date": date}

        # ATM IV (moneyness 0.95-1.05)
        atm = grp[(grp["moneyness"] >= 0.95) & (grp["moneyness"] <= 1.05)]
        if len(atm) > 0:
            row["atm_iv"] = atm["iv"].mean()
            calls_atm = atm[atm["option_type"] == "C"]
            puts_atm = atm[atm["option_type"] == "P"]
            if len(calls_atm) > 0 and len(puts_atm) > 0:
                row["put_call_iv_spread"] = puts_atm["iv"].mean() - calls_atm["iv"].mean()

        # OTM put vol (skew indicator)
        otm_puts = grp[(grp["option_type"] == "P") & (grp["moneyness"] < 0.95)]
        if len(otm_puts) > 0:
            row["otm_put_iv"] = otm_puts["iv"].mean()

        # Volume-weighted IV
        if "volume" in grp.columns and grp["volume"].sum() > 0:
            valid = grp[(grp["iv"] > 0) & (grp["volume"] > 0)]
            if len(valid) > 0:
                row["vwap_iv"] = np.average(valid["iv"], weights=valid["volume"])

        # Put/Call volume ratio
        if "volume" in grp.columns:
            c_vol = grp[grp["option_type"] == "C"]["volume"].sum()
            p_vol = grp[grp["option_type"] == "P"]["volume"].sum()
            if c_vol > 0:
                row["put_call_vol_ratio"] = p_vol / c_vol

        daily_opt.append(row)

    daily_opt = pd.DataFrame(daily_opt).set_index("date")

    # Merge
    combined = fut.join(daily_opt, how="left")

    # Volatility risk premium = IV - RV
    if "atm_iv" in combined.columns and "rv_20d" in combined.columns:
        combined["vrp"] = combined["atm_iv"] - combined["rv_20d"]

    return combined
