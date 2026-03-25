"""
Data generator
=========================
Creates a realistic options_daily.parquet matching the CME data schema
observed in the uploaded screenshot. Replace with real data for production.

Column schema (from CME):
  Utcdate, Ticker, CallputLast, StrikeLast, MonthLast, ExpirationyearLast,
  UtctimebarstartLast, LocaldateLast, LocaltimebarstartLast,
  OpenbidtimeLast, OpenbidpriceLast, OpenaskpriceLast, OpenaskTimeLast,
  HighbidpriceLast, HighaskpriceLast, LowbidpriceLast, LowaskpriceLast,
  LastpriceLast, SettlementpriceLast, VolumeLast, OpeninterestLast
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from pathlib import Path
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, RISK_FREE_RATE

# CME Month codes
MONTH_CODES = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
               7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}


def _bs_price(S, K, T, r, sigma, is_call):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if is_call else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def generate_synthetic_parquet(futures_csv_path, output_path=None, seed=42):
    """
    Generate options data consistent with provided futures data.

    Parameters
    ----------
    futures_csv_path : str or Path
        Path to oi_weighted_futures.csv
    output_path : str or Path, optional
        Where to save the parquet. Defaults to DATA_DIR/options_daily.parquet

    Returns
    -------
    pd.DataFrame
        The generated options data
    """
    np.random.seed(seed)
    output_path = Path(output_path) if output_path else DATA_DIR / "options_daily.parquet"

    # Load futures to get price levels and dates
    fut = pd.read_csv(futures_csv_path, parse_dates=["Date"])
    fut = fut.set_index("Date").sort_index()
    fut["log_ret"] = np.log(fut["Close"] / fut["Close"].shift(1))

    # Realised vol (rolling 20d)
    fut["rv20"] = fut["log_ret"].rolling(20).std() * np.sqrt(252)
    fut["rv20"] = fut["rv20"].fillna(fut["rv20"].median())

    # Define expiry cycles (quarterly: H, M, U, Z + serial months)
    all_dates = fut.index
    records = []

    # Generate options for a sample of dates (every trading day)
    # Use quarterly expiries + 2 nearest monthly
    ticker_prefix = "HXE"  # Matching observed ticker pattern

    for trade_date in all_dates:
        S = fut.loc[trade_date, "Close"]
        rv = fut.loc[trade_date, "rv20"]
        if np.isnan(S) or np.isnan(rv):
            continue

        base_vol = max(rv, 0.10)  # Floor at 10%

        # Generate 2-3 expiries per trade date
        year = trade_date.year
        month = trade_date.month

        expiry_months = []
        for offset in [1, 2, 3, 6]:
            exp_m = month + offset
            exp_y = year
            while exp_m > 12:
                exp_m -= 12
                exp_y += 1
            expiry_months.append((exp_y, exp_m))

        for exp_year, exp_month in expiry_months:
            exp_date = pd.Timestamp(exp_year, exp_month, 15)  # Approx 3rd Friday
            dte = (exp_date - trade_date).days
            if dte <= 0 or dte > 365:
                continue
            T = dte / 365.0

            month_code = MONTH_CODES[exp_month]
            year_suffix = str(exp_year)[-1]

            # Strike range: roughly ±30% around spot, in round increments
            step = max(5, int(S * 0.02))  # ~2% increments
            strike_min = int(S * 0.70 / step) * step
            strike_max = int(S * 1.30 / step) * step
            strikes = np.arange(strike_min, strike_max + step, step)

            for K in strikes:
                moneyness = K / S
                # Volatility smile: skew + curvature
                skew = -0.15 * (moneyness - 1.0)
                smile = 0.3 * (moneyness - 1.0) ** 2
                term_adj = 0.02 * np.sqrt(T)
                iv = base_vol + skew + smile + term_adj
                iv = max(iv, 0.03)
                iv += np.random.normal(0, 0.005)  # Small noise
                iv = max(iv, 0.02)

                for cp in ["C", "P"]:
                    is_call = (cp == "C")
                    price = _bs_price(S, K, T, RISK_FREE_RATE, iv, is_call)
                    if price < 0.01:
                        continue

                    # Bid-ask spread: wider for OTM, narrower for ATM
                    spread_pct = 0.02 + 0.05 * abs(moneyness - 1.0) + 0.01 / max(T, 0.01)
                    half_spread = price * spread_pct * 0.5
                    bid = max(price - half_spread, 0.01)
                    ask = price + half_spread

                    # Volume and OI: higher for ATM
                    atm_factor = np.exp(-10 * (moneyness - 1.0) ** 2)
                    volume = max(int(np.random.exponential(100 * atm_factor)), 0)
                    oi = max(int(np.random.exponential(500 * atm_factor)), 0)

                    ticker = f"{ticker_prefix}{month_code}{year_suffix} {cp}{int(K)}"
                    local_date = int(trade_date.strftime("%Y%m%d"))

                    records.append({
                        "Utcdate": trade_date,
                        "Ticker": ticker,
                        "CallputLast": cp,
                        "StrikeLast": int(K),
                        "MonthLast": month_code,
                        "ExpirationyearLast": exp_year,
                        "UtctimebarstartLast": "13:08",
                        "LocaldateLast": local_date,
                        "LocaltimebarstartLast": "07:08",
                        "OpenbidtimeLast": "13:08:57.908",
                        "OpenbidpriceLast": round(bid, 2),
                        "OpenaskpriceLast": round(ask, 2),
                        "OpenasktimeLast": "13:08:57.908",
                        "HighbidpriceLast": round(bid * (1 + np.random.uniform(0, 0.02)), 2),
                        "HighaskpriceLast": round(ask * (1 + np.random.uniform(0, 0.02)), 2),
                        "LowbidpriceLast": round(bid * (1 - np.random.uniform(0, 0.02)), 2),
                        "LowaskpriceLast": round(ask * (1 - np.random.uniform(0, 0.02)), 2),
                        "LastpriceLast": round(price + np.random.normal(0, half_spread * 0.3), 2),
                        "SettlementpriceLast": round(price, 2),
                        "VolumeLast": volume,
                        "OpeninterestLast": oi,
                    })

    df = pd.DataFrame(records)

    # Save as parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(output_path, index=False)
        print(f"Saved options data to {output_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df['Utcdate'].min()} to {df['Utcdate'].max()}")
        print(f"  Unique strikes: {df['StrikeLast'].nunique()}")
        print(f"  Unique expiries: {df[['MonthLast', 'ExpirationyearLast']].drop_duplicates().shape[0]}")
    except ImportError:
        csv_path = output_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"pyarrow not available, saved as CSV: {csv_path}")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--futures", default=str(DATA_DIR / "oi_weighted_futures.csv"))
    parser.add_argument("--output", default=str(DATA_DIR / "options_daily.parquet"))
    args = parser.parse_args()
    generate_synthetic_parquet(args.futures, args.output)
