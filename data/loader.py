"""
Data loader
============
Load and validate options and futures data from CME files.
"""
import pandas as pd
import numpy as np
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR


def load_futures(path=None):
    """Load OI-weighted futures data."""
    path = Path(path) if path else DATA_DIR / "oi_weighted_futures.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    df.columns = [c.strip() for c in df.columns]

    # Validate
    assert "Close" in df.columns, f"Expected 'Close' column, found {df.columns.tolist()}"
    assert not df.index.duplicated().any(), "Duplicate dates in futures data"

    # Derived fields
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["simple_return"] = df["Close"].pct_change()
    return df


def load_options(path=None):
    """
    Load options daily data from parquet.
    Expected columns (from CME data):
      Utcdate, Ticker, CallputLast, StrikeLast, MonthLast, ExpirationyearLast,
      LocaldateLast, SettlementPrice, Volume, OpenInterest, ...
    """
    path = Path(path) if path else DATA_DIR / "options_daily.parquet"
    try:
        if str(path).endswith(".csv"):
            df = pd.read_csv(path)
        else:
            try:
                df = pd.read_parquet(path)
            except Exception:
                # Try CSV with same stem
                csv_path = path.with_suffix(".csv")
                if csv_path.exists():
                    print(f"Parquet unavailable, loading CSV from {csv_path}")
                    df = pd.read_csv(csv_path)
                else:
                    raise
    except Exception as e:
        print(f"Error reading options data: {e}")
        raise

    # Standardise column names
    # Priority: use CLOSE bid/ask (end-of-day) over OPEN bid/ask
    col_map = {}

    # Explicit exact mappings (highest priority)
    exact_map = {
        "Utcdate": "date",
        "CallputLast": "option_type",
        "StrikeLast": "strike",
        "MonthLast": "month_code",
        "ExpirationyearLast": "expiry_year",
        "Ticker": "ticker",
        # Use CLOSE bid/ask for end-of-day pricing
        "ClosebidpriceLast": "bid",
        "CloseaskpriceLast": "ask",
        # Settlement price (if present)
        "SettlementpriceLast": "settlement",
        # Last trade price (close)
        "ClosetradepriceLast": "last_price",
        # Volume: use Sum (total for the day) or Last
        "VolumeSum": "volume",
    }
    _used_targets = set()
    for col_name, target in exact_map.items():
        if col_name in df.columns and target not in _used_targets:
            col_map[col_name] = target
            _used_targets.add(target)

    # Fallback pattern matching for columns not found above
    for col in df.columns:
        if col in col_map:
            continue
        col_lower = col.lower().strip()
        target = None

        if "utcdate" in col_lower and "date" not in _used_targets:
            target = "date"
        elif "callput" in col_lower and "option_type" not in _used_targets:
            target = "option_type"
        elif "strike" in col_lower and "last" in col_lower and "strike" not in _used_targets:
            target = "strike"
        elif "month" in col_lower and "last" in col_lower and "month_code" not in _used_targets:
            target = "month_code"
        elif "expirationyear" in col_lower and "expiry_year" not in _used_targets:
            target = "expiry_year"
        elif ("settlement" in col_lower or "settle" in col_lower) and "settlement" not in _used_targets:
            target = "settlement"
        elif col_lower == "volumelast" and "volume" not in _used_targets:
            target = "volume"
        elif col_lower == "volumesum" and "volume" not in _used_targets:
            target = "volume"
        elif "openinterest" in col_lower and "open_interest" not in _used_targets:
            target = "open_interest"
        elif "closebidpric" in col_lower and "bid" not in _used_targets:
            target = "bid"
        elif "closeaskpric" in col_lower and "ask" not in _used_targets:
            target = "ask"
        elif "openbidpric" in col_lower and "last" in col_lower and "bid" not in _used_targets:
            target = "bid"
        elif "openaskpric" in col_lower and "last" in col_lower and "ask" not in _used_targets:
            target = "ask"
        elif col_lower == "ticker" and "ticker" not in _used_targets:
            target = "ticker"

        if target is not None and target not in _used_targets:
            col_map[col] = target
            _used_targets.add(target)

    df = df.rename(columns=col_map)
    print(f"  Column mapping: {', '.join(f'{k} -> {v}' for k, v in sorted(col_map.items(), key=lambda x: x[1]))}")

    # Drop any duplicate column names (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Ensure numeric strikes
    if "strike" in df.columns:
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")

    # Standardise option_type to 'C' / 'P'
    if "option_type" in df.columns:
        df["option_type"] = df["option_type"].astype(str).str.upper().str[0]

    return df


# Month code mapping (CME futures convention)
MONTH_CODES = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}


def compute_expiry_date(month_code, year):
    """Approximate expiry date from month code and year (3rd Friday of expiry month)."""
    month = MONTH_CODES.get(month_code, None)
    if month is None:
        return pd.NaT
    # Third Friday
    import calendar
    cal = calendar.Calendar()
    fridays = [d for d in cal.itermonthdays2(year, month)
               if d[0] > 0 and d[1] == 4]  # Friday = 4
    if len(fridays) >= 3:
        day = fridays[2][0]
    else:
        day = fridays[-1][0]
    return pd.Timestamp(year, month, day)


def enrich_options(options_df, futures_df):
    """
    Add derived columns to options data:
      - expiry_date, days_to_expiry, time_to_expiry (in years)
      - underlying_price (from futures on same date)
      - moneyness (strike / underlying)
      - mid_price, spread
    """
    df = options_df.copy()

    # Expiry date
    if "month_code" in df.columns and "expiry_year" in df.columns:
        df["expiry_date"] = df.apply(
            lambda row: compute_expiry_date(row.get("month_code", ""),
                                            int(row.get("expiry_year", 2019))),
            axis=1
        )
    elif "expiry_date" not in df.columns:
        # Fallback: assign a dummy
        df["expiry_date"] = df["date"] + pd.Timedelta(days=30)

    # Days to expiry
    df["dte"] = (df["expiry_date"] - df["date"]).dt.days
    df["time_to_expiry"] = df["dte"] / 365.0

    # Merge underlying price
    if "Close" in futures_df.columns:
        fut_price = futures_df[["Close"]].rename(columns={"Close": "underlying"})
        df = df.merge(fut_price, left_on="date", right_index=True, how="left")
    else:
        df["underlying"] = np.nan

    # Moneyness
    if "underlying" in df.columns and "strike" in df.columns:
        df["moneyness"] = df["strike"] / df["underlying"]
        df["log_moneyness"] = np.log(df["moneyness"])

    # Mid price — ensure bid/ask are single columns (not DataFrames from duplicate mapping)
    if "bid" in df.columns and "ask" in df.columns:
        bid_col = df["bid"]
        ask_col = df["ask"]
        # If duplicate columns exist, take the first one
        if isinstance(bid_col, pd.DataFrame):
            bid_col = bid_col.iloc[:, 0]
        if isinstance(ask_col, pd.DataFrame):
            ask_col = ask_col.iloc[:, 0]
        bid_col = pd.to_numeric(bid_col, errors="coerce")
        ask_col = pd.to_numeric(ask_col, errors="coerce")
        df["mid_price"] = 0.5 * (bid_col + ask_col)
        df["spread"] = ask_col - bid_col
    elif "settlement" in df.columns:
        df["mid_price"] = df["settlement"]
    elif "last_price" in df.columns:
        df["mid_price"] = df["last_price"]

    # Filter out clearly bad data
    df = df[df["dte"] > 0]
    if "mid_price" in df.columns:
        df = df[df["mid_price"] > 0]

    return df
