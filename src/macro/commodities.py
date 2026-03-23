"""Fetch commodity, currency, and sentiment data via yfinance."""

import yfinance as yf
import pandas as pd
from datetime import datetime
from src.common.config import load_yaml
from src.common.logger import get_logger

log = get_logger("macro.commodities")


def fetch_yf_macro_series(symbol, name, days=60):
    """Fetch a single yfinance time series for a macro indicator.

    Returns DataFrame with: date, indicator, value, symbol
    """
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=f"{days}d")
        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        df["date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df["indicator"] = name
        df["symbol"] = symbol
        df["value"] = df["Close"]
        df["change_pct"] = df["Close"].pct_change(1)

        return df[["date", "indicator", "symbol", "value", "change_pct"]]
    except Exception as e:
        log.warning(f"Failed to fetch {name} ({symbol}): {e}")
        return pd.DataFrame()


def fetch_all_yf_macro(days=60):
    """Fetch all yfinance-sourced macro indicators (commodities, currencies, sentiment).

    Reads indicator list from config/macro_indicators.yaml.
    Returns combined DataFrame.
    """
    cfg = load_yaml("macro_indicators.yaml")
    all_data = []

    for category in ["sentiment", "currencies", "commodities", "crypto"]:
        items = cfg.get(category, [])
        for item in items:
            df = fetch_yf_macro_series(item["symbol"], item["name"], days)
            if not df.empty:
                df["category"] = category
                all_data.append(df)
                log.info(f"  {item['name']}: {len(df)} days")

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    log.info(f"Fetched {len(result)} macro data points ({result['indicator'].nunique()} indicators)")
    return result


def get_macro_latest(macro_df):
    """Get the latest value for each macro indicator.

    Returns dict of indicator -> {value, change_pct, date}
    """
    if macro_df.empty:
        return {}

    latest = (
        macro_df.sort_values("date")
        .groupby("indicator")
        .tail(1)
        .set_index("indicator")
    )

    result = {}
    for name, row in latest.iterrows():
        result[name] = {
            "value": round(row["value"], 2) if pd.notna(row["value"]) else None,
            "change_pct": round(row["change_pct"], 4) if pd.notna(row["change_pct"]) else None,
            "date": row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else None,
            "symbol": row.get("symbol", ""),
        }

    return result
