"""FRED API client for US economic indicators."""

import pandas as pd
from datetime import datetime, timedelta
from src.common.config import load_yaml
from src.common.logger import get_logger

log = get_logger("macro.fred")

# Try to import fredapi, fall back to requests-based approach
try:
    from fredapi import Fred
    HAS_FREDAPI = True
except ImportError:
    HAS_FREDAPI = False

import requests


def _fetch_fred_via_api(series_id, api_key, start_date=None):
    """Fetch a FRED series using the fredapi package."""
    fred = Fred(api_key=api_key)
    data = fred.get_series(series_id, observation_start=start_date)
    if data is None or data.empty:
        return pd.DataFrame()
    df = data.reset_index()
    df.columns = ["date", "value"]
    df["series_id"] = series_id
    return df


def _fetch_fred_via_requests(series_id, api_key, start_date=None):
    """Fetch a FRED series using raw HTTP requests."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    if start_date:
        params["observation_start"] = start_date

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        observations = data.get("observations", [])
        if not observations:
            return pd.DataFrame()

        rows = []
        for obs in observations:
            val = obs.get("value", ".")
            if val == ".":
                continue
            rows.append({
                "date": pd.to_datetime(obs["date"]),
                "value": float(val),
                "series_id": series_id,
            })

        return pd.DataFrame(rows)
    except Exception as e:
        log.warning(f"Failed to fetch FRED {series_id}: {e}")
        return pd.DataFrame()


def _fetch_fred_public_csv(series_id, start_date=None):
    """Fetch a FRED series via public CSV endpoint (no API key required)."""
    from io import StringIO
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    try:
        resp = requests.get(url, params={"id": series_id}, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["value"] != "."].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        df["series_id"] = series_id
        return df
    except Exception as e:
        log.warning(f"Failed to fetch FRED public CSV {series_id}: {e}")
        return pd.DataFrame()


def fetch_fred_series(series_id, api_key=None, days_back=365):
    """Fetch a single FRED series.

    Args:
        series_id: FRED series ID (e.g., "DGS10").
        api_key: FRED API key. If None, tries env var FRED_API_KEY, then public CSV.
        days_back: Number of days of history to fetch.
    """
    import os
    if api_key is None:
        api_key = os.environ.get("FRED_API_KEY")

    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    if not api_key:
        log.info(f"No FRED API key — using public CSV for {series_id}")
        return _fetch_fred_public_csv(series_id, start_date)

    if HAS_FREDAPI:
        return _fetch_fred_via_api(series_id, api_key, start_date)
    else:
        return _fetch_fred_via_requests(series_id, api_key, start_date)


def fetch_all_fred_series(api_key=None):
    """Fetch all FRED series defined in macro_indicators.yaml.

    Returns combined DataFrame with: date, series_id, name, value, frequency
    """
    cfg = load_yaml("macro_indicators.yaml")
    fred_cfg = cfg.get("fred_series", {})
    all_data = []

    for frequency, series_list in fred_cfg.items():
        days_back = {"daily": 365, "monthly": 365 * 5, "quarterly": 365 * 10}.get(
            frequency, 365
        )
        for item in series_list:
            df = fetch_fred_series(item["series_id"], api_key, days_back)
            if not df.empty:
                df["name"] = item["name"]
                df["frequency"] = frequency
                all_data.append(df)
                log.info(f"  FRED {item['name']} ({item['series_id']}): {len(df)} obs")

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    log.info(f"Fetched {len(result)} FRED data points")
    return result


def compute_yield_curve(fred_df):
    """Compute yield curve spread (10Y - 2Y) from FRED data.

    Returns DataFrame with: date, spread
    """
    if fred_df.empty:
        return pd.DataFrame()

    ten_y = fred_df[fred_df["series_id"] == "DGS10"][["date", "value"]].copy()
    ten_y.columns = ["date", "yield_10y"]
    two_y = fred_df[fred_df["series_id"] == "DGS2"][["date", "value"]].copy()
    two_y.columns = ["date", "yield_2y"]

    if ten_y.empty or two_y.empty:
        return pd.DataFrame()

    merged = ten_y.merge(two_y, on="date", how="inner")
    merged["spread"] = merged["yield_10y"] - merged["yield_2y"]
    merged["inverted"] = merged["spread"] < 0

    return merged
