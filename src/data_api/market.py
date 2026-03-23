"""
Per-market data access functions for the Global Tech Market pipeline.

Each function reads one logical dataset for one market and returns a typed,
ready-to-use object. Functions always return empty containers (empty DataFrame,
empty list, empty dict) when the underlying file is missing — they never raise.

Intended for use by DataAgent, NewsAgent, SignalAgent, and the GlobalAgent's
market-scanning loops.
"""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime

import pandas as pd

from src.common.config import get_data_path


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

def get_universe(market: str) -> pd.DataFrame:
    """Return the stock universe for a given market.

    Args:
        market: Two-letter market code (e.g. "US", "CN", "JP").

    Returns:
        DataFrame with columns: ticker, name, sector, industry,
        currency, yf_symbol. Empty DataFrame if file not yet written.
    """
    try:
        return pd.read_parquet(get_data_path("markets", market, "universe.parquet"))
    except Exception:
        return pd.DataFrame()


def get_universe_tickers(market: str) -> list[str]:
    """Return the list of ticker symbols for a given market.

    Args:
        market: Two-letter market code (e.g. "US", "CN", "JP").

    Returns:
        List of ticker strings. Empty list if universe file not found.
    """
    df = get_universe(market)
    if df.empty or "ticker" not in df.columns:
        return []
    return df["ticker"].tolist()


# ---------------------------------------------------------------------------
# Market daily snapshots
# ---------------------------------------------------------------------------

def get_market_daily(market: str, date: str | None = None) -> pd.DataFrame:
    """Return the OHLCV and return snapshot for a single trading day.

    Args:
        market: Two-letter market code (e.g. "US", "CN", "JP").
        date:   Date string in YYYYMMDD format. Defaults to today.
                Falls back to most recent available snapshot if
                the specified date file does not exist.

    Returns:
        DataFrame with columns: date, ticker, open, high, low, close,
        volume, return_1d, return_5d, return_20d, volume_ratio.
        Empty DataFrame if no snapshot is available.
    """
    if date is None:
        date = datetime.now().strftime("%Y%m%d")
    path = get_data_path("markets", market, f"market_daily_{date}.parquet")
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    # Fallback: most recent available snapshot
    return _latest_snapshot(market)


def get_market_history(market: str, days: int = 60) -> pd.DataFrame:
    """Return multi-day OHLCV history for all tickers in a market.

    Concatenates the most recent N daily snapshots into one sorted
    DataFrame. Duplicate (ticker, date) pairs are deduplicated, keeping
    the latest version.

    Args:
        market: Two-letter market code (e.g. "US", "CN", "JP").
        days:   Maximum number of most-recent daily files to include.

    Returns:
        DataFrame with columns: date, ticker, open, high, low, close,
        volume, return_1d, return_5d, return_20d, volume_ratio, sorted
        by ticker then date ascending. Empty DataFrame if no files found.
    """
    pattern = get_data_path("markets", market, "market_daily_*.parquet")
    files = sorted(glob.glob(pattern))[-days:]
    if not files:
        return pd.DataFrame()
    try:
        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(["ticker", "date"])
        df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def get_latest_signals(market: str) -> pd.DataFrame:
    """Return the most recent signal row for every ticker in a market.

    Args:
        market: Two-letter market code (e.g. "US", "CN", "JP").

    Returns:
        DataFrame with one row per ticker and columns: date, ticker,
        close, return_1d, return_5d, return_20d, volume_ratio.
        Empty DataFrame if no snapshot is available.
    """
    df = _latest_snapshot(market)
    if df.empty or "ticker" not in df.columns:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return (
        df.sort_values("date")
        .groupby("ticker", sort=False)
        .tail(1)
        .reset_index(drop=True)
    )


def get_available_snapshot_dates(market: str) -> list[str]:
    """Return all available daily snapshot dates for a market, oldest first.

    Args:
        market: Two-letter market code (e.g. "US", "CN", "JP").

    Returns:
        Sorted list of YYYYMMDD strings. Empty list if none found.
    """
    pattern = get_data_path("markets", market, "market_daily_*.parquet")
    files = sorted(glob.glob(pattern))
    dates = []
    for f in files:
        base = os.path.basename(f)  # market_daily_YYYYMMDD.parquet
        parts = base.replace(".parquet", "").split("_")
        if parts:
            dates.append(parts[-1])
    return dates


# ---------------------------------------------------------------------------
# Market cap
# ---------------------------------------------------------------------------

def get_market_cap(market: str) -> pd.DataFrame:
    """Return the most recent market capitalisation data for a market.

    Args:
        market: Two-letter market code (e.g. "US", "CN", "JP").

    Returns:
        DataFrame with columns: ticker, market_cap.
        Empty DataFrame if file not written for this market.
    """
    try:
        return pd.read_parquet(get_data_path("markets", market, "market_cap.parquet"))
    except Exception:
        return pd.DataFrame()


def get_market_cap_map(market: str) -> dict[str, float | None]:
    """Return a ticker-to-market-cap mapping for O(1) lookups.

    Args:
        market: Two-letter market code (e.g. "US", "CN", "JP").

    Returns:
        Dict mapping ticker string to market_cap float (or None).
        Empty dict if no file exists.
    """
    df = get_market_cap(market)
    if df.empty or "ticker" not in df.columns or "market_cap" not in df.columns:
        return {}
    return dict(zip(df["ticker"], df["market_cap"].where(df["market_cap"].notna(), None)))


# ---------------------------------------------------------------------------
# Indices
# ---------------------------------------------------------------------------

def get_indices(market: str, days: int | None = None) -> pd.DataFrame:
    """Return index price history for a market.

    Args:
        market: Two-letter market code (e.g. "US", "CN", "JP").
        days:   If provided, return only rows from the last N days.

    Returns:
        DataFrame with columns: date, symbol, name, close, change_pct,
        sorted by symbol then date ascending. Empty DataFrame if not found.
    """
    try:
        df = pd.read_parquet(get_data_path("markets", market, "indices.parquet"))
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if days is not None:
            cutoff = df["date"].max() - pd.Timedelta(days=days)
            df = df[df["date"] >= cutoff]
        return df.sort_values(["symbol", "date"]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def get_latest_index_values(market: str) -> pd.DataFrame:
    """Return the most recent closing value for each index in a market.

    Args:
        market: Two-letter market code (e.g. "US", "CN", "JP").

    Returns:
        DataFrame with one row per index: symbol, name, close,
        change_pct, date. Empty DataFrame if not available.
    """
    df = get_indices(market)
    if df.empty:
        return df
    return (
        df.sort_values("date")
        .groupby("symbol", sort=False)
        .tail(1)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

def get_news(market: str, matched_only: bool = False) -> pd.DataFrame:
    """Return company news articles for a market.

    Args:
        market:       Two-letter market code (e.g. "US", "CN", "JP").
        matched_only: If True, return only rows where hit_count > 0.

    Returns:
        DataFrame with columns: date, ticker, title, source, hit_count,
        keywords_matched. Empty DataFrame if file not found.
    """
    try:
        df = pd.read_parquet(get_data_path("markets", market, "news.parquet"))
        if matched_only and "hit_count" in df.columns:
            df = df[df["hit_count"] > 0]
        return df
    except Exception:
        return pd.DataFrame()


def get_news_for_ticker(
    market: str, ticker: str, matched_only: bool = False
) -> pd.DataFrame:
    """Return all news articles for a single ticker in a market.

    Args:
        market:       Two-letter market code (e.g. "US", "CN", "JP").
        ticker:       Ticker symbol as it appears in the universe.
        matched_only: If True, return only keyword-matched articles.

    Returns:
        DataFrame with columns: date, ticker, title, source, hit_count,
        keywords_matched. Empty DataFrame if no articles found.
    """
    df = get_news(market, matched_only=matched_only)
    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame()
    return df[df["ticker"] == ticker].reset_index(drop=True)


def get_news_counts(market: str) -> pd.DataFrame:
    """Return per-ticker article counts and keyword hit totals for a market.

    Args:
        market: Two-letter market code (e.g. "US", "CN", "JP").

    Returns:
        DataFrame with columns: ticker, total_articles, keyword_hits.
        Empty DataFrame if no news file exists.
    """
    df = get_news(market)
    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame()
    hit_col = "hit_count" if "hit_count" in df.columns else None
    agg = {"ticker": "count"}
    if hit_col:
        agg[hit_col] = "sum"
    result = df.groupby("ticker").agg(
        total_articles=("ticker", "count"),
        **({} if hit_col is None else {"keyword_hits": (hit_col, "sum")}),
    ).reset_index()
    return result


# ---------------------------------------------------------------------------
# Capital flow
# ---------------------------------------------------------------------------

def get_capital_flow(market: str, days: int | None = None) -> pd.DataFrame:
    """Return capital flow data for a market.

    For CN this is northbound Stock Connect flow. For other markets it
    is an ETF-proxy volume signal. Returns empty DataFrame for markets
    with no configured capital flow source.

    Args:
        market: Two-letter market code (e.g. "CN", "HK", "JP").
        days:   If provided, return only the most recent N rows by date.

    Returns:
        DataFrame with columns: date, flow_type, net_flow,
        net_flow_proxy. Empty DataFrame if no flow data written.
    """
    try:
        df = pd.read_parquet(get_data_path("markets", market, "capital_flow.parquet"))
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")
        if days is not None:
            df = df.tail(days)
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def get_latest_capital_flow(market: str) -> dict:
    """Return the most recent capital flow entry for a market.

    Args:
        market: Two-letter market code (e.g. "CN", "HK", "JP").

    Returns:
        Dict with keys: date, flow_type, net_flow, net_flow_proxy.
        Empty dict if no flow data is available.
    """
    df = get_capital_flow(market)
    if df.empty:
        return {}
    return df.iloc[-1].to_dict()


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

def get_market_alerts(market: str) -> list[dict]:
    """Return all alerts generated for a market on the most recent run.

    Args:
        market: Two-letter market code (e.g. "US", "CN", "JP").

    Returns:
        List of alert dicts, each containing at minimum: date, market,
        alert_type, ticker, signal. Empty list if file not found.
    """
    try:
        path = get_data_path("markets", market, "alerts.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def get_market_alerts_by_type(market: str, alert_type: str) -> list[dict]:
    """Return alerts of a specific type for a market.

    Args:
        market:     Two-letter market code (e.g. "US", "CN", "JP").
        alert_type: Alert category to filter on (e.g. "volume_spike",
                    "price_alert", "gap_alert", "news_spike").

    Returns:
        Filtered list of alert dicts. Empty list if none match.
    """
    return [a for a in get_market_alerts(market) if a.get("alert_type") == alert_type]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _latest_snapshot(market: str) -> pd.DataFrame:
    """Load the most recent daily snapshot file for a market."""
    pattern = get_data_path("markets", market, "market_daily_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    try:
        return pd.read_parquet(files[-1])
    except Exception:
        return pd.DataFrame()
