"""
Global (cross-market) data access functions for the Global Tech Market pipeline.

These functions read data produced by the GlobalAgent and cover macro indicators,
merged universes, sector performance, index correlations, and the merged global
alerts feed. All functions return empty containers on missing files — never raise.
"""

from __future__ import annotations

import json

import pandas as pd

from src.common.config import get_data_path, load_yaml
from src.data_api.market import get_indices as _market_indices
from src.data_api.market import get_latest_signals as _market_signals


# ---------------------------------------------------------------------------
# Markets config
# ---------------------------------------------------------------------------

def list_markets() -> list[str]:
    """Return the list of all configured market codes.

    Returns:
        List of two-letter market code strings, e.g.
        ["CN", "US", "HK", "JP", "IN", "UK", "DE", "FR", "KR", "TW", "AU", "BR", "SA"].
    """
    try:
        return list(load_yaml("markets.yaml")["markets"].keys())
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Universe master
# ---------------------------------------------------------------------------

def get_universe_master() -> pd.DataFrame:
    """Return the merged universe of all markets.

    Reads the master universe produced by the GlobalAgent, which concatenates
    per-market universe files into one table with a 'market' column added.

    Returns:
        DataFrame with columns: ticker, name, sector, industry, currency,
        yf_symbol, market. Empty DataFrame if not yet built.
    """
    try:
        return pd.read_parquet(get_data_path("global", "universe_master.parquet"))
    except Exception:
        return pd.DataFrame()


def get_universe_for_markets(markets: list[str]) -> pd.DataFrame:
    """Return the master universe filtered to a subset of markets.

    Args:
        markets: List of two-letter market codes to include.

    Returns:
        Filtered DataFrame. Empty DataFrame if no matching rows exist.
    """
    df = get_universe_master()
    if df.empty or "market" not in df.columns:
        return pd.DataFrame()
    return df[df["market"].isin(markets)].reset_index(drop=True)


def search_universe(query: str, markets: list[str] | None = None) -> pd.DataFrame:
    """Search the master universe by ticker or company name substring.

    Args:
        query:   Case-insensitive substring matched against ticker and name.
        markets: If provided, restrict search to these market codes.

    Returns:
        DataFrame of matching rows: ticker, name, sector, currency, market.
        Empty DataFrame if no matches.
    """
    df = get_universe_master()
    if df.empty:
        return pd.DataFrame()
    if markets:
        df = df[df["market"].isin(markets)]
    mask = (
        df.get("ticker", pd.Series(dtype=str)).str.contains(query, case=False, na=False)
        | df.get("name", pd.Series(dtype=str)).str.contains(query, case=False, na=False)
    )
    cols = [c for c in ["ticker", "name", "sector", "currency", "market"] if c in df.columns]
    return df.loc[mask, cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Macro indicators
# ---------------------------------------------------------------------------

def get_macro_indicators(
    category: str | None = None,
    days: int | None = None,
) -> pd.DataFrame:
    """Return macro indicator time-series data.

    Covers commodities (oil, gold, BTC), currencies (DXY, FX pairs),
    US rates (FRED), World Bank GDP/CPI, and CN macro series.

    Args:
        category: If provided, filter to this category string
                  ("fred", "cn_macro", "world_bank", "commodities",
                  "currencies", "sentiment", "crypto"). None = all.
        days:     If provided, return only rows from the last N days.

    Returns:
        DataFrame with columns: date, indicator, value, change_pct,
        category. Empty DataFrame if file not found.
    """
    try:
        df = pd.read_parquet(get_data_path("global", "macro_indicators.parquet"))
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if category and "category" in df.columns:
            df = df[df["category"] == category]
        if days is not None:
            cutoff = df["date"].max() - pd.Timedelta(days=days)
            df = df[df["date"] >= cutoff]
        return df.sort_values(["indicator", "date"]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def get_macro_latest(indicators: list[str] | None = None) -> dict:
    """Return the most recent value for all macro indicators.

    Reads macro_latest.json which stores the most recently observed value
    for every indicator without requiring a full parquet scan.

    Args:
        indicators: If provided, return only these indicator names.
                    None = return all.

    Returns:
        Dict mapping indicator name to sub-dict with keys:
        value, change_pct, date, symbol. Empty dict if file not found.
    """
    try:
        path = get_data_path("global", "macro_latest.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if indicators:
            data = {k: v for k, v in data.items() if k in indicators}
        return data
    except Exception:
        return {}


def get_macro_indicator_series(indicator: str) -> pd.DataFrame:
    """Return the full time series for a single named macro indicator.

    Args:
        indicator: Exact indicator name as stored, e.g. "VIX", "Gold",
                   "EUR/USD", "Fed Funds Rate",
                   "GDP growth (annual %): United States".

    Returns:
        DataFrame with columns: date, value, change_pct sorted by date
        ascending. Empty DataFrame if the indicator is not found.
    """
    df = get_macro_indicators()
    if df.empty or "indicator" not in df.columns:
        return pd.DataFrame()
    subset = df[df["indicator"] == indicator][["date", "value", "change_pct"]]
    return subset.sort_values("date").reset_index(drop=True)


def list_macro_indicators() -> list[str]:
    """Return all available macro indicator names.

    Returns:
        Sorted list of indicator name strings.
    """
    df = get_macro_indicators()
    if df.empty or "indicator" not in df.columns:
        return []
    return sorted(df["indicator"].unique().tolist())


# ---------------------------------------------------------------------------
# Sector performance
# ---------------------------------------------------------------------------

def get_sector_performance(
    markets: list[str] | None = None,
    min_stock_count: int = 2,
) -> pd.DataFrame:
    """Return cross-market sector performance data.

    Args:
        markets:         If provided, filter to these market codes.
        min_stock_count: Exclude sectors with fewer than this many stocks.

    Returns:
        DataFrame with columns: market, sector, avg_return_1d,
        avg_return_5d, avg_return_20d, stock_count.
        Empty DataFrame if file not found.
    """
    try:
        df = pd.read_parquet(get_data_path("global", "sector_performance.parquet"))
        if markets and "market" in df.columns:
            df = df[df["market"].isin(markets)]
        if "stock_count" in df.columns:
            df = df[df["stock_count"] >= min_stock_count]
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def get_top_sectors(
    n: int = 5,
    horizon: str = "1d",
    markets: list[str] | None = None,
) -> pd.DataFrame:
    """Return the top-performing sectors globally by average return.

    Args:
        n:       Number of top sectors to return.
        horizon: Return horizon — "1d", "5d", or "20d".
        markets: If provided, restrict to these market codes before ranking.

    Returns:
        DataFrame with columns: market, sector, avg_return_{horizon},
        stock_count. Empty DataFrame if no data available.
    """
    df = get_sector_performance(markets=markets)
    col = f"avg_return_{horizon}"
    if df.empty or col not in df.columns:
        return pd.DataFrame()
    return df.nlargest(n, col).reset_index(drop=True)


def get_sector_performance_for_ticker(ticker: str) -> dict:
    """Return the sector performance row for the sector a ticker belongs to.

    Looks up the ticker in the master universe to find its market and
    sector, then returns the matching sector performance row.

    Args:
        ticker: Ticker symbol to look up (any market).

    Returns:
        Dict with keys: market, sector, avg_return_1d, avg_return_5d,
        avg_return_20d, stock_count. Empty dict if not found.
    """
    universe = get_universe_master()
    if universe.empty or "ticker" not in universe.columns:
        return {}
    row = universe[universe["ticker"] == ticker]
    if row.empty:
        return {}
    market = row.iloc[0].get("market", "")
    sector = row.iloc[0].get("sector", "")
    perf = get_sector_performance(markets=[market])
    if perf.empty:
        return {}
    match = perf[(perf["market"] == market) & (perf["sector"] == sector)]
    if match.empty:
        return {}
    return match.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------

def get_correlations() -> dict:
    """Return the cross-market index correlation matrices.

    Returns:
        Dict with structure: {"matrix": {symbol: {symbol: float}},
        "symbols": [...], "computed_at": str}.
        Empty dict if file not found.
    """
    try:
        path = get_data_path("global", "correlations.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_correlation_pair(symbol_a: str, symbol_b: str) -> float | None:
    """Return the pairwise correlation between two index symbols.

    Args:
        symbol_a: First index symbol (e.g. "^GSPC", "^HSI").
        symbol_b: Second index symbol (e.g. "^N225", "^FTSE").

    Returns:
        Float correlation in [-1, 1], or None if the pair is not found.
    """
    data = get_correlations()
    matrix = data.get("matrix", {})
    return matrix.get(symbol_a, {}).get(symbol_b)


# ---------------------------------------------------------------------------
# Global alerts
# ---------------------------------------------------------------------------

def get_global_alerts(
    alert_types: list[str] | None = None,
    markets: list[str] | None = None,
) -> list[dict]:
    """Return the merged global alerts from the most recent pipeline run.

    Covers all alert types: volume_spike, price_alert, gap_alert,
    news_spike, capital_flow, macro_alert, sector_rotation,
    cross_market_divergence, index_breakout, peer_divergence.

    Args:
        alert_types: If provided, return only alerts matching one of
                     these type strings. None = all types.
        markets:     If provided, return only alerts for these market
                     codes. None = all markets.

    Returns:
        List of alert dicts, each with at minimum: date, market,
        alert_type, ticker, signal. Empty list if no file exists.
    """
    try:
        path = get_data_path("global", "alerts.json")
        with open(path, "r", encoding="utf-8") as f:
            alerts = json.load(f)
        if not isinstance(alerts, list):
            return []
        if alert_types:
            alerts = [a for a in alerts if a.get("alert_type") in alert_types]
        if markets:
            alerts = [a for a in alerts if a.get("market") in markets]
        return alerts
    except Exception:
        return []


def get_global_alert_summary() -> dict:
    """Return a count summary of global alerts by type and market.

    Returns:
        Dict: {"by_type": {alert_type: count}, "by_market": {market: count},
        "total": int}. Returns zero counts if no alerts file exists.
    """
    alerts = get_global_alerts()
    by_type: dict[str, int] = {}
    by_market: dict[str, int] = {}
    for a in alerts:
        t = a.get("alert_type", "unknown")
        m = a.get("market", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
        by_market[m] = by_market.get(m, 0) + 1
    return {"by_type": by_type, "by_market": by_market, "total": len(alerts)}


def get_alerts_for_ticker(ticker: str) -> list[dict]:
    """Return all global alerts that reference a specific ticker.

    Args:
        ticker: Ticker symbol to search for (any market).

    Returns:
        List of alert dicts sorted by date descending.
        Empty list if no alerts reference this ticker.
    """
    alerts = [a for a in get_global_alerts() if a.get("ticker") == ticker]
    return sorted(alerts, key=lambda a: a.get("date", ""), reverse=True)


# ---------------------------------------------------------------------------
# All-markets convenience
# ---------------------------------------------------------------------------

def get_all_latest_signals() -> pd.DataFrame:
    """Return the most recent signal row for every ticker across all markets.

    Calls get_latest_signals() for each configured market and concatenates
    the results. Adds a 'market' column if not already present.

    Returns:
        DataFrame with columns: date, ticker, market, close, return_1d,
        return_5d, return_20d, volume_ratio. One row per ticker.
        Empty DataFrame if no market data is available.
    """
    frames = []
    for market in list_markets():
        df = _market_signals(market)
        if df.empty:
            continue
        if "market" not in df.columns:
            df = df.copy()
            df["market"] = market
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_all_indices(days: int = 60) -> pd.DataFrame:
    """Return index price history for all markets combined.

    Args:
        days: Return only rows from the last N days.

    Returns:
        DataFrame with columns: date, symbol, name, market, close,
        change_pct. Sorted by symbol then date ascending.
        Empty DataFrame if no index data is available.
    """
    frames = []
    for market in list_markets():
        df = _market_indices(market, days=days)
        if df.empty:
            continue
        if "market" not in df.columns:
            df = df.copy()
            df["market"] = market
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )
