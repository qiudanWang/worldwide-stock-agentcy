"""Orchestrate financial data refresh for all markets.

Saves per-market financials.parquet files to data/markets/{MARKET}/financials.parquet.
Schema: long format — one row per (ticker, period_end).
"""

import os
import pandas as pd
from src.common.config import get_settings
from src.common.logger import get_logger
from src.financials.cn_financials import fetch_cn_financials_batch
from src.financials.yf_financials import fetch_yf_financials_batch
from src.common.tracing import observe

log = get_logger("financials.normalize")

_DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "markets")

_YF_MARKETS = ["US", "HK", "JP", "KR", "TW", "DE", "FR", "UK", "IN", "BR", "AU", "SA"]


@observe(name="_universe_tickers", type="span")
def _universe_tickers(market: str) -> list:
    path = os.path.join(_DATA_ROOT, market, "universe.parquet")
    if not os.path.exists(path):
        return []
    df = pd.read_parquet(path)
    return df["ticker"].astype(str).tolist() if "ticker" in df.columns else []


@observe(name="_save", type="span")
def _save(df: pd.DataFrame, market: str):
    if df.empty:
        return
    path = os.path.join(_DATA_ROOT, market, "financials.parquet")
    df.to_parquet(path, index=False)
    log.info(f"[{market}] Saved financials.parquet — {df['ticker'].nunique()} stocks, {len(df)} periods")


@observe(name="refresh_cn_financials", type="span")
def refresh_cn_financials():
    """Refresh CN financials from akshare."""
    tickers = _universe_tickers("CN")
    if not tickers:
        log.warning("[CN] No universe tickers found")
        return pd.DataFrame()
    log.info(f"[CN] Fetching financials for {len(tickers)} tickers...")
    df = fetch_cn_financials_batch(tickers)
    _save(df, "CN")
    return df


@observe(name="refresh_market_financials", type="span")
def refresh_market_financials(market: str):
    """Refresh financials for a single non-CN market via yfinance."""
    tickers = _universe_tickers(market)
    if not tickers:
        log.warning(f"[{market}] No universe tickers found")
        return pd.DataFrame()
    log.info(f"[{market}] Fetching financials for {len(tickers)} tickers...")
    df = fetch_yf_financials_batch(tickers, market)
    _save(df, market)
    return df


@observe(name="refresh_all_financials", type="span")
def refresh_all_financials(markets: list = None):
    """Refresh financials for all markets (or a subset).

    Args:
        markets: list of market codes to refresh. Defaults to all.
    """
    if markets is None:
        markets = ["CN"] + _YF_MARKETS

    results = {}
    for mkt in markets:
        try:
            if mkt == "CN":
                results[mkt] = refresh_cn_financials()
            else:
                results[mkt] = refresh_market_financials(mkt)
        except Exception as e:
            log.error(f"[{mkt}] refresh_all_financials failed: {e}")
            results[mkt] = pd.DataFrame()

    ok = [m for m, df in results.items() if not df.empty]
    log.info(f"Financials refresh complete: {len(ok)}/{len(markets)} markets succeeded")
    return results


# ── Backward-compat shim ──────────────────────────────────────────────────
@observe(name="build_financial_snapshot", type="span")
def build_financial_snapshot(cn_tickers=None, us_tickers=None):
    """Legacy entry point. Runs CN + US refresh."""
    refresh_cn_financials()
    refresh_market_financials("US")
