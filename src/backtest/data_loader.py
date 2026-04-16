"""
Backtest Data Loader
====================
Fetches and caches per-ticker OHLCV history for backtesting.
Cache path: data/backtest/history/{MARKET}/{TICKER}.parquet  (24h TTL)
"""

import os
import time
import pandas as pd
from datetime import datetime, timedelta

from src.common.config import get_data_path
from src.common.logger import get_logger

log = get_logger("backtest.data")

CACHE_TTL_HOURS = 24
HISTORY_YEARS   = 2   # how far back to fetch

# yfinance exchange suffixes per market
_YF_SUFFIX = {
    "HK": ".HK", "JP": ".T",  "AU": ".AX", "IN": ".NS",
    "KR": ".KS", "TW": ".TW", "DE": ".DE", "FR": ".PA",
    "UK": ".L",  "BR": ".SA", "SA": ".SR",
}


def _cache_path(market: str, ticker: str) -> str:
    return get_data_path("backtest", "history", market, f"{ticker}.parquet")


def _is_cache_fresh(path: str) -> bool:
    if not os.path.exists(path):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    return age < timedelta(hours=CACHE_TTL_HOURS)


def _standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to lowercase, ensure date column."""
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns and df.index.name and "date" in str(df.index.name).lower():
        df = df.reset_index().rename(columns={df.index.name: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Only select columns that are present; volume is optional
    keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    if "close" not in keep:
        return pd.DataFrame()
    return df[keep].dropna(subset=["close"])


def _fetch_yf_batch(yf_symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Bulk-fetch OHLCV from yfinance. Returns {yf_symbol: df}."""
    import yfinance as yf
    if not yf_symbols:
        return {}
    try:
        raw = yf.download(
            " ".join(yf_symbols),
            start=start, end=end,
            auto_adjust=True, progress=False,
            group_by="ticker",
        )
    except Exception as e:
        log.warning(f"yf.download batch failed: {e}")
        return {}

    result = {}
    if len(yf_symbols) == 1:
        # Single ticker → flat columns
        sym = yf_symbols[0]
        try:
            df = raw.reset_index()
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            result[sym] = _standardise(df)
        except Exception as e:
            log.debug(f"Failed to parse yf data for {sym}: {e}")
    else:
        for sym in yf_symbols:
            try:
                df = raw[sym].dropna(how="all").reset_index()
                result[sym] = _standardise(df)
            except Exception as e:
                log.debug(f"Failed to parse yf data for {sym}: {e}")
    return result


def _fetch_akshare_single(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch CN A-share OHLCV via akshare."""
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist(
            symbol=ticker,
            period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust="hfq",   # back-adjusted
        )
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "最高": "high",
            "最低": "low", "收盘": "close", "成交量": "volume",
        })
        return _standardise(df)
    except Exception as e:
        log.debug(f"akshare fetch failed for {ticker}: {e}")
        return pd.DataFrame()


def load_universe(market: str) -> pd.DataFrame:
    """Load universe.parquet for the given market."""
    path = get_data_path("markets", market, "universe.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_market_history_batch(
    market: str,
    start_date: str,
    end_date: str,
    progress_callback=None,
) -> dict[str, pd.DataFrame]:
    """
    Load OHLCV history for all tickers in a market.
    Returns {ticker: df} — missing/failed tickers are absent from the dict.
    """
    universe = load_universe(market)
    if universe.empty:
        log.warning(f"[{market}] No universe found")
        return {}

    tickers   = universe["ticker"].tolist()
    total     = len(tickers)
    result    = {}
    need_fetch = []   # (ticker, yf_symbol) pairs not in fresh cache

    # Check cache first
    for tk in tickers:
        cp = _cache_path(market, tk)
        if _is_cache_fresh(cp):
            try:
                result[tk] = pd.read_parquet(cp)
            except Exception:
                need_fetch.append(tk)
        else:
            need_fetch.append(tk)

    log.info(f"[{market}] {len(result)} from cache, {len(need_fetch)} to fetch")

    if not need_fetch:
        return result

    # ── CN: akshare individual calls ──────────────────────────────────────
    if market == "CN":
        for i, tk in enumerate(need_fetch):
            df = _fetch_akshare_single(tk, start_date, end_date)
            if not df.empty:
                df.to_parquet(_cache_path(market, tk), index=False)
                result[tk] = df
            if progress_callback:
                progress_callback(len(result), total)
            time.sleep(0.3)   # akshare rate limit

    # ── All others: yfinance bulk ─────────────────────────────────────────
    else:
        suffix    = _YF_SUFFIX.get(market, "")
        yf_map    = {}   # yf_symbol → ticker
        uni_map   = dict(zip(universe["ticker"], universe.get("yf_symbol", universe["ticker"])))

        for tk in need_fetch:
            yf_sym = uni_map.get(tk, tk)
            if suffix and not str(yf_sym).endswith(suffix):
                yf_sym = f"{yf_sym}{suffix}"
            yf_map[yf_sym] = tk

        # Batch in chunks of 200 to stay within yfinance limits
        yf_syms = list(yf_map.keys())
        chunk_size = 200
        for i in range(0, len(yf_syms), chunk_size):
            chunk   = yf_syms[i:i + chunk_size]
            fetched = _fetch_yf_batch(chunk, start_date, end_date)
            for yf_sym, df in fetched.items():
                tk = yf_map.get(yf_sym, yf_sym)
                if not df.empty:
                    df.to_parquet(_cache_path(market, tk), index=False)
                    result[tk] = df
            if progress_callback:
                progress_callback(len(result), total)

    log.info(f"[{market}] History loaded: {len(result)}/{total} tickers")
    return result
