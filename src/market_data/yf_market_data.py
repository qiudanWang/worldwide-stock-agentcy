"""Generalized yfinance market data fetcher for all non-CN markets.

Uses batch downloads (much faster than one-by-one Ticker.history calls).
"""

import time
import pandas as pd
import yfinance as yf
from src.common.config import get_settings
from src.common.logger import get_logger
from src.common.rate_limiter import yf_limiter

log = get_logger("market.yf")

_BATCH_SIZE = 50


def _yf_download_with_retry(symbols_str, period, max_retries=3, **kwargs):
    """yf.download with exponential backoff on rate limit errors."""
    for attempt in range(max_retries):
        try:
            with yf_limiter:
                return yf.download(symbols_str, period=period, **kwargs)
        except Exception as e:
            msg = str(e)
            if "rate" in msg.lower() or "429" in msg or "too many" in msg.lower():
                wait = 30 * (2 ** attempt)  # 30s, 60s, 120s
                log.warning(f"Rate limited, waiting {wait}s before retry {attempt+1}/{max_retries}")
                time.sleep(wait)
            else:
                raise
    # Final attempt without catching
    with yf_limiter:
        return yf.download(symbols_str, period=period, **kwargs)


def _build_yf_symbol(ticker: str, market: str) -> str:
    """Build the correct Yahoo Finance symbol for a given market."""
    if market == "HK":
        # HK tickers are numeric; Yahoo needs zero-padded 4-digit + .HK
        return ticker.zfill(4) + ".HK"
    # All other markets already have the correct yf_symbol in the universe
    return ticker


def _parse_batch(raw: pd.DataFrame, yf_syms: list, ticker_map: dict, market: str) -> list:
    """Parse a yfinance batch download result into a list of DataFrames."""
    results = []
    if raw.empty:
        return results

    if isinstance(raw.columns, pd.MultiIndex):
        ticker_level = "Ticker" if "Ticker" in raw.columns.names else 1
        available = set(raw.columns.get_level_values(ticker_level).unique())
        for yf_sym in yf_syms:
            if yf_sym not in available:
                continue
            try:
                df = raw.xs(yf_sym, axis=1, level=ticker_level).copy().reset_index()
                df.columns = [c.lower() for c in df.columns]
                if "date" not in df.columns:
                    df = df.rename(columns={df.columns[0]: "date"})
                if df.empty or "close" not in df.columns or df["close"].isna().all():
                    continue
                df["ticker"] = ticker_map[yf_sym]
                df["market"] = market
                df["volume"] = df.get("volume", 0)
                df["turnover"] = df["close"] * df["volume"]
                cols = ["date", "ticker", "market", "open", "high", "low",
                        "close", "volume", "turnover"]
                results.append(df[[c for c in cols if c in df.columns]])
            except Exception as e:
                log.debug(f"  Skip {yf_sym}: {e}")
    else:
        # Single ticker — flat DataFrame
        df = raw.reset_index()
        df.columns = [c.lower() for c in df.columns]
        if "date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "date"})
        if not df.empty and "close" in df.columns and not df["close"].isna().all():
            yf_sym = yf_syms[0]
            df["ticker"] = ticker_map.get(yf_sym, yf_sym)
            df["market"] = market
            df["volume"] = df.get("volume", 0)
            df["turnover"] = df["close"] * df["volume"]
            cols = ["date", "ticker", "market", "open", "high", "low",
                    "close", "volume", "turnover"]
            results.append(df[[c for c in cols if c in df.columns]])
    return results


def fetch_yf_stock_history(ticker, yf_symbol=None, market="US", days=None):
    """Fetch daily OHLCV for a single stock via yfinance."""
    if days is None:
        days = get_settings()["market_data"].get("history_days", 60)
    symbol = yf_symbol or _build_yf_symbol(ticker, market)
    try:
        raw = _yf_download_with_retry(symbol, period=f"{days}d", progress=False,
                                      auto_adjust=True, actions=False)
        results = _parse_batch(raw, [symbol], {symbol: ticker}, market)
        return results[0] if results else pd.DataFrame()
    except Exception as e:
        log.warning(f"Failed to fetch {symbol}: {e}")
        return pd.DataFrame()


def fetch_yf_batch(tickers, market="US", ticker_suffix="", days=None):
    """Fetch history for all tickers via yfinance batch downloads."""
    if not tickers:
        return pd.DataFrame()
    if days is None:
        days = get_settings()["market_data"].get("history_days", 60)

    all_data = []
    total = len(tickers)

    for batch_start in range(0, total, _BATCH_SIZE):
        batch = tickers[batch_start:batch_start + _BATCH_SIZE]

        # Build yf symbols
        if ticker_suffix:
            yf_syms = [f"{t}{ticker_suffix}" for t in batch]
        else:
            yf_syms = [_build_yf_symbol(t, market) for t in batch]
        ticker_map = {yf: t for yf, t in zip(yf_syms, batch)}

        if batch_start > 0:
            time.sleep(3)  # 3s between batches to stay within yfinance rate limits
        log.info(f"  [{market}] Fetching batch {batch_start + 1}-{min(batch_start + _BATCH_SIZE, total)}/{total}")
        try:
            raw = _yf_download_with_retry(
                " ".join(yf_syms),
                period=f"{days}d",
                progress=False,
                auto_adjust=True,
                actions=False,
                group_by="ticker",
            )
            results = _parse_batch(raw, yf_syms, ticker_map, market)
            all_data.extend(results)
        except Exception as e:
            log.warning(f"  [{market}] Batch {batch_start}-{batch_start + _BATCH_SIZE} failed: {e}, falling back to single fetches")
            for ticker in batch:
                df = fetch_yf_stock_history(ticker, None, market, days)
                if not df.empty:
                    all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    log.info(f"[{market}] Fetched {result['ticker'].nunique()}/{total} tickers")
    return result
