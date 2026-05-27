"""
Backtest Data Loader
====================
Fetches and caches per-ticker OHLCV history for backtesting.
Cache path: data/backtest/history/{MARKET}/{TICKER}.parquet  (24h TTL)
Delta fetch: on subsequent runs, only new trading days are appended.
"""

import os
import time
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta

from src.common.config import get_data_path
from src.common.logger import get_logger
from src.common.tracing import observe

log = get_logger("backtest.data")

CACHE_TTL_HOURS = 24
HISTORY_YEARS   = 2   # how far back to fetch on first run

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
    keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    if "close" not in keep:
        return pd.DataFrame()
    return df[keep].dropna(subset=["close"])


def _merge_delta(existing: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    """Append new rows to existing, deduplicate by date, sort."""
    combined = pd.concat([existing, df_new], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    return combined.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)


@observe(name="_fetch_yf_batch", type="tool")
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


@observe(name="_fetch_akshare_single", type="tool")
def _fetch_akshare_single(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch CN A-share OHLCV via akshare."""
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist(
            symbol=ticker,
            period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust="hfq",
        )
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "最高": "high",
            "最低": "low", "收盘": "close", "成交量": "volume",
        })
        return _standardise(df)
    except Exception as e:
        log.debug(f"akshare fetch failed for {ticker}: {e}")
        return pd.DataFrame()


@observe(name="load_universe", type="tool")
def load_universe(market: str) -> pd.DataFrame:
    """Load universe.parquet for the given market."""
    path = get_data_path("markets", market, "universe.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


@observe(name="load_market_history_batch", type="tool")
def load_market_history_batch(
    market: str,
    start_date: str,
    end_date: str,
    progress_callback=None,
) -> dict[str, pd.DataFrame]:
    """
    Load OHLCV history for all tickers in a market.

    Three cache states per ticker:
      - Fresh (< 24h): use as-is, no network call.
      - Stale (file exists, > 24h): fetch only new days since last cached date (delta).
      - Missing: fetch full history (HISTORY_YEARS years).

    Returns {ticker: df} — missing/failed tickers are absent from the dict.
    """
    universe = load_universe(market)
    if universe.empty:
        log.warning(f"[{market}] No universe found")
        return {}

    tickers = universe["ticker"].tolist()
    total   = len(tickers)
    result  = {}
    need_full  = []   # (ticker,) — no cache file, fetch full history
    need_delta = []   # (ticker, delta_start, existing_df) — stale cache, fetch delta

    today = datetime.now().strftime("%Y-%m-%d")

    for tk in tickers:
        cp = _cache_path(market, tk)
        if _is_cache_fresh(cp):
            try:
                result[tk] = pd.read_parquet(cp)
            except Exception:
                need_full.append(tk)
        elif os.path.exists(cp):
            try:
                existing = pd.read_parquet(cp)
                last_date = pd.to_datetime(existing["date"]).max()
                delta_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
                if delta_start >= today:
                    # Already up to date — just reset TTL
                    result[tk] = existing
                    os.utime(cp, None)
                else:
                    need_delta.append((tk, delta_start, existing))
            except Exception:
                need_full.append(tk)
        else:
            need_full.append(tk)

    log.info(
        f"[{market}] {len(result)} from cache, "
        f"{len(need_delta)} delta, {len(need_full)} full fetch"
    )

    if not need_full and not need_delta:
        return result

    # ── CN: akshare individual calls ──────────────────────────────────────
    if market == "CN":
        for tk in need_full:
            df = _fetch_akshare_single(tk, start_date, end_date)
            if not df.empty:
                df.to_parquet(_cache_path(market, tk), index=False)
                result[tk] = df
            if progress_callback:
                progress_callback(len(result), total)
            time.sleep(0.3)

        for tk, delta_start, existing in need_delta:
            df_new = _fetch_akshare_single(tk, delta_start, end_date)
            cp = _cache_path(market, tk)
            if not df_new.empty:
                combined = _merge_delta(existing, df_new)
                combined.to_parquet(cp, index=False)
                result[tk] = combined
            else:
                result[tk] = existing
                os.utime(cp, None)  # reset TTL even if no new data
            if progress_callback:
                progress_callback(len(result), total)
            time.sleep(0.3)

    # ── All others: yfinance bulk ─────────────────────────────────────────
    else:
        suffix  = _YF_SUFFIX.get(market, "")
        uni_map = dict(zip(universe["ticker"], universe.get("yf_symbol", universe["ticker"])))
        chunk_size = 200

        def _build_yf_map(tks):
            m = {}
            for tk in tks:
                yf_sym = uni_map.get(tk, tk)
                if suffix and not str(yf_sym).endswith(suffix):
                    yf_sym = f"{yf_sym}{suffix}"
                m[yf_sym] = tk
            return m

        # Full fetches
        if need_full:
            yf_map  = _build_yf_map(need_full)
            yf_syms = list(yf_map.keys())
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

        # Delta fetches — group by delta_start so tickers with the same last date
        # are fetched in a single yfinance batch call
        if need_delta:
            groups = defaultdict(list)
            for tk, delta_start, existing in need_delta:
                groups[delta_start].append((tk, existing))

            for delta_start, items in sorted(groups.items()):
                tks    = [tk for tk, _ in items]
                ex_map = {tk: ex for tk, ex in items}
                yf_map  = _build_yf_map(tks)
                yf_syms = list(yf_map.keys())

                for i in range(0, len(yf_syms), chunk_size):
                    chunk   = yf_syms[i:i + chunk_size]
                    fetched = _fetch_yf_batch(chunk, delta_start, end_date)
                    for yf_sym, df_new in fetched.items():
                        tk       = yf_map.get(yf_sym, yf_sym)
                        existing = ex_map.get(tk, pd.DataFrame())
                        cp       = _cache_path(market, tk)
                        if not df_new.empty and not existing.empty:
                            combined = _merge_delta(existing, df_new)
                            combined.to_parquet(cp, index=False)
                            result[tk] = combined
                        elif not df_new.empty:
                            df_new.to_parquet(cp, index=False)
                            result[tk] = df_new
                        elif not existing.empty:
                            result[tk] = existing
                            os.utime(cp, None)
                    if progress_callback:
                        progress_callback(len(result), total)

    log.info(f"[{market}] History loaded: {len(result)}/{total} tickers")
    return result
