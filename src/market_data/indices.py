"""Fetch major market indices — config-driven for all markets."""

import os
import yfinance as yf
import pandas as pd
from src.common.config import load_yaml, get_data_path
from src.common.logger import get_logger
from src.common.rate_limiter import yf_limiter

log = get_logger("market.indices")


def _load_prev_close_lookup():
    """Build {symbol: prev_close} from stored indices.parquet files as fallback."""
    lookup = {}
    try:
        import glob
        for p in glob.glob(get_data_path("markets", "*", "indices.parquet")):
            df = pd.read_parquet(p)
            if df.empty or "symbol" not in df.columns or "close" not in df.columns:
                continue
            df = df.sort_values("date")
            # For each symbol keep the second-to-last row as prev_close
            for sym, grp in df.groupby("symbol"):
                closes = grp["close"].dropna().tolist()
                if len(closes) >= 2:
                    lookup[sym] = closes[-2]
                elif len(closes) == 1:
                    # Only one stored row — use it as prev_close since yfinance
                    # is returning today's live price and the stored value is yesterday's
                    lookup[sym] = closes[-1]
    except Exception:
        pass
    return lookup


def _get_all_indices():
    """Build index list from markets.yaml config."""
    try:
        markets_cfg = load_yaml("markets.yaml")["markets"]
    except Exception:
        markets_cfg = {}

    indices_by_market = {}
    for market_code, cfg in markets_cfg.items():
        indices_by_market[market_code] = [
            {"symbol": idx["symbol"], "name": idx["name"]}
            for idx in cfg.get("indices", [])
        ]
    return indices_by_market


def fetch_indices(markets=None):
    """Fetch latest index values and daily change.

    Args:
        markets: Optional list of market codes to fetch. None = all markets.

    Returns list of dicts with: market, name, close, change_pct
    """
    all_indices = _get_all_indices()
    if markets:
        all_indices = {m: v for m, v in all_indices.items() if m in markets}

    prev_close_lookup = _load_prev_close_lookup()

    results = []
    for market, indices in all_indices.items():
        for idx in indices:
            try:
                with yf_limiter:
                    t = yf.Ticker(idx["symbol"])
                    h = t.history(period="5d")
                if h.empty:
                    continue
                close = h["Close"].iloc[-1]
                change_pct = None
                if len(h) >= 2:
                    prev = h["Close"].iloc[-2]
                    change_pct = (close - prev) / prev * 100
                elif idx["symbol"] in prev_close_lookup:
                    # yfinance only returned 1 row — use stored previous close
                    prev = prev_close_lookup[idx["symbol"]]
                    change_pct = (close - prev) / prev * 100
                results.append({
                    "market": market,
                    "symbol": idx["symbol"],
                    "name": idx["name"],
                    "close": round(close, 2),
                    "change_pct": round(change_pct, 2) if change_pct is not None else None,
                })
            except Exception as e:
                log.warning(f"Failed to fetch {idx['name']}: {e}")

    return results
