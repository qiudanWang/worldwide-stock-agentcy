"""Watchlist-based universe builder for non-CN markets."""

import pandas as pd
from src.common.config import load_yaml, get_data_path
from src.common.logger import get_logger

log = get_logger("universe.yf")


def build_yf_universe(market, watchlist_config, ticker_suffix=""):
    """Build stock universe from a YAML watchlist file.

    Args:
        market: Market code (e.g., "HK", "JP").
        watchlist_config: YAML config filename (e.g., "hk_watchlist.yaml").
        ticker_suffix: yfinance ticker suffix (e.g., ".HK").

    Returns:
        DataFrame with columns: ticker, name, sector, subsector, market, yf_symbol
    """
    cfg = load_yaml(watchlist_config)

    rows = []
    seen = set()
    for sector, tickers in cfg.items():
        for ticker in tickers:
            ticker_str = str(ticker)
            if ticker_str not in seen:
                yf_symbol = f"{ticker_str}{ticker_suffix}" if ticker_suffix else ticker_str
                rows.append({
                    "ticker": ticker_str,
                    "name": ticker_str,  # will be enriched later
                    "sector": sector,
                    "subsector": "",
                    "market": market,
                    "yf_symbol": yf_symbol,
                })
                seen.add(ticker_str)

    result = pd.DataFrame(rows)
    log.info(f"[{market}] Universe: {len(result)} stocks from {watchlist_config}")
    return result


def enrich_subsector_yf(df: pd.DataFrame, ticker_col: str = "yf_symbol") -> pd.DataFrame:
    """Fetch yfinance industry field for each ticker and store as subsector.

    Only fetches for rows where subsector is empty/missing.
    Uses the yf_symbol column (with exchange suffix) for the API call.
    """
    import yfinance as yf

    if "subsector" not in df.columns:
        df = df.copy()
        df["subsector"] = ""

    needs_enrich = df["subsector"].isna() | (df["subsector"] == "")
    if not needs_enrich.any():
        return df

    symbols = df.loc[needs_enrich, ticker_col].dropna().unique().tolist()
    log.info(f"Enriching subsector for {len(symbols)} tickers via yfinance...")

    industry_map = {}
    # yfinance Tickers batch — faster than individual calls
    try:
        batch = yf.Tickers(" ".join(symbols))
        for sym in symbols:
            try:
                info = batch.tickers[sym].info or {}
                industry_map[sym] = info.get("industry", "")
            except Exception:
                industry_map[sym] = ""
    except Exception:
        for sym in symbols:
            try:
                industry_map[sym] = yf.Ticker(sym).info.get("industry", "")
            except Exception:
                industry_map[sym] = ""

    df = df.copy()
    df.loc[needs_enrich, "subsector"] = df.loc[needs_enrich, ticker_col].map(industry_map).fillna("")
    filled = (df["subsector"] != "").sum()
    log.info(f"Subsector enriched: {filled}/{len(df)} stocks")
    return df


def save_market_universe(df, market):
    """Save a single market's universe to its data directory."""
    path = get_data_path("markets", market, "universe.parquet")
    df.to_parquet(path, index=False)
    log.info(f"[{market}] Saved universe to {path}")
    return path
