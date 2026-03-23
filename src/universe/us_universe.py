"""Dynamic US tech universe builder using NASDAQ screener API."""

import re
import requests
import pandas as pd
from src.common.config import load_yaml, get_data_path
from src.common.logger import get_logger

log = get_logger("universe.us")

NASDAQ_API_URL = "https://api.nasdaq.com/api/screener/stocks"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

TECH_SECTORS = {"Technology", "Telecommunications", "Consumer Services"}

TECH_INDUSTRY_KEYWORDS = [
    "Computer", "Software", "Semiconductor", "Electronic", "Internet",
    "Data Processing", "Telecommunications", "EDP", "Radio", "Electrical",
]

# Suffixes indicating warrants, units, rights, etc.
_EXCLUDE_SUFFIXES = ("-W", "-U", "-R", "-WS", "-WT", "-UN")


def _is_valid_ticker(symbol: str) -> bool:
    """Return True if the symbol looks like a common stock ticker."""
    if not symbol:
        return False
    # Exclude symbols with dots (class shares like BRK.B, warrants, units)
    if "." in symbol:
        return False
    upper = symbol.upper()
    for suffix in _EXCLUDE_SUFFIXES:
        if upper.endswith(suffix):
            return False
    # Only keep simple alpha tickers (1-5 uppercase letters)
    if not re.match(r"^[A-Z]{1,5}$", upper):
        return False
    return True


def _matches_tech(sector: str, industry: str) -> bool:
    """Return True if the stock belongs to a tech-related sector/industry."""
    sector = (sector or "").strip()
    industry = (industry or "").strip()

    if sector in TECH_SECTORS:
        return True

    for kw in TECH_INDUSTRY_KEYWORDS:
        if kw.lower() in industry.lower():
            return True

    return False


def _fetch_exchange(exchange: str) -> list[dict]:
    """Fetch all stocks for a given exchange from the NASDAQ API."""
    params = {
        "tableType": "listed",
        "limit": "10000",
        "exchange": exchange,
    }
    log.info(f"Fetching {exchange} listings from NASDAQ API...")
    resp = requests.get(NASDAQ_API_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = data.get("data", {}).get("table", {}).get("rows", [])
    if not rows:
        # Some API versions nest differently
        rows = data.get("data", {}).get("rows", [])
    log.info(f"  {exchange}: {len(rows)} total listings returned")
    return rows


def build_us_tech_universe() -> pd.DataFrame:
    """Fetch US tech stocks dynamically from NASDAQ and NYSE listings.

    Downloads all listed stocks from both exchanges via the NASDAQ screener
    API, filters for tech-related sectors and industries, removes ETFs /
    warrants / preferred shares, and de-duplicates across exchanges.

    Falls back to the static YAML watchlist if the API is unavailable.

    Returns:
        DataFrame with columns: ticker, name, sector, industry, exchange,
        currency, yf_symbol, market
    """
    all_rows = []

    try:
        for exchange in ("NASDAQ", "NYSE"):
            rows = _fetch_exchange(exchange)
            for row in rows:
                symbol = (row.get("symbol") or "").strip()
                sector = (row.get("sector") or "").strip()
                industry = (row.get("industry") or "").strip()
                name = (row.get("name") or "").strip()

                if not _is_valid_ticker(symbol):
                    continue

                if not _matches_tech(sector, industry):
                    continue

                all_rows.append({
                    "ticker": symbol,
                    "name": name,
                    "sector": sector,
                    "industry": industry,
                    "exchange": exchange,
                    "currency": "USD",
                    "yf_symbol": symbol,
                    "market": "US",
                })

    except Exception as e:
        log.warning(f"NASDAQ API failed: {e}. Falling back to static watchlist.")
        return _fallback_watchlist()

    if not all_rows:
        log.warning("No tech stocks found from NASDAQ API. "
                     "Falling back to static watchlist.")
        return _fallback_watchlist()

    result = pd.DataFrame(all_rows)
    result = result.drop_duplicates(subset=["ticker"], keep="first")
    log.info(f"US tech universe: {len(result)} unique stocks "
             f"(NASDAQ + NYSE, tech-filtered)")

    # Save to processed directory
    path = get_data_path("processed", "us_tech_universe.parquet")
    result.to_parquet(path, index=False)
    log.info(f"Saved to {path}")

    return result


def _fallback_watchlist() -> pd.DataFrame:
    """Load the static YAML watchlist as fallback."""
    from src.universe.yf_universe import build_yf_universe
    log.info("Using static us_watchlist.yaml as fallback")
    return build_yf_universe("US", "us_watchlist.yaml", ticker_suffix="")
