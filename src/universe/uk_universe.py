"""Dynamic UK tech universe builder using LSE API with yfinance fallback."""

import requests
import pandas as pd
import yfinance as yf
from src.common.config import load_yaml, get_data_path
from src.common.logger import get_logger

log = get_logger("universe.uk")

LSE_API_URL = "https://api.londonstockexchange.com/api/gw/lse/instruments/alldata/all"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-GB,en;q=0.9",
}

TECH_SECTORS = {
    "Technology", "Information Technology", "Software", "Telecommunications",
    "Communication Services", "Electronic Equipment", "Semiconductors",
    "IT Services", "Consumer Electronics",
}

TECH_SECTOR_KEYWORDS = [
    "technolog", "software", "semiconductor", "electronic", "internet",
    "computer", "telecom", "data processing", "it service", "cyber",
    "cloud", "digital",
]

# Extended seed tickers beyond the static watchlist for discovery
_EXTRA_SEED_TICKERS = [
    "AVST", "SSPG", "PHNX", "FDM", "ALFA", "SGE", "RDW", "KWS",
    "APTD", "TCS", "ZOO", "TSTL", "DIGI", "BVXP", "CML", "CRVX",
    "IGP", "IQE", "MAXC", "NET", "SUPR", "TMT", "IDOX", "LOK",
    "GBGP", "LTG", "NCC", "DPH", "CSN", "BOO",
]


def _try_lse_api() -> list[dict] | None:
    """Attempt to fetch tech listings from the LSE API."""
    try:
        log.info("Trying LSE API for instrument listings...")
        resp = requests.get(LSE_API_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        rows = []
        instruments = data if isinstance(data, list) else data.get("data", [])
        for item in instruments:
            sector = (item.get("sector") or item.get("icbSector") or "").strip()
            ticker = (item.get("tidm") or item.get("ticker") or "").strip()
            name = (item.get("issuername") or item.get("name") or "").strip()

            if not ticker:
                continue

            sector_lower = sector.lower()
            is_tech = any(kw in sector_lower for kw in TECH_SECTOR_KEYWORDS)
            if not is_tech:
                continue

            rows.append({
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "industry": "",
                "exchange": "LSE",
                "currency": "GBP",
                "yf_symbol": f"{ticker}.L",
                "market": "UK",
            })

        if rows:
            log.info(f"LSE API returned {len(rows)} tech stocks")
            return rows
        log.warning("LSE API returned no tech stocks")
        return None

    except Exception as e:
        log.warning(f"LSE API failed: {e}")
        return None


def _validate_with_yfinance(tickers: list[str], suffix: str = ".L") -> list[dict]:
    """Validate tickers via yfinance, keeping only tech-related ones."""
    rows = []
    for ticker in tickers:
        yf_symbol = f"{ticker}{suffix}"
        try:
            info = yf.Ticker(yf_symbol).info
            sector = (info.get("sector") or "").strip()
            industry = (info.get("industry") or "").strip()
            name = (info.get("shortName") or info.get("longName") or ticker).strip()

            combined = f"{sector} {industry}".lower()
            is_tech = (
                sector in TECH_SECTORS
                or any(kw in combined for kw in TECH_SECTOR_KEYWORDS)
            )
            if not is_tech:
                continue

            rows.append({
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "industry": industry,
                "exchange": "LSE",
                "currency": "GBP",
                "yf_symbol": yf_symbol,
                "market": "UK",
            })
        except Exception:
            continue

    return rows


def _get_seed_tickers() -> list[str]:
    """Collect seed tickers from the static watchlist plus extras."""
    tickers = set(_EXTRA_SEED_TICKERS)
    try:
        cfg = load_yaml("uk_watchlist.yaml")
        for sector_tickers in cfg.values():
            for t in sector_tickers:
                tickers.add(str(t))
    except Exception as e:
        log.warning(f"Could not load uk_watchlist.yaml: {e}")
    return sorted(tickers)


def build_uk_tech_universe() -> pd.DataFrame:
    """Fetch UK tech stocks dynamically from LSE API or yfinance validation.

    Strategy:
    1. Try the LSE API for full instrument listings
    2. If that fails, validate seed tickers via yfinance sector data
    3. Fall back to static watchlist on total failure

    Returns:
        DataFrame with columns: ticker, name, sector, industry, exchange,
        currency, yf_symbol, market
    """
    # Strategy 1: LSE API
    rows = _try_lse_api()

    if rows:
        result = pd.DataFrame(rows)
        result = result.drop_duplicates(subset=["ticker"], keep="first")
        log.info(f"UK tech universe (LSE API): {len(result)} stocks")
        _save(result)
        return result

    # Strategy 2: yfinance validation of seed tickers
    log.info("Falling back to yfinance validation of seed tickers...")
    try:
        seed = _get_seed_tickers()
        log.info(f"Validating {len(seed)} seed tickers via yfinance...")
        rows = _validate_with_yfinance(seed)
        if rows:
            result = pd.DataFrame(rows)
            result = result.drop_duplicates(subset=["ticker"], keep="first")
            log.info(f"UK tech universe (yfinance validated): {len(result)} stocks")
            _save(result)
            return result
    except Exception as e:
        log.warning(f"yfinance validation failed: {e}")

    # Strategy 3: static fallback
    log.warning("All dynamic methods failed. Falling back to static watchlist.")
    return _fallback_watchlist()


def _save(df: pd.DataFrame) -> None:
    """Save universe to processed directory."""
    path = get_data_path("processed", "uk_tech_universe.parquet")
    df.to_parquet(path, index=False)
    log.info(f"Saved to {path}")


def _fallback_watchlist() -> pd.DataFrame:
    """Load the static YAML watchlist as fallback."""
    from src.universe.yf_universe import build_yf_universe
    log.info("Using static uk_watchlist.yaml as fallback")
    return build_yf_universe("UK", "uk_watchlist.yaml", ticker_suffix=".L")
