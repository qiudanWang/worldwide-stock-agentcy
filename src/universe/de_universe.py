"""Dynamic DE tech universe builder using Börse Frankfurt API with yfinance fallback."""

import requests
import pandas as pd
import yfinance as yf
from src.common.config import load_yaml, get_data_path
from src.common.logger import get_logger

log = get_logger("universe.de")

BOERSE_SEARCH_URL = "https://api.boerse-frankfurt.de/v1/search/equity_search"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
}

TECH_SECTORS = {
    "Technology", "Information Technology", "Software", "Telecommunications",
    "Communication Services", "Electronic Equipment", "Semiconductors",
    "IT Services", "Consumer Electronics",
}

TECH_SECTOR_KEYWORDS = [
    "technolog", "software", "semiconductor", "electronic", "internet",
    "computer", "telecom", "data processing", "it service", "halbleiter",
    "cloud", "digital", "cyber",
]

# Extended seed tickers beyond the static watchlist
_EXTRA_SEED_TICKERS = [
    "NB2", "DLG", "TMV", "DBAN", "O2D", "SRT3", "RIB", "YSN",
    "SMHN", "TC1", "PBB", "DTE", "QSC", "ADL", "COP", "EUZ",
    "NDX1", "GXI", "AM3D", "SANT", "PSM", "WDI", "SBS", "CYR",
    "DMG", "DLX", "G1A", "MOR", "VBK", "EVT", "MDO", "NEM",
]


def _try_boerse_api() -> list[dict] | None:
    """Attempt to fetch tech listings from the Börse Frankfurt API."""
    try:
        log.info("Trying Börse Frankfurt API for equity listings...")
        params = {
            "searchTerms": "",
            "market": "XETRA",
            "type": "EQUITY",
            "pageSize": 500,
            "pageNumber": 0,
        }
        resp = requests.get(
            BOERSE_SEARCH_URL, params=params, headers=HEADERS, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()

        records = data if isinstance(data, list) else data.get("data", data.get("results", []))
        if isinstance(records, dict):
            records = records.get("results", records.get("hits", []))

        rows = []
        for item in records:
            sector = (
                item.get("sector") or item.get("icbSector")
                or item.get("industry") or ""
            ).strip()
            ticker = (
                item.get("wkn") or item.get("ticker")
                or item.get("symbol") or ""
            ).strip()
            isin = (item.get("isin") or "").strip()
            name = (item.get("name") or item.get("shortName") or "").strip()

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
                "exchange": "XETRA",
                "currency": "EUR",
                "yf_symbol": f"{ticker}.DE",
                "market": "DE",
            })

        if rows:
            log.info(f"Börse Frankfurt API returned {len(rows)} tech stocks")
            return rows
        log.warning("Börse Frankfurt API returned no tech stocks")
        return None

    except Exception as e:
        log.warning(f"Börse Frankfurt API failed: {e}")
        return None


def _validate_with_yfinance(tickers: list[str], suffix: str = ".DE") -> list[dict]:
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
                "exchange": "XETRA",
                "currency": "EUR",
                "yf_symbol": yf_symbol,
                "market": "DE",
            })
        except Exception:
            continue

    return rows


def _get_seed_tickers() -> list[str]:
    """Collect seed tickers from the static watchlist plus extras."""
    tickers = set(_EXTRA_SEED_TICKERS)
    try:
        cfg = load_yaml("de_watchlist.yaml")
        for sector_tickers in cfg.values():
            for t in sector_tickers:
                tickers.add(str(t))
    except Exception as e:
        log.warning(f"Could not load de_watchlist.yaml: {e}")
    return sorted(tickers)


def build_de_tech_universe() -> pd.DataFrame:
    """Fetch DE tech stocks dynamically from Börse Frankfurt or yfinance.

    Strategy:
    1. Try the Börse Frankfurt equity search API
    2. If that fails, validate seed tickers via yfinance sector data
    3. Fall back to static watchlist on total failure

    Returns:
        DataFrame with columns: ticker, name, sector, industry, exchange,
        currency, yf_symbol, market
    """
    # Strategy 1: Börse Frankfurt API
    rows = _try_boerse_api()

    if rows:
        result = pd.DataFrame(rows)
        result = result.drop_duplicates(subset=["ticker"], keep="first")
        log.info(f"DE tech universe (Börse Frankfurt API): {len(result)} stocks")
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
            log.info(f"DE tech universe (yfinance validated): {len(result)} stocks")
            _save(result)
            return result
    except Exception as e:
        log.warning(f"yfinance validation failed: {e}")

    # Strategy 3: static fallback
    log.warning("All dynamic methods failed. Falling back to static watchlist.")
    return _fallback_watchlist()


def _save(df: pd.DataFrame) -> None:
    """Save universe to processed directory."""
    path = get_data_path("processed", "de_tech_universe.parquet")
    df.to_parquet(path, index=False)
    log.info(f"Saved to {path}")


def _fallback_watchlist() -> pd.DataFrame:
    """Load the static YAML watchlist as fallback."""
    from src.universe.yf_universe import build_yf_universe
    log.info("Using static de_watchlist.yaml as fallback")
    return build_yf_universe("DE", "de_watchlist.yaml", ticker_suffix=".DE")
