"""Dynamic SA tech universe builder using Tadawul data with yfinance fallback."""

import requests
import pandas as pd
import yfinance as yf
from src.common.config import load_yaml, get_data_path
from src.common.logger import get_logger

log = get_logger("universe.sa")

TADAWUL_URL = (
    "https://www.saudiexchange.sa/wps/portal/saudiexchange/"
    "hidden/company-profile-main/"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
}

TECH_SECTORS = {
    "Technology", "Information Technology", "Software", "Telecommunications",
    "Communication Services", "Electronic Equipment", "IT Services",
    "Telecommunication Services",
}

TECH_SECTOR_KEYWORDS = [
    "technolog", "software", "semiconductor", "electronic", "internet",
    "computer", "telecom", "it service", "cloud", "digital", "cyber",
    "تقنية", "اتصالات",  # Arabic: technology, telecom
]

# Comprehensive seed list of Saudi tech/telecom tickers (numeric codes)
# Saudi market has limited tech; telecom is the main tech-adjacent sector
_EXTRA_SEED_TICKERS = [
    "7010",  # STC (Saudi Telecom Company)
    "7020",  # Etihad Etisalat (Mobily)
    "7030",  # Zain KSA
    "7040",  # Dawiyat (Integrated Telecom)
    "9512",  # Elm Company
    "4003",  # Extra (United Electronics)
    "7050",  # Etihad Atheeb (GO Telecom)
    "6004",  # CATRION (catering tech)
    "2381",  # Rasan Information Technology
    "7202",  # Maharah Human Resources (tech services)
    "6015",  # Thiqah (Digital Business Services)
    "7200",  # Batic Investments and Logistics
    "9528",  # Marn (formerly National Information Technology)
    "9531",  # Aljazira Takaful (digital insurance)
    "9527",  # Arab Sea Information Systems
    "1183",  # Salam (telecom infrastructure)
    "7201",  # Arabian Internet & Communications Services (solutions)
]


def _try_tadawul_api() -> list[dict] | None:
    """Attempt to fetch listings from the Tadawul (Saudi Exchange) API.

    Note: Tadawul's public API is limited and often requires session tokens.
    This is a best-effort attempt.
    """
    try:
        log.info("Trying Tadawul API for listed companies...")
        resp = requests.get(TADAWUL_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()

        # Tadawul may return HTML or JSON depending on endpoint
        content_type = resp.headers.get("Content-Type", "")
        if "json" not in content_type:
            log.warning("Tadawul returned non-JSON response, skipping API approach")
            return None

        data = resp.json()
        records = data if isinstance(data, list) else data.get("data", data.get("results", []))

        rows = []
        for item in records:
            sector = (item.get("sector") or item.get("sectorName") or "").strip()
            ticker = (item.get("symbol") or item.get("code") or "").strip()
            name = (item.get("companyName") or item.get("name") or "").strip()

            if not ticker:
                continue

            combined = sector.lower()
            is_tech = any(kw in combined for kw in TECH_SECTOR_KEYWORDS)
            if not is_tech:
                continue

            rows.append({
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "industry": "",
                "exchange": "Tadawul",
                "currency": "SAR",
                "yf_symbol": f"{ticker}.SR",
                "market": "SA",
            })

        if rows:
            log.info(f"Tadawul API returned {len(rows)} tech stocks")
            return rows
        log.warning("Tadawul API returned no tech stocks")
        return None

    except Exception as e:
        log.warning(f"Tadawul API failed: {e}")
        return None


def _validate_with_yfinance(tickers: list[str], suffix: str = ".SR") -> list[dict]:
    """Validate tickers via yfinance, keeping only tech-related ones."""
    rows = []
    for ticker in tickers:
        yf_symbol = f"{ticker}{suffix}"
        try:
            info = yf.Ticker(yf_symbol).info
            sector = (info.get("sector") or "").strip()
            industry = (info.get("industry") or "").strip()
            name = (info.get("shortName") or info.get("longName") or ticker).strip()

            # For Saudi, be more permissive — keep telecom and any tech
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
                "exchange": "Tadawul",
                "currency": "SAR",
                "yf_symbol": yf_symbol,
                "market": "SA",
            })
        except Exception:
            continue

    return rows


def _get_seed_tickers() -> list[str]:
    """Collect seed tickers from the static watchlist plus extras."""
    tickers = set(_EXTRA_SEED_TICKERS)
    try:
        cfg = load_yaml("sa_watchlist.yaml")
        for sector_tickers in cfg.values():
            for t in sector_tickers:
                tickers.add(str(t))
    except Exception as e:
        log.warning(f"Could not load sa_watchlist.yaml: {e}")
    return sorted(tickers)


def build_sa_tech_universe() -> pd.DataFrame:
    """Fetch SA tech stocks dynamically from Tadawul or yfinance.

    Strategy:
    1. Try the Tadawul API (best-effort, often requires auth)
    2. Validate comprehensive seed ticker list via yfinance sector data
    3. Fall back to static watchlist on total failure

    Returns:
        DataFrame with columns: ticker, name, sector, industry, exchange,
        currency, yf_symbol, market
    """
    # Strategy 1: Tadawul API
    rows = _try_tadawul_api()

    if rows:
        result = pd.DataFrame(rows)
        result = result.drop_duplicates(subset=["ticker"], keep="first")
        log.info(f"SA tech universe (Tadawul API): {len(result)} stocks")
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
            log.info(f"SA tech universe (yfinance validated): {len(result)} stocks")
            _save(result)
            return result
    except Exception as e:
        log.warning(f"yfinance validation failed: {e}")

    # Strategy 3: static fallback
    log.warning("All dynamic methods failed. Falling back to static watchlist.")
    return _fallback_watchlist()


def _save(df: pd.DataFrame) -> None:
    """Save universe to processed directory."""
    path = get_data_path("processed", "sa_tech_universe.parquet")
    df.to_parquet(path, index=False)
    log.info(f"Saved to {path}")


def _fallback_watchlist() -> pd.DataFrame:
    """Load the static YAML watchlist as fallback."""
    from src.universe.yf_universe import build_yf_universe
    log.info("Using static sa_watchlist.yaml as fallback")
    return build_yf_universe("SA", "sa_watchlist.yaml", ticker_suffix=".SR")
