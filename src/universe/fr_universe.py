"""Dynamic FR tech universe builder using Euronext API with yfinance fallback."""

import requests
import pandas as pd
import yfinance as yf
from src.common.config import load_yaml, get_data_path
from src.common.logger import get_logger

log = get_logger("universe.fr")

EURONEXT_URL = "https://live.euronext.com/en/pd/data/stock"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Content-Type": "application/x-www-form-urlencoded",
}

TECH_SECTORS = {
    "Technology", "Information Technology", "Software", "Telecommunications",
    "Communication Services", "Electronic Equipment", "Semiconductors",
    "IT Services", "Consumer Electronics",
    # French sector names
    "Technologie", "Informatique", "Logiciels", "Semiconducteurs",
    "Télécommunications", "Electronique",
}

TECH_SECTOR_KEYWORDS = [
    "technolog", "software", "logiciel", "semiconductor", "semiconducteur",
    "electronic", "electronique", "internet", "computer", "informatique",
    "telecom", "télécommunication", "it service", "cloud", "digital",
    "numérique", "cyber",
]

# Extended seed tickers beyond the static watchlist
_EXTRA_SEED_TICKERS = [
    "DG", "EN", "GEN", "ALD", "WLN", "DBV", "MRN", "LIN",
    "ALMDG", "ALNOV", "ALSEI", "ALPLA", "ALDBL", "ALWEC",
    "ERF", "HO", "TCH", "ALKAL", "QDT", "NET", "BOI", "VLA",
    "ALAN", "ALTHE", "ALVIV", "ALDMS", "ALLUX", "ALTEV",
    "SW", "BVI", "MERY", "ATOS", "NEX", "CGM",
]


def _try_euronext_api() -> list[dict] | None:
    """Attempt to fetch Paris-listed equities from the Euronext API."""
    try:
        log.info("Trying Euronext API for Paris equity listings...")
        payload = {
            "mics": "XPAR",
            "display_datapoints": "dp_stock",
            "display_filters": "df_stock",
        }
        resp = requests.post(
            EURONEXT_URL, data=payload, headers=HEADERS, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()

        records = data if isinstance(data, list) else data.get("aaData", data.get("data", []))

        rows = []
        for item in records:
            # Euronext returns arrays or dicts depending on endpoint
            if isinstance(item, list):
                # Typical format: [name_html, isin, symbol, ...]
                ticker = str(item[2]).strip() if len(item) > 2 else ""
                name = str(item[0]).strip() if len(item) > 0 else ""
                sector = str(item[5]).strip() if len(item) > 5 else ""
                # Strip HTML tags from name
                import re
                name = re.sub(r"<[^>]+>", "", name).strip()
            elif isinstance(item, dict):
                ticker = (item.get("symbol") or item.get("ticker") or "").strip()
                name = (item.get("name") or "").strip()
                sector = (item.get("sector") or item.get("icbSector") or "").strip()
            else:
                continue

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
                "exchange": "XPAR",
                "currency": "EUR",
                "yf_symbol": f"{ticker}.PA",
                "market": "FR",
            })

        if rows:
            log.info(f"Euronext API returned {len(rows)} tech stocks")
            return rows
        log.warning("Euronext API returned no tech stocks")
        return None

    except Exception as e:
        log.warning(f"Euronext API failed: {e}")
        return None


def _validate_with_yfinance(tickers: list[str], suffix: str = ".PA") -> list[dict]:
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
                "exchange": "XPAR",
                "currency": "EUR",
                "yf_symbol": yf_symbol,
                "market": "FR",
            })
        except Exception:
            continue

    return rows


def _get_seed_tickers() -> list[str]:
    """Collect seed tickers from the static watchlist plus extras."""
    tickers = set(_EXTRA_SEED_TICKERS)
    try:
        cfg = load_yaml("fr_watchlist.yaml")
        for sector_tickers in cfg.values():
            for t in sector_tickers:
                tickers.add(str(t))
    except Exception as e:
        log.warning(f"Could not load fr_watchlist.yaml: {e}")
    return sorted(tickers)


def build_fr_tech_universe() -> pd.DataFrame:
    """Fetch FR tech stocks dynamically from Euronext or yfinance.

    Strategy:
    1. Try the Euronext Paris API for listed equities
    2. If that fails, validate seed tickers via yfinance sector data
    3. Fall back to static watchlist on total failure

    Returns:
        DataFrame with columns: ticker, name, sector, industry, exchange,
        currency, yf_symbol, market
    """
    # Strategy 1: Euronext API
    rows = _try_euronext_api()

    if rows:
        result = pd.DataFrame(rows)
        result = result.drop_duplicates(subset=["ticker"], keep="first")
        log.info(f"FR tech universe (Euronext API): {len(result)} stocks")
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
            log.info(f"FR tech universe (yfinance validated): {len(result)} stocks")
            _save(result)
            return result
    except Exception as e:
        log.warning(f"yfinance validation failed: {e}")

    # Strategy 3: static fallback
    log.warning("All dynamic methods failed. Falling back to static watchlist.")
    return _fallback_watchlist()


def _save(df: pd.DataFrame) -> None:
    """Save universe to processed directory."""
    path = get_data_path("processed", "fr_tech_universe.parquet")
    df.to_parquet(path, index=False)
    log.info(f"Saved to {path}")


def _fallback_watchlist() -> pd.DataFrame:
    """Load the static YAML watchlist as fallback."""
    from src.universe.yf_universe import build_yf_universe
    log.info("Using static fr_watchlist.yaml as fallback")
    return build_yf_universe("FR", "fr_watchlist.yaml", ticker_suffix=".PA")
