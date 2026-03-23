"""Dynamic Australia tech universe builder using ASX company directory API."""

import requests
import pandas as pd
from src.common.config import get_data_path
from src.common.logger import get_logger

log = get_logger("universe.au")

# ASX company directory API (public, no auth needed)
ASX_DIRECTORY_URL = (
    "https://asx.api.markitdigital.com/asx-research/1.0/companies/directory"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-AU,en;q=0.9",
}

# ASX sector/industry filters for tech stocks
TECH_CLASSIFICATION_GROUPS = {
    "Information Technology",
}

TECH_SECTOR_KEYWORDS = [
    "Technology",
    "Telecommunication",
    "Software",
    "Hardware",
    "Semiconductor",
    "Information",
    "Digital",
    "Cyber",
    "Cloud",
    "Data",
]


def _matches_au_tech(classification_group: str, industry_sector: str) -> bool:
    """Return True if the company belongs to a tech sector."""
    group = (classification_group or "").strip()
    sector = (industry_sector or "").strip()

    # Direct classification group match
    if group in TECH_CLASSIFICATION_GROUPS:
        return True

    # Keyword match on sector and group
    combined = f"{group} {sector}".lower()
    for kw in TECH_SECTOR_KEYWORDS:
        if kw.lower() in combined:
            return True

    return False


def build_au_tech_universe() -> pd.DataFrame:
    """Fetch Australian tech stocks dynamically from ASX company directory.

    Uses the ASX MarkitDigital API to get all listed companies, filters for
    technology-related sectors, and formats tickers for yfinance (.AX suffix).

    Falls back to the static YAML watchlist if the API is unavailable.

    Returns:
        DataFrame with columns: ticker, name, sector, industry, exchange,
        currency, yf_symbol, market
    """
    all_rows = []

    try:
        page = 0
        page_size = 2000
        total_fetched = 0

        while True:
            params = {
                "page": page,
                "pageSize": page_size,
                "order": "ascending",
                "orderBy": "companyName",
            }

            log.info(f"Fetching ASX directory page {page}...")
            resp = requests.get(
                ASX_DIRECTORY_URL, params=params, headers=HEADERS, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()

            items = data.get("data", {}).get("items", [])
            if not items:
                break

            total_fetched += len(items)
            log.info(f"  Page {page}: {len(items)} companies")

            for item in items:
                symbol = (item.get("symbol") or "").strip()
                name = (item.get("displayName") or
                        item.get("companyName") or "").strip()
                classification_group = (
                    item.get("classificationGroup") or ""
                ).strip()
                industry_sector = (
                    item.get("industrySector") or ""
                ).strip()

                if not symbol:
                    continue

                if not _matches_au_tech(classification_group, industry_sector):
                    continue

                yf_symbol = f"{symbol}.AX"

                all_rows.append({
                    "ticker": symbol,
                    "name": name,
                    "sector": classification_group or industry_sector,
                    "industry": industry_sector or classification_group,
                    "exchange": "ASX",
                    "currency": "AUD",
                    "yf_symbol": yf_symbol,
                    "market": "AU",
                })

            # Check if we need more pages
            total_count = data.get("data", {}).get("count", 0)
            if total_fetched >= total_count or len(items) < page_size:
                break
            page += 1

        log.info(f"ASX directory: {total_fetched} total companies fetched")

    except Exception as e:
        log.warning(f"ASX API failed: {e}. Falling back to static watchlist.")
        return _fallback_watchlist()

    if not all_rows:
        log.warning("No tech stocks found from ASX data. "
                     "Falling back to static watchlist.")
        return _fallback_watchlist()

    result = pd.DataFrame(all_rows)
    result = result.drop_duplicates(subset=["ticker"], keep="first")
    log.info(f"AU tech universe: {len(result)} unique stocks "
             f"(ASX, tech-filtered)")

    # Save to processed directory
    path = get_data_path("processed", "au_tech_universe.parquet")
    result.to_parquet(path, index=False)
    log.info(f"Saved to {path}")

    return result


def _fallback_watchlist() -> pd.DataFrame:
    """Load the static YAML watchlist as fallback."""
    from src.universe.yf_universe import build_yf_universe
    log.info("Using static au_watchlist.yaml as fallback")
    return build_yf_universe("AU", "au_watchlist.yaml", ticker_suffix=".AX")
