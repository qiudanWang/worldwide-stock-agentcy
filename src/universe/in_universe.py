"""Dynamic IN tech universe builder using NSE India APIs."""

import requests
import pandas as pd
from src.common.config import get_data_path
from src.common.logger import get_logger

log = get_logger("universe.in")

_NSE_BASE = "https://www.nseindia.com"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}

# NSE sector index names for tech stocks
_TECH_INDICES = [
    "NIFTY IT",
    "NIFTY NEXT 50",
]

# Comprehensive list of known Indian IT/tech companies (NSE symbols)
_KNOWN_IN_TECH = {
    # IT Services (large cap)
    "INFY": "Infosys",
    "TCS": "Tata Consultancy Services",
    "WIPRO": "Wipro",
    "HCLTECH": "HCL Technologies",
    "TECHM": "Tech Mahindra",
    "LTIM": "LTIMindtree",
    # IT Services (mid cap)
    "MPHASIS": "Mphasis",
    "COFORGE": "Coforge",
    "PERSISTENT": "Persistent Systems",
    "BIRLASOFT": "Birlasoft",
    "CYIENT": "Cyient",
    "ZENSAR": "Zensar Technologies",
    "MASTEK": "Mastek",
    "ECLERX": "eClerx Services",
    "NIIT": "NIIT",
    # Engineering & R&D
    "KPITTECH": "KPIT Technologies",
    "LTTS": "L&T Technology Services",
    "TATAELXSI": "Tata Elxsi",
    "SONATA": "Sonata Software",
    # Product & Platform
    "NAUKRI": "Info Edge (Naukri)",
    "ROUTE": "Route Mobile",
    "HAPPSTMNDS": "Happiest Minds",
    "INTELLECT": "Intellect Design Arena",
    "NEWGEN": "Newgen Software",
    "TANLA": "Tanla Platforms",
    "MAPMYINDIA": "CE Info Systems",
    "RATEGAIN": "RateGain Travel",
    # Internet & Digital
    "ZOMATO": "Zomato",
    "PAYTM": "One97 Communications",
    "POLICYBZR": "PB Fintech",
    "DELHIVERY": "Delhivery",
    # Additional tech stocks beyond the static watchlist
    "OFSS": "Oracle Financial Services",
    "MSSL": "Motherson Sumi Wiring India",
    "LATENTVIEW": "Latent View Analytics",
    "DATAPATTNS": "Data Patterns India",
    "TATACOMM": "Tata Communications",
    "BHARTIARTL": "Bharti Airtel",
    "IDEA": "Vodafone Idea",
    "NAZARA": "Nazara Technologies",
    "AFFLE": "Affle India",
    "JUSTDIAL": "Just Dial",
    "BSOFT": "Birlasoft",
    "ABORTIVETC": "Absorbed in tech",
    "NETWEB": "Netweb Technologies",
    "QUICKHEAL": "Quick Heal Technologies",
    "SASKEN": "Sasken Technologies",
    "CMSINFO": "CMS Info Systems",
    "IEXNET": "Indian Energy Exchange",
    "RAILTEL": "RailTel Corporation",
}

# Remove placeholder entries that aren't real stocks
_KNOWN_IN_TECH.pop("ABORTIVETC", None)


def _get_nse_session() -> requests.Session:
    """Create a session with NSE cookies (required to bypass anti-bot)."""
    session = requests.Session()
    session.headers.update(_HEADERS)
    # First request to get cookies
    try:
        session.get(_NSE_BASE, timeout=10)
    except Exception:
        pass  # cookies may still be set
    return session


def _fetch_index_constituents(session: requests.Session, index_name: str) -> list[dict]:
    """Fetch constituents of an NSE index."""
    url = f"{_NSE_BASE}/api/equity-stockIndices"
    params = {"index": index_name}

    log.info(f"Fetching NSE index: {index_name}")
    resp = session.get(url, params=params, timeout=15)
    resp.raise_for_status()

    data = resp.json()
    stocks = data.get("data", [])
    log.info(f"  {index_name}: {len(stocks)} constituents")
    return stocks


def build_in_tech_universe() -> pd.DataFrame:
    """Fetch Indian tech stocks dynamically from NSE sector indices.

    Fetches NIFTY IT and NIFTY NEXT 50 index constituents from NSE,
    filters for tech, and combines with a comprehensive known-stocks list.

    Falls back to the static YAML watchlist if the API is unavailable.

    Returns:
        DataFrame with columns: ticker, name, sector, industry, exchange,
        currency, yf_symbol, market
    """
    rows = []
    seen = set()

    try:
        session = _get_nse_session()

        for index_name in _TECH_INDICES:
            try:
                stocks = _fetch_index_constituents(session, index_name)

                for stock in stocks:
                    symbol = (stock.get("symbol") or "").strip()
                    name = (stock.get("meta", {}).get("companyName")
                            or stock.get("companyName")
                            or symbol)
                    industry = (stock.get("meta", {}).get("industry") or "")

                    if not symbol or symbol == index_name:
                        continue

                    # For NIFTY NEXT 50, only include known tech stocks
                    if index_name == "NIFTY NEXT 50":
                        if symbol not in _KNOWN_IN_TECH:
                            # Check if industry looks like tech
                            ind_lower = industry.lower()
                            is_tech = any(kw in ind_lower for kw in [
                                "it", "software", "technology", "computer",
                                "telecom", "internet", "digital",
                            ])
                            if not is_tech:
                                continue

                    if symbol in seen:
                        continue
                    seen.add(symbol)

                    rows.append({
                        "ticker": symbol,
                        "name": name,
                        "sector": "Technology",
                        "industry": industry,
                        "exchange": "NSE",
                        "currency": "INR",
                        "yf_symbol": symbol,
                        "market": "IN",
                    })

            except Exception as e:
                log.warning(f"  Failed to fetch {index_name}: {e}")

    except Exception as e:
        log.warning(f"NSE session failed: {e}. Using known stocks baseline.")

    # Always ensure all known IN tech stocks are included
    for symbol, name in _KNOWN_IN_TECH.items():
        if symbol not in seen:
            rows.append({
                "ticker": symbol,
                "name": name,
                "sector": "Technology",
                "industry": "",
                "exchange": "NSE",
                "currency": "INR",
                "yf_symbol": symbol,
                "market": "IN",
            })
            seen.add(symbol)

    if not rows:
        log.warning("No IN tech stocks found. Falling back to static watchlist.")
        return _fallback_watchlist()

    result = pd.DataFrame(rows)
    result = result.drop_duplicates(subset=["ticker"], keep="first")
    log.info(f"IN tech universe: {len(result)} unique stocks")

    path = get_data_path("processed", "in_tech_universe.parquet")
    result.to_parquet(path, index=False)
    log.info(f"Saved to {path}")

    return result


def _fallback_watchlist() -> pd.DataFrame:
    """Load the static YAML watchlist as fallback."""
    from src.universe.yf_universe import build_yf_universe
    log.info("Using static in_watchlist.yaml as fallback")
    return build_yf_universe("IN", "in_watchlist.yaml", ticker_suffix="")
