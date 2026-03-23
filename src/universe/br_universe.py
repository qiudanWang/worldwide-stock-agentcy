"""Dynamic BR tech universe builder using B3 API with yfinance fallback."""

import requests
import pandas as pd
import yfinance as yf
from src.common.config import load_yaml, get_data_path
from src.common.logger import get_logger

log = get_logger("universe.br")

B3_API_URL = (
    "https://sistemaswebb3-listados.b3.com.br/"
    "listedCompaniesProxy/CompanyCall/GetInitialCompanies/"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Content-Type": "application/json",
}

TECH_SECTORS = {
    "Technology", "Information Technology", "Software", "Telecommunications",
    "Communication Services", "Electronic Equipment", "Semiconductors",
    "IT Services", "Consumer Electronics", "Financial Technology",
}

TECH_SECTOR_KEYWORDS = [
    "technolog", "tecnologia", "software", "semiconductor",
    "electronic", "eletrônic", "internet", "computer", "computador",
    "telecom", "telecomunicaç", "it service", "informação",
    "fintech", "digital", "cloud", "cyber", "dados",
]

# B3 sector classifications that indicate tech
_B3_TECH_SEGMENTS = [
    "Tecnologia da Informação", "Computadores e Equipamentos",
    "Programas e Serviços", "Telecomunicações",
]

# Extended seed tickers beyond the static watchlist
_EXTRA_SEED_TICKERS = [
    "LINX3", "SQIA3", "MLAS3", "POSI3", "TIMS3", "VIVT3",
    "OIBR3", "BRIT3", "NGRD3", "WEST3", "CIEL3", "IFCM3",
    "DOTZ3", "DESK3", "LVTC3", "MOSI3", "G2DI33", "MODL3",
    "PDTC3", "ENJU3", "MBLY3", "NINJ3", "CLSA3",
]


def _try_b3_api() -> list[dict] | None:
    """Attempt to fetch company listings from the B3 API."""
    try:
        log.info("Trying B3 API for listed companies...")
        payload = {
            "language": "en-us",
            "pageNumber": 1,
            "pageSize": 1000,
        }
        resp = requests.post(
            B3_API_URL, json=payload, headers=HEADERS, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            results = data if isinstance(data, list) else []

        rows = []
        for company in results:
            segment = (
                company.get("segment") or company.get("segmento") or ""
            ).strip()
            sector = (
                company.get("sector") or company.get("setor") or ""
            ).strip()
            trading_name = (
                company.get("tradingName") or company.get("companyName") or ""
            ).strip()
            code = (
                company.get("codeCVM") or company.get("issuingCompany") or ""
            ).strip()

            # B3 returns company-level data; we need ticker codes
            # The API may return issuingCompany as the base ticker
            issuing = (company.get("issuingCompany") or "").strip()
            if not issuing:
                continue

            combined = f"{sector} {segment}".lower()
            is_tech = (
                any(seg.lower() in combined for seg in _B3_TECH_SEGMENTS)
                or any(kw in combined for kw in TECH_SECTOR_KEYWORDS)
            )
            if not is_tech:
                continue

            # Common share suffix for B3
            ticker = f"{issuing}3"
            rows.append({
                "ticker": ticker,
                "name": trading_name,
                "sector": sector or segment,
                "industry": segment,
                "exchange": "B3",
                "currency": "BRL",
                "yf_symbol": f"{ticker}.SA",
                "market": "BR",
            })

        if rows:
            log.info(f"B3 API returned {len(rows)} tech stocks")
            return rows
        log.warning("B3 API returned no tech stocks")
        return None

    except Exception as e:
        log.warning(f"B3 API failed: {e}")
        return None


def _validate_with_yfinance(tickers: list[str], suffix: str = ".SA") -> list[dict]:
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
                "exchange": "B3",
                "currency": "BRL",
                "yf_symbol": yf_symbol,
                "market": "BR",
            })
        except Exception:
            continue

    return rows


def _get_seed_tickers() -> list[str]:
    """Collect seed tickers from the static watchlist plus extras."""
    tickers = set(_EXTRA_SEED_TICKERS)
    try:
        cfg = load_yaml("br_watchlist.yaml")
        for sector_tickers in cfg.values():
            for t in sector_tickers:
                tickers.add(str(t))
    except Exception as e:
        log.warning(f"Could not load br_watchlist.yaml: {e}")
    return sorted(tickers)


def build_br_tech_universe() -> pd.DataFrame:
    """Fetch BR tech stocks dynamically from B3 API or yfinance.

    Strategy:
    1. Try the B3 listed companies API
    2. If that fails, validate seed tickers via yfinance sector data
    3. Fall back to static watchlist on total failure

    Returns:
        DataFrame with columns: ticker, name, sector, industry, exchange,
        currency, yf_symbol, market
    """
    # Strategy 1: B3 API
    rows = _try_b3_api()

    if rows:
        result = pd.DataFrame(rows)
        result = result.drop_duplicates(subset=["ticker"], keep="first")
        log.info(f"BR tech universe (B3 API): {len(result)} stocks")
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
            log.info(f"BR tech universe (yfinance validated): {len(result)} stocks")
            _save(result)
            return result
    except Exception as e:
        log.warning(f"yfinance validation failed: {e}")

    # Strategy 3: static fallback
    log.warning("All dynamic methods failed. Falling back to static watchlist.")
    return _fallback_watchlist()


def _save(df: pd.DataFrame) -> None:
    """Save universe to processed directory."""
    path = get_data_path("processed", "br_tech_universe.parquet")
    df.to_parquet(path, index=False)
    log.info(f"Saved to {path}")


def _fallback_watchlist() -> pd.DataFrame:
    """Load the static YAML watchlist as fallback."""
    from src.universe.yf_universe import build_yf_universe
    log.info("Using static br_watchlist.yaml as fallback")
    return build_yf_universe("BR", "br_watchlist.yaml", ticker_suffix=".SA")
