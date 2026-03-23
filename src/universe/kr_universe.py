"""Dynamic South Korea tech universe builder using KRX data."""

import io
import requests
import pandas as pd
from src.common.config import get_data_path
from src.common.logger import get_logger

log = get_logger("universe.kr")

# KRX corporate listing download URL (returns HTML table of all listed companies)
KRX_CORP_LIST_URL = (
    "http://kind.krx.co.kr/corpgeneral/corpList.do"
    "?method=download&searchType=13"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}

# Korean tech-related sector/industry keywords
TECH_SECTOR_KEYWORDS = [
    "반도체",       # semiconductor
    "전자",         # electronics
    "IT",
    "소프트웨어",   # software
    "통신",         # telecom
    "게임",         # gaming
    "디스플레이",   # display
    "배터리",       # battery
    "컴퓨터",       # computer
    "정보",         # information
    "인터넷",       # internet
    "클라우드",     # cloud
    "데이터",       # data
    "로봇",         # robot
    "AI",
    "전기전자",     # electrical/electronics
    "기술",         # technology
]


def _matches_kr_tech(sector: str) -> bool:
    """Return True if the sector string matches Korean tech keywords."""
    sector = (sector or "").strip()
    if not sector:
        return False
    for kw in TECH_SECTOR_KEYWORDS:
        if kw.lower() in sector.lower():
            return True
    return False


def _format_ticker(code: str) -> str:
    """Zero-pad a KRX stock code to 6 digits."""
    code = str(code).strip()
    return code.zfill(6)


def build_kr_tech_universe() -> pd.DataFrame:
    """Fetch Korean tech stocks dynamically from KRX corporate listings.

    Downloads the full KRX company listing, filters for tech-related sectors,
    and formats tickers for yfinance (.KS suffix for KOSPI).

    Falls back to the static YAML watchlist if the API is unavailable.

    Returns:
        DataFrame with columns: ticker, name, sector, industry, exchange,
        currency, yf_symbol, market
    """
    try:
        log.info("Fetching KRX corporate listing...")
        resp = requests.get(KRX_CORP_LIST_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()

        # KRX returns an HTML page with a table; pandas can parse it
        dfs = pd.read_html(io.StringIO(resp.text))
        if not dfs:
            log.warning("No tables found in KRX response.")
            return _fallback_watchlist()

        df = dfs[0]
        log.info(f"KRX listing: {len(df)} total companies, "
                 f"columns: {list(df.columns)}")

        # Expected columns (Korean): 종목코드, 회사명, 업종, 주요제품, ...
        # Map to English names based on position
        col_map = {}
        cols = list(df.columns)
        # Try to find columns by known Korean names
        for c in cols:
            c_str = str(c)
            if "종목코드" in c_str or "코드" in c_str:
                col_map["code"] = c
            elif "회사명" in c_str or "기업명" in c_str or "종목명" in c_str:
                col_map["name"] = c
            elif "업종" in c_str:
                col_map["sector"] = c
            elif "주요제품" in c_str or "주요사업" in c_str:
                col_map["product"] = c

        # Fallback: use positional mapping if name-based didn't work
        if "code" not in col_map and len(cols) >= 2:
            col_map["code"] = cols[0] if "code" not in col_map else col_map["code"]
            col_map["name"] = cols[1] if "name" not in col_map else col_map["name"]
        if "sector" not in col_map and len(cols) >= 3:
            col_map["sector"] = cols[2]

        if "code" not in col_map or "name" not in col_map:
            log.warning("Could not identify required columns in KRX data.")
            return _fallback_watchlist()

        rows = []
        for _, row in df.iterrows():
            code = _format_ticker(row[col_map["code"]])
            name = str(row.get(col_map["name"], "")).strip()
            sector = str(row.get(col_map.get("sector", ""), "")).strip()
            product = str(row.get(col_map.get("product", ""), "")).strip()

            # Check sector and product fields for tech keywords
            combined = f"{sector} {product}"
            if not _matches_kr_tech(combined):
                continue

            # KOSPI stocks use .KS suffix, KOSDAQ uses .KQ
            # Default to .KS; the market config has ticker_suffix=".KS"
            yf_symbol = f"{code}.KS"

            rows.append({
                "ticker": code,
                "name": name,
                "sector": sector,
                "industry": product if product != "nan" else sector,
                "exchange": "KRX",
                "currency": "KRW",
                "yf_symbol": yf_symbol,
                "market": "KR",
            })

    except Exception as e:
        log.warning(f"KRX API failed: {e}. Falling back to static watchlist.")
        return _fallback_watchlist()

    if not rows:
        log.warning("No tech stocks found from KRX data. "
                     "Falling back to static watchlist.")
        return _fallback_watchlist()

    result = pd.DataFrame(rows)
    result = result.drop_duplicates(subset=["ticker"], keep="first")
    log.info(f"KR tech universe: {len(result)} unique stocks (KRX, tech-filtered)")

    # Save to processed directory
    path = get_data_path("processed", "kr_tech_universe.parquet")
    result.to_parquet(path, index=False)
    log.info(f"Saved to {path}")

    return result


def _fallback_watchlist() -> pd.DataFrame:
    """Load the static YAML watchlist as fallback."""
    from src.universe.yf_universe import build_yf_universe
    log.info("Using static kr_watchlist.yaml as fallback")
    return build_yf_universe("KR", "kr_watchlist.yaml", ticker_suffix=".KS")
