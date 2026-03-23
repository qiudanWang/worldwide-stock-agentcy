"""Dynamic Taiwan tech universe builder using TWSE data."""

import io
import requests
import pandas as pd
from src.common.config import get_data_path
from src.common.logger import get_logger

log = get_logger("universe.tw")

# TWSE ISIN code listing — returns HTML table of all listed stocks with industry
TWSE_ISIN_URL = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
}

# TWSE tech-related industry names (Traditional Chinese)
TECH_INDUSTRIES = {
    "半導體業",           # Semiconductor
    "光電業",             # Optoelectronics
    "電子零組件業",       # Electronic parts/components
    "電腦及週邊設備業",   # Computer & peripherals
    "資訊服務業",         # Information services
    "通信網路業",         # Communications & networking
    "電子通路業",         # Electronic products distribution
    "其他電子業",         # Other electronics
}

# Additional keyword matching for edge cases
TECH_INDUSTRY_KEYWORDS = [
    "半導體", "光電", "電子", "電腦", "資訊", "通信",
    "網路", "軟體", "科技", "晶片", "面板",
]


def _matches_tw_tech(industry: str) -> bool:
    """Return True if the industry string matches Taiwan tech industries."""
    industry = (industry or "").strip()
    if not industry:
        return False
    # Exact match first
    if industry in TECH_INDUSTRIES:
        return True
    # Keyword fallback
    for kw in TECH_INDUSTRY_KEYWORDS:
        if kw in industry:
            return True
    return False


def build_tw_tech_universe() -> pd.DataFrame:
    """Fetch Taiwanese tech stocks dynamically from TWSE ISIN listing.

    Downloads the full TWSE stock listing with industry codes, filters
    for tech-related industries, and formats tickers for yfinance (.TW suffix).

    Falls back to the static YAML watchlist if the API is unavailable.

    Returns:
        DataFrame with columns: ticker, name, sector, industry, exchange,
        currency, yf_symbol, market
    """
    try:
        log.info("Fetching TWSE ISIN listing...")
        resp = requests.get(TWSE_ISIN_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()

        # TWSE page is encoded in big5 or MS950
        resp.encoding = "big5"
        html_text = resp.text

        # Parse the HTML table
        dfs = pd.read_html(io.StringIO(html_text))
        if not dfs:
            log.warning("No tables found in TWSE response.")
            return _fallback_watchlist()

        df = dfs[0]
        log.info(f"TWSE listing: {len(df)} rows, columns: {list(df.columns)}")

        # The TWSE ISIN table typically has columns:
        # 有價證券代號及名稱 | 國際證券辨識號碼 | 上市日 | 市場別 | 產業別 | CFICode | 備註
        # The first column contains "code\u3000name" (code + fullwidth space + name)
        # or "code name"

        rows = []
        for _, row in df.iterrows():
            # First column: combined code and name
            combined = str(row.iloc[0]).strip()
            if not combined or combined == "nan":
                continue

            # Split code from name — code is numeric, separated by space or \u3000
            parts = combined.replace("\u3000", " ").split(None, 1)
            if len(parts) < 2:
                continue

            code = parts[0].strip()
            name = parts[1].strip()

            # Only keep numeric stock codes (skip ETFs, warrants, etc.)
            if not code.isdigit():
                continue

            # Standard TWSE stock codes are 4 digits
            if len(code) < 4 or len(code) > 6:
                continue

            # Get industry column (typically column index 4 or labeled 產業別)
            industry = ""
            for col_idx, col_name in enumerate(df.columns):
                if "產業別" in str(col_name):
                    industry = str(row[col_name]).strip()
                    break
            if not industry or industry == "nan":
                # Try positional: industry is often the 5th column
                if len(df.columns) > 4:
                    industry = str(row.iloc[4]).strip()

            if not _matches_tw_tech(industry):
                continue

            yf_symbol = f"{code}.TW"

            rows.append({
                "ticker": code,
                "name": name,
                "sector": industry,
                "industry": industry,
                "exchange": "TWSE",
                "currency": "TWD",
                "yf_symbol": yf_symbol,
                "market": "TW",
            })

    except Exception as e:
        log.warning(f"TWSE API failed: {e}. Falling back to static watchlist.")
        return _fallback_watchlist()

    if not rows:
        log.warning("No tech stocks found from TWSE data. "
                     "Falling back to static watchlist.")
        return _fallback_watchlist()

    result = pd.DataFrame(rows)
    result = result.drop_duplicates(subset=["ticker"], keep="first")
    log.info(f"TW tech universe: {len(result)} unique stocks "
             f"(TWSE, tech-filtered)")

    # Save to processed directory
    path = get_data_path("processed", "tw_tech_universe.parquet")
    result.to_parquet(path, index=False)
    log.info(f"Saved to {path}")

    return result


def _fallback_watchlist() -> pd.DataFrame:
    """Load the static YAML watchlist as fallback."""
    from src.universe.yf_universe import build_yf_universe
    log.info("Using static tw_watchlist.yaml as fallback")
    return build_yf_universe("TW", "tw_watchlist.yaml", ticker_suffix=".TW")
