"""Dynamic JP tech universe builder using JPX data and yfinance verification."""

import requests
import pandas as pd
from src.common.config import get_data_path
from src.common.logger import get_logger

log = get_logger("universe.jp")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# JPX sector codes that map to tech
_TECH_SECTORS_JP = {
    "Information & Communication",
    "Electric Appliances",
    "Precision Instruments",
    # Japanese sector names (if CSV has them)
    "情報・通信業",
    "電気機器",
    "精密機器",
}

# English keywords for tech filtering in sector/industry fields
_TECH_KEYWORDS = [
    "technology", "electronic", "semiconductor", "software", "information",
    "communication", "gaming", "computer", "internet", "electric appliance",
    "precision instrument", "telecom", "automation", "robot",
]

# Comprehensive list of known JP tech stocks (ticker -> name)
# Used as baseline and also as fallback enrichment source
_KNOWN_JP_TECH = {
    # Semiconductor equipment
    "8035": "Tokyo Electron",
    "6857": "Advantest",
    "6920": "Lasertec",
    "6146": "Disco Corporation",
    "7735": "Screen Holdings",
    "6525": "Kokusai Electric",
    "6526": "Socionext",
    "3436": "SUMCO",
    "4063": "Shin-Etsu Chemical",
    "4186": "Tokyo Ohka Kogyo",
    "7741": "HOYA Corporation",
    # Electronic components
    "6981": "Murata Manufacturing",
    "6762": "TDK Corporation",
    "6971": "Kyocera",
    "6594": "Nidec",
    "6963": "Rohm",
    "4062": "Ibiden",
    "6723": "Renesas Electronics",
    "6976": "Taiyo Yuden",
    "6988": "Nitto Denko",
    "6807": "Japan Aviation Electronics",
    "6479": "MinebeaMitsumi",
    "5803": "Fujikura",
    "5802": "Sumitomo Electric Industries",
    # Software / IT
    "6701": "NEC",
    "6702": "Fujitsu",
    "9613": "NTT Data Group",
    "4684": "Obic",
    "4776": "Cybozu",
    "4704": "Trend Micro",
    "9719": "SCSK",
    "4475": "Hennge",
    "4307": "Nomura Research Institute",
    "2413": "M3 Inc",
    "6098": "Recruit Holdings",
    # Internet / Platform
    "9984": "SoftBank Group",
    "4689": "LY Corp",
    "4385": "Mercari",
    "4751": "CyberAgent",
    "9449": "GMO Internet Group",
    "4755": "Rakuten Group",
    "2432": "DeNA",
    # Hardware
    "6758": "Sony Group",
    "6501": "Hitachi",
    "7751": "Canon",
    "7752": "Ricoh",
    "6448": "Brother Industries",
    "6861": "Keyence",
    "6645": "Omron",
    "6752": "Panasonic Holdings",
    "6753": "Sharp",
    "4902": "Konica Minolta",
    "6504": "Fuji Electric",
    "6503": "Mitsubishi Electric",
    # Gaming
    "7974": "Nintendo",
    "9697": "Capcom",
    "7832": "Bandai Namco Holdings",
    "9684": "Square Enix Holdings",
    "3635": "Koei Tecmo Holdings",
    "3659": "Nexon",
    "9766": "Konami Group",
    # Telecom equipment
    "6754": "Anritsu",
    "6841": "Yokogawa Electric",
    # Robotics / Automation
    "6954": "Fanuc",
    "6273": "SMC Corporation",
    "6506": "Yaskawa Electric",
    "6324": "Harmonic Drive Systems",
    "6383": "Daifuku",
    # EV / Battery / Lithography
    "6674": "GS Yuasa",
    "7731": "Nikon",
    # Telecom carriers
    "9432": "NTT",
    "9433": "KDDI",
    "9434": "SoftBank Corp",
    # Medical tech
    "6869": "Sysmex",
    "4543": "Terumo",
    "7733": "Olympus",
    "4901": "Fujifilm Holdings",
    # Additional tech stocks not in the static watchlist
    "6902": "Denso",
    "6967": "Shinko Electric Industries",
    "6986": "Dual Scope",
    "6965": "Hamamatsu Photonics",
    "6361": "Ebara Corporation",
    "7203": "Toyota (connected/autonomous tech)",
    "6902": "Denso",
    "6871": "Micronics Japan",
    "7729": "Tokyo Seimitsu",
    "3132": "Macnica Holdings",
    "6855": "Japan Electronic Materials",
    "6315": "TOWA Corporation",
}

# URL for JPX listed companies data (English)
_JPX_DATA_URL = "https://www.jpx.co.jp/english/listing/stocks/dlsearch/data/data_e.csv"


def _fetch_jpx_listings() -> pd.DataFrame:
    """Try to fetch the full JPX listed companies CSV."""
    log.info("Fetching JPX listed companies CSV...")
    resp = requests.get(_JPX_DATA_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    # JPX CSV is Shift-JIS or UTF-8 encoded
    from io import StringIO
    for encoding in ("utf-8", "shift_jis", "cp932"):
        try:
            df = pd.read_csv(StringIO(resp.content.decode(encoding)))
            if not df.empty:
                log.info(f"  JPX CSV decoded with {encoding}: {len(df)} rows")
                return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    return pd.DataFrame()


def _matches_tech_sector(sector_str: str) -> bool:
    """Check if a sector string matches tech criteria."""
    if not sector_str:
        return False
    sector_str = sector_str.strip()
    if sector_str in _TECH_SECTORS_JP:
        return True
    sector_lower = sector_str.lower()
    for kw in _TECH_KEYWORDS:
        if kw in sector_lower:
            return True
    return False


def build_jp_tech_universe() -> pd.DataFrame:
    """Fetch JP tech stocks dynamically from JPX data.

    Attempts to download the JPX listed companies CSV and filter for
    tech sectors. Enriches with the comprehensive known-stocks baseline.

    Falls back to the static YAML watchlist if the API is unavailable.

    Returns:
        DataFrame with columns: ticker, name, sector, industry, exchange,
        currency, yf_symbol, market
    """
    rows = []
    seen = set()

    try:
        jpx_df = _fetch_jpx_listings()

        if not jpx_df.empty:
            # JPX CSV columns vary; common ones:
            # "Local Code" or "Code", "Company Name", "33 Sector(name)" or similar
            code_col = None
            name_col = None
            sector_col = None

            for col in jpx_df.columns:
                col_lower = col.lower().strip()
                if "code" in col_lower and code_col is None:
                    code_col = col
                elif "name" in col_lower and "company" in col_lower:
                    name_col = col
                elif "sector" in col_lower or "industry" in col_lower:
                    sector_col = col

            if code_col:
                log.info(f"  Using columns: code={code_col}, name={name_col}, "
                         f"sector={sector_col}")

                for _, row in jpx_df.iterrows():
                    raw_code = str(row.get(code_col, "")).strip()
                    if not raw_code or not raw_code.isdigit():
                        continue

                    ticker = raw_code.lstrip("0") or raw_code
                    name = str(row.get(name_col, ticker)).strip() if name_col else ticker
                    sector = str(row.get(sector_col, "")).strip() if sector_col else ""

                    # Include if it's a known tech stock or matches tech sector
                    is_known = ticker in _KNOWN_JP_TECH
                    is_tech_sector = _matches_tech_sector(sector)

                    if not is_known and not is_tech_sector:
                        continue

                    if ticker in seen:
                        continue
                    seen.add(ticker)

                    rows.append({
                        "ticker": ticker,
                        "name": name,
                        "sector": sector or "Technology",
                        "industry": "",
                        "exchange": "TSE",
                        "currency": "JPY",
                        "yf_symbol": ticker,
                        "market": "JP",
                    })

                log.info(f"  JPX CSV filtered: {len(rows)} tech stocks")
            else:
                log.warning("Could not identify code column in JPX CSV")

    except Exception as e:
        log.warning(f"JPX CSV fetch failed: {e}. Using known stocks baseline.")

    # Always ensure all known JP tech stocks are included
    for ticker, name in _KNOWN_JP_TECH.items():
        if ticker not in seen:
            rows.append({
                "ticker": ticker,
                "name": name,
                "sector": "Technology",
                "industry": "",
                "exchange": "TSE",
                "currency": "JPY",
                "yf_symbol": ticker,
                "market": "JP",
            })
            seen.add(ticker)

    if not rows:
        log.warning("No JP tech stocks found. Falling back to static watchlist.")
        return _fallback_watchlist()

    result = pd.DataFrame(rows)
    result = result.drop_duplicates(subset=["ticker"], keep="first")
    log.info(f"JP tech universe: {len(result)} unique stocks")

    path = get_data_path("processed", "jp_tech_universe.parquet")
    result.to_parquet(path, index=False)
    log.info(f"Saved to {path}")

    return result


def _fallback_watchlist() -> pd.DataFrame:
    """Load the static YAML watchlist as fallback."""
    from src.universe.yf_universe import build_yf_universe
    log.info("Using static jp_watchlist.yaml as fallback")
    return build_yf_universe("JP", "jp_watchlist.yaml", ticker_suffix="")
