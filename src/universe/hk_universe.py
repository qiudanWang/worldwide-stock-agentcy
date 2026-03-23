"""Dynamic HK tech universe builder using akshare HKEX data."""

import pandas as pd
from src.common.config import get_data_path
from src.common.logger import get_logger

log = get_logger("universe.hk")

# Well-known HK tech stock codes (baseline, always included)
_BASELINE_CODES = {
    "00700", "09988", "03690", "09618", "01024", "09626", "09999", "09888",
    "00981", "01810", "00992", "02382", "00268", "00763", "01347", "09961",
    "01211", "09866", "09868", "02015", "00285", "02018", "03888", "01478",
    "02013", "00799",
}

# Chinese keywords indicating tech / internet / electronics companies
_TECH_NAME_KEYWORDS = [
    "科技", "软件", "半导体", "互联网", "电子", "信息", "智能", "数据",
    "网络", "芯片", "通信", "计算", "云", "数字", "人工智能",
    "腾讯", "阿里", "美团", "小米", "百度", "京东", "网易",
    "快手", "哔哩", "比亚迪电子", "中芯", "华虹", "联想",
    "舜宇", "瑞声", "金蝶", "金山", "中兴",
]


def _normalize_code(raw_code: str) -> str:
    """Normalize HK stock code to 5-digit zero-padded string."""
    return raw_code.strip().zfill(5)


def _strip_leading_zeros(code: str) -> str:
    """Strip leading zeros for the ticker field (e.g., '00700' -> '700')."""
    return str(int(code))


def _matches_tech(name: str, code: str) -> bool:
    """Return True if the stock matches tech criteria by name or code."""
    if code in _BASELINE_CODES:
        return True
    name_lower = name.lower()
    for kw in _TECH_NAME_KEYWORDS:
        if kw in name_lower or kw.lower() in name_lower:
            return True
    return False


def build_hk_tech_universe() -> pd.DataFrame:
    """Fetch HK tech stocks dynamically from akshare HKEX spot data.

    Uses ak.stock_hk_spot_em() to get all HK-listed stocks, then filters
    for tech-related companies by name keywords and a baseline code set.

    Falls back to the static YAML watchlist if the API is unavailable.

    Returns:
        DataFrame with columns: ticker, name, sector, industry, exchange,
        currency, yf_symbol, market
    """
    try:
        import akshare as ak

        log.info("Fetching HK stock listings from akshare...")
        df = ak.stock_hk_spot_em()

        if df is None or df.empty:
            log.warning("akshare returned empty HK data. Falling back.")
            return _fallback_watchlist()

        log.info(f"  Total HK listings: {len(df)}")

        rows = []
        seen = set()

        for _, row in df.iterrows():
            raw_code = str(row.get("代码", "")).strip()
            name = str(row.get("名称", "")).strip()

            if not raw_code:
                continue

            code = _normalize_code(raw_code)

            if not _matches_tech(name, code):
                continue

            ticker = _strip_leading_zeros(code)
            if ticker in seen:
                continue
            seen.add(ticker)

            rows.append({
                "ticker": ticker,
                "name": name,
                "sector": "Technology",
                "industry": "",
                "exchange": "HKEX",
                "currency": "HKD",
                "yf_symbol": ticker,
                "market": "HK",
            })

        # Ensure all baseline codes are present even if akshare data
        # didn't include them (they may have different column formats)
        for code in _BASELINE_CODES:
            ticker = _strip_leading_zeros(code)
            if ticker not in seen:
                rows.append({
                    "ticker": ticker,
                    "name": ticker,
                    "sector": "Technology",
                    "industry": "",
                    "exchange": "HKEX",
                    "currency": "HKD",
                    "yf_symbol": ticker,
                    "market": "HK",
                })
                seen.add(ticker)

    except Exception as e:
        log.warning(f"akshare HK fetch failed: {e}. Falling back to static watchlist.")
        return _fallback_watchlist()

    if not rows:
        log.warning("No HK tech stocks found. Falling back to static watchlist.")
        return _fallback_watchlist()

    result = pd.DataFrame(rows)
    result = result.drop_duplicates(subset=["ticker"], keep="first")
    log.info(f"HK tech universe: {len(result)} unique stocks")

    path = get_data_path("processed", "hk_tech_universe.parquet")
    result.to_parquet(path, index=False)
    log.info(f"Saved to {path}")

    return result


def _fallback_watchlist() -> pd.DataFrame:
    """Load the static YAML watchlist as fallback."""
    from src.universe.yf_universe import build_yf_universe
    log.info("Using static hk_watchlist.yaml as fallback")
    return build_yf_universe("HK", "hk_watchlist.yaml", ticker_suffix="")
