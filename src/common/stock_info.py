"""Stock info utilities: board/exchange resolution and US stock name mapping."""

import pandas as pd
from src.common.config import get_data_path
from src.common.logger import get_logger

log = get_logger("common.stock_info")

# US stock full names
US_STOCK_NAMES = {
    "NVDA": "NVIDIA Corp.",
    "AMD": "Advanced Micro Devices",
    "AVGO": "Broadcom Inc.",
    "MRVL": "Marvell Technology",
    "GOOGL": "Alphabet Inc.",
    "META": "Meta Platforms",
    "MSFT": "Microsoft Corp.",
    "PLTR": "Palantir Technologies",
    "AI": "C3.ai Inc.",
    "AMZN": "Amazon.com Inc.",
    "SNOW": "Snowflake Inc.",
    "NET": "Cloudflare Inc.",
    "ORCL": "Oracle Corp.",
    "CRM": "Salesforce Inc.",
    "NOW": "ServiceNow Inc.",
    "WDAY": "Workday Inc.",
    "INTU": "Intuit Inc.",
    "PANW": "Palo Alto Networks",
    "CRWD": "CrowdStrike Holdings",
    "FTNT": "Fortinet Inc.",
    "ZS": "Zscaler Inc.",
    "QCOM": "Qualcomm Inc.",
    "ARM": "Arm Holdings",
    "TSM": "Taiwan Semiconductor",
    "INTC": "Intel Corp.",
    "SNPS": "Synopsys Inc.",
    "CDNS": "Cadence Design Systems",
    "SMCI": "Super Micro Computer",
    "DELL": "Dell Technologies",
    "HPE": "Hewlett Packard Enterprise",
    "CSCO": "Cisco Systems",
    "ANET": "Arista Networks",
    "JNPR": "Juniper Networks",
    "ADBE": "Adobe Inc.",
    "COIN": "Coinbase Global",
    "SQ": "Block Inc.",
    "AFRM": "Affirm Holdings",
}


def get_cn_board(ticker):
    """Determine the board/exchange for a Chinese A-share ticker."""
    if ticker.startswith("300"):
        return "ChiNext 创业板 (Shenzhen)"
    elif ticker.startswith("688"):
        return "STAR 科创板 (Shanghai)"
    elif ticker.startswith("002"):
        return "SME 中小板 (Shenzhen)"
    elif ticker.startswith("001") or ticker.startswith("000"):
        return "Main Board (Shenzhen)"
    elif ticker.startswith("6"):
        return "Main Board (Shanghai)"
    else:
        return "A-Share"


def get_us_exchange(ticker):
    """Determine the exchange for a US stock ticker."""
    nyse_tickers = {"BRK.B", "JPM", "V", "MA", "BAC", "WMT", "JNJ", "PG",
                    "HD", "ORCL", "CRM", "DELL", "HPE", "ARM", "COIN", "SQ",
                    "AFRM", "AVGO", "TSM", "NOW", "WDAY"}
    if ticker in nyse_tickers:
        return "NYSE"
    return "NASDAQ"


def get_board(ticker, market):
    """Get the board/exchange string for a ticker."""
    if market == "CN":
        return get_cn_board(ticker)
    elif market == "US":
        return get_us_exchange(ticker)
    elif market == "HK":
        return "HKEX"
    elif market == "JP":
        return "TSE"
    elif market == "IN":
        return "NSE"
    elif market == "UK":
        return "LSE"
    elif market in ("DE", "FR"):
        return "Euronext"
    elif market == "KR":
        return "KRX"
    elif market == "TW":
        return "TWSE"
    elif market == "AU":
        return "ASX"
    elif market == "BR":
        return "B3"
    elif market == "SA":
        return "Tadawul"
    return market


def get_stock_name(ticker, market):
    """Get the stock name for a ticker."""
    if market == "US":
        return US_STOCK_NAMES.get(ticker, ticker)
    # For CN stocks, try to look up from universe
    return ticker


def load_name_lookup():
    """Load a ticker->name lookup dict from the universe master.

    Tries global/universe_master.parquet first, falls back to legacy location.
    """
    paths = [
        get_data_path("global", "universe_master.parquet"),
        get_data_path("processed", "tech_universe_master.parquet"),
    ]
    for path in paths:
        try:
            df = pd.read_parquet(path, columns=["ticker", "name", "market"])
            lookup = {}
            for _, row in df.iterrows():
                if row["market"] == "US":
                    lookup[row["ticker"]] = US_STOCK_NAMES.get(row["ticker"], row["name"])
                else:
                    lookup[row["ticker"]] = row["name"]
            return lookup
        except Exception:
            continue
    log.warning("Could not load name lookup from any location")
    return {}


def enrich_alert(alert, name_lookup=None):
    """Add 'name' and 'board' fields to an alert dict."""
    ticker = alert.get("ticker", "")
    market = alert.get("market", "")
    alert["board"] = get_board(ticker, market)

    if name_lookup and ticker in name_lookup:
        alert["name"] = name_lookup[ticker]
    elif market == "US":
        alert["name"] = US_STOCK_NAMES.get(ticker, ticker)
    else:
        alert["name"] = ticker

    return alert


def enrich_alerts(alerts_list):
    """Add 'name' and 'board' fields to all alerts."""
    name_lookup = load_name_lookup()
    return [enrich_alert(a, name_lookup) for a in alerts_list]
