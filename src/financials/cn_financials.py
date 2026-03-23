import akshare as ak
import pandas as pd
from src.common.logger import get_logger

log = get_logger("financials.cn")


def fetch_cn_financials(ticker):
    """Fetch key financial indicators for a single A-share stock."""
    try:
        df = ak.stock_financial_abstract_ths(symbol=ticker, indicator="按报告期")
        if df is None or df.empty:
            return {}

        latest = df.iloc[0]

        result = {
            "ticker": ticker,
            "market": "CN",
            "report_date": str(latest.get("报告期", "")),
            "eps": _safe_float(latest.get("基本每股收益")),
            "revenue": _safe_float(latest.get("营业总收入")),
            "revenue_growth": _safe_float(latest.get("营业总收入同比增长率")),
            "net_profit": _safe_float(latest.get("净利润")),
            "net_margin": _safe_float(latest.get("销售净利率")),
            "roe": _safe_float(latest.get("净资产收益率")),
        }
        return result
    except Exception as e:
        log.warning(f"Failed to fetch financials for {ticker}: {e}")
        return {}


def fetch_cn_financials_batch(tickers):
    """Fetch financials for a list of A-share tickers."""
    results = []
    for i, ticker in enumerate(tickers):
        if i > 0 and i % 20 == 0:
            log.info(f"  Progress: {i}/{len(tickers)}")
        data = fetch_cn_financials(ticker)
        if data:
            results.append(data)

    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    log.info(f"Fetched CN financials for {len(df)} stocks")
    return df


def _safe_float(val):
    try:
        if val is None or str(val).strip() in ("", "-", "None", "nan"):
            return None
        return float(val)
    except (ValueError, TypeError):
        return None
