"""Fetch and cache fundamentals for non-CN markets via yfinance.

Called by DataAgent (weekly refresh) and as a live fallback.
Saves data/markets/{MARKET}/fundamentals.parquet.
"""
import time
import pandas as pd

try:
    from src.common.logger import get_logger
    log = get_logger("financials.yf")
except Exception:
    import logging
    log = logging.getLogger("financials.yf")

_YF_SUFFIX = {
    "HK": ".HK", "JP": ".T", "AU": ".AX", "IN": ".NS",
    "KR": ".KS", "TW": ".TW", "DE": ".DE", "FR": ".PA",
    "UK": ".L",  "BR": ".SA", "SA": ".SR", "US": "",
}

_WANTED_INFO = [
    "shortName", "sector", "industry",
    "trailingPE", "forwardPE", "priceToBook",
    "returnOnEquity", "profitMargins", "grossMargins", "operatingMargins",
    "debtToEquity", "currentRatio",
    "totalRevenue", "netIncomeToCommon", "freeCashflow", "ebitda",
    "revenueGrowth", "earningsGrowth",
    "trailingEps", "forwardEps",
    "marketCap", "dividendYield",
    "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
    "targetMeanPrice", "targetHighPrice", "targetLowPrice",
    "recommendationKey", "numberOfAnalystOpinions",
]


def _fetch_one(yf_ticker: str) -> dict:
    """Fetch fundamentals for a single yfinance symbol. Returns {} on failure."""
    try:
        import yfinance as yf
        t = yf.Ticker(yf_ticker)
        info = t.info or {}
        # Consider it empty if no meaningful financial data
        if not info.get("marketCap") and not info.get("totalRevenue") and not info.get("trailingPE"):
            return {}

        result = {"yf_ticker": yf_ticker}
        for k in _WANTED_INFO:
            v = info.get(k)
            if v is not None:
                result[k] = v

        # Quarterly revenue + net income (last 3 quarters, in 亿 units)
        try:
            qf = t.quarterly_financials
            if qf is not None and not qf.empty:
                for col in qf.columns[:3]:
                    q = str(col.date())
                    rev = qf.loc["Total Revenue", col] if "Total Revenue" in qf.index else None
                    ni  = qf.loc["Net Income",    col] if "Net Income"    in qf.index else None
                    if rev is not None:
                        result[f"q_rev_{q}"] = round(float(rev) / 1e8, 2)
                    if ni is not None:
                        result[f"q_ni_{q}"]  = round(float(ni)  / 1e8, 2)
        except Exception:
            pass

        # Latest balance sheet snapshot
        try:
            bs = t.quarterly_balance_sheet
            if bs is not None and not bs.empty:
                col = bs.columns[0]
                for label, key in [
                    ("bs_cash",   "Cash And Cash Equivalents"),
                    ("bs_debt",   "Total Debt"),
                    ("bs_equity", "Stockholders Equity"),
                    ("bs_assets", "Total Assets"),
                ]:
                    if key in bs.index:
                        v = bs.loc[key, col]
                        if pd.notna(v):
                            result[label] = round(float(v) / 1e8, 2)
                result["bs_date"] = str(col.date())
        except Exception:
            pass

        return result
    except Exception:
        return {}


def fetch_yf_fundamentals_batch(tickers: list, market: str,
                                 delay: float = 0.4, limit: int = 60) -> pd.DataFrame:
    """Fetch fundamentals for up to `limit` tickers in a market.

    Returns a DataFrame with columns: ticker, market, + fundamental fields.
    Empty DataFrame if all fetches fail.
    """
    suffix = _YF_SUFFIX.get(market, "")
    tickers = tickers[:limit]
    results = []

    for i, ticker in enumerate(tickers):
        yf_ticker = ticker if (not suffix or str(ticker).endswith(suffix)) \
                    else str(ticker) + suffix
        row = _fetch_one(yf_ticker)
        if row:
            row["ticker"] = ticker
            row["market"] = market
            results.append(row)
        if (i + 1) % 20 == 0:
            log.info(f"  [{market}] Fundamentals: {i+1}/{len(tickers)} ({len(results)} ok)")
            time.sleep(delay)

    if not results:
        log.warning(f"[{market}] No fundamentals fetched for any of {len(tickers)} tickers")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    log.info(f"[{market}] Fundamentals: {len(df)}/{len(tickers)} stocks fetched")
    return df
