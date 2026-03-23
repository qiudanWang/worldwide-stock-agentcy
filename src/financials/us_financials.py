import yfinance as yf
import pandas as pd
from src.common.logger import get_logger

log = get_logger("financials.us")


def fetch_us_financials(ticker):
    """Fetch key financial indicators for a single US stock."""
    try:
        t = yf.Ticker(ticker)
        info = t.info

        revenue = info.get("totalRevenue")
        gross_profit = info.get("grossProfits")
        operating_income = info.get("operatingIncome") or info.get("ebitda")
        net_income = info.get("netIncomeToCommon")
        market_cap = info.get("marketCap")

        gross_margin = None
        if revenue and gross_profit and revenue > 0:
            gross_margin = round(gross_profit / revenue * 100, 2)

        operating_margin = None
        if revenue and operating_income and revenue > 0:
            operating_margin = round(operating_income / revenue * 100, 2)

        net_margin = None
        if revenue and net_income and revenue > 0:
            net_margin = round(net_income / revenue * 100, 2)

        result = {
            "ticker": ticker,
            "market": "US",
            "report_date": "",
            "revenue": revenue,
            "revenue_growth": info.get("revenueGrowth"),
            "gross_margin": gross_margin,
            "operating_margin": operating_margin,
            "net_margin": net_margin,
            "rd_ratio": None,
            "eps": info.get("trailingEps"),
            "pe": info.get("trailingPE"),
            "ps": info.get("priceToSalesTrailing12Months"),
            "market_cap": market_cap,
            "earnings_date": _get_next_earnings(info),
        }
        return result
    except Exception as e:
        log.warning(f"Failed to fetch financials for {ticker}: {e}")
        return {}


def fetch_us_financials_batch(tickers):
    """Fetch financials for a list of US tickers."""
    results = []
    for i, ticker in enumerate(tickers):
        if i > 0 and i % 10 == 0:
            log.info(f"  Progress: {i}/{len(tickers)}")
        data = fetch_us_financials(ticker)
        if data:
            results.append(data)

    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    log.info(f"Fetched US financials for {len(df)} stocks")
    return df


def _get_next_earnings(info):
    try:
        ts = info.get("mostRecentQuarter")
        if ts:
            return pd.Timestamp(ts, unit="s").strftime("%Y-%m-%d")
    except Exception:
        pass
    return None
