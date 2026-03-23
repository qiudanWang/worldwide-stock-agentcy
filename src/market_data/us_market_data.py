import yfinance as yf
import pandas as pd
from src.common.config import get_settings
from src.common.logger import get_logger

log = get_logger("market.us")


def fetch_us_stock_history(ticker, days=None):
    """Fetch daily OHLCV for a single US stock."""
    if days is None:
        days = get_settings()["market_data"]["us_history_days"]

    period = f"{days}d"
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period)
        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        df = df.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        df["ticker"] = ticker
        df["market"] = "US"
        df["turnover"] = 0.0  # yfinance doesn't provide turnover
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df[["date", "ticker", "market", "open", "high", "low", "close", "volume", "turnover"]]
    except Exception as e:
        log.warning(f"Failed to fetch {ticker}: {e}")
        return pd.DataFrame()


def fetch_us_batch(tickers):
    """Fetch history for a list of US tickers."""
    all_data = []
    for i, ticker in enumerate(tickers):
        if i > 0 and i % 10 == 0:
            log.info(f"  Progress: {i}/{len(tickers)}")
        df = fetch_us_stock_history(ticker)
        if not df.empty:
            all_data.append(df)

    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)
