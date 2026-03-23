import yfinance as yf
import pandas as pd
from datetime import datetime
from src.common.logger import get_logger

log = get_logger("news.us")


def fetch_us_news(ticker):
    """Fetch recent news for a US stock via yfinance."""
    try:
        t = yf.Ticker(ticker)
        news = t.news
        if not news:
            return []

        results = []
        for item in news:
            results.append({
                "ticker": ticker,
                "market": "US",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "title": item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "link": item.get("link", ""),
                "source": "yfinance",
            })
        return results
    except Exception as e:
        log.warning(f"Failed to fetch news for {ticker}: {e}")
        return []


def fetch_us_news_batch(tickers):
    """Fetch news for multiple US tickers."""
    all_news = []
    for i, ticker in enumerate(tickers):
        if i > 0 and i % 10 == 0:
            log.info(f"  Progress: {i}/{len(tickers)}")
        news = fetch_us_news(ticker)
        all_news.extend(news)

    if not all_news:
        return pd.DataFrame()

    df = pd.DataFrame(all_news)
    log.info(f"Fetched {len(df)} US news items for {len(tickers)} tickers")
    return df
