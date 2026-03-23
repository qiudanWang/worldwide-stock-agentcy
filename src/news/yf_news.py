"""Generalized yfinance news fetcher for all non-CN markets."""

import yfinance as yf
import pandas as pd
from datetime import datetime
from src.common.logger import get_logger

log = get_logger("news.yf")


def fetch_yf_news(ticker, yf_symbol=None, market="US"):
    """Fetch recent news for a stock via yfinance.

    Args:
        ticker: Raw ticker code.
        yf_symbol: Full yfinance symbol. If None, uses ticker.
        market: Market code for tagging.
    """
    symbol = yf_symbol or ticker
    try:
        t = yf.Ticker(symbol)
        news = t.news
        if not news:
            return []

        results = []
        for item in news:
            # yfinance >= 0.2.x nests fields under "content"
            content = item.get("content") or {}
            title = content.get("title") or item.get("title", "")
            publisher = (
                (content.get("provider") or {}).get("displayName")
                or item.get("publisher", "")
            )
            link = (
                (content.get("canonicalUrl") or {}).get("url")
                or (content.get("clickThroughUrl") or {}).get("url")
                or item.get("link", "")
            )
            pub_date = content.get("pubDate") or ""
            date_str = pub_date[:10] if pub_date else datetime.now().strftime("%Y-%m-%d")
            if not title:
                continue
            results.append({
                "ticker": ticker,
                "market": market,
                "date": date_str,
                "title": title,
                "publisher": publisher,
                "link": link,
                "source": "yfinance",
            })
        return results
    except Exception as e:
        log.warning(f"Failed to fetch news for {symbol}: {e}")
        return []


def fetch_yf_news_batch(tickers, market="US", ticker_suffix=""):
    """Fetch news for multiple tickers via yfinance.

    Args:
        tickers: List of raw ticker codes.
        market: Market code.
        ticker_suffix: Suffix for yfinance symbols.
    """
    all_news = []
    for i, ticker in enumerate(tickers):
        if i > 0 and i % 10 == 0:
            log.info(f"  [{market}] News progress: {i}/{len(tickers)}")
        yf_symbol = f"{ticker}{ticker_suffix}" if ticker_suffix else ticker
        news = fetch_yf_news(ticker, yf_symbol, market)
        all_news.extend(news)

    if not all_news:
        return pd.DataFrame()

    df = pd.DataFrame(all_news)
    log.info(f"[{market}] Fetched {len(df)} news items for {len(tickers)} tickers")
    return df
