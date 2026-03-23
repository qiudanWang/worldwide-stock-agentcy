import akshare as ak
import pandas as pd
from datetime import datetime
from src.common.logger import get_logger

log = get_logger("news.cn")


def fetch_cn_news(ticker):
    """Fetch recent news for an A-share stock via AKShare."""
    try:
        # Temporarily use python string storage to avoid pyarrow regex bug
        # with AKShare's \u3000 pattern
        original = pd.options.mode.string_storage
        pd.options.mode.string_storage = "python"
        try:
            df = ak.stock_news_em(symbol=ticker)
        finally:
            pd.options.mode.string_storage = original

        if df is None or df.empty:
            return []

        results = []
        for _, row in df.iterrows():
            results.append({
                "ticker": ticker,
                "market": "CN",
                "date": str(row.get("发布时间", datetime.now().strftime("%Y-%m-%d"))),
                "title": row.get("新闻标题", ""),
                "publisher": row.get("文章来源", ""),
                "link": row.get("新闻链接", ""),
                "source": "eastmoney",
            })
        return results
    except Exception as e:
        log.warning(f"Failed to fetch news for {ticker}: {e}")
        return []


def fetch_cn_news_batch(tickers):
    """Fetch news for multiple A-share tickers."""
    all_news = []
    for i, ticker in enumerate(tickers):
        if i > 0 and i % 10 == 0:
            log.info(f"  Progress: {i}/{len(tickers)}")
        news = fetch_cn_news(ticker)
        all_news.extend(news)

    if not all_news:
        return pd.DataFrame()

    df = pd.DataFrame(all_news)
    log.info(f"Fetched {len(df)} CN news items for {len(tickers)} tickers")
    return df
