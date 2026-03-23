"""Fetch news for key stocks and World Monitor geopolitical context."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from src.common.config import get_data_path
from src.news.us_news import fetch_us_news_batch
from src.news.cn_news import fetch_cn_news_batch
from src.news.keyword_filter import filter_by_keywords, compute_news_counts
from src.news.world_monitor import fetch_geopolitical_context
from src.common.logger import get_logger

log = get_logger("script.news")

# Focus on key stocks for news (not all 600+)
CN_NEWS_TICKERS = [
    "002230", "688256", "000977", "603019", "301269",
    "300454", "688561", "600588", "600570", "688111",
    "688981", "603501", "000063", "000938", "688041",
]
US_NEWS_TICKERS = [
    "NVDA", "AMD", "MSFT", "GOOGL", "META", "PLTR",
    "ORCL", "PANW", "CRWD", "SMCI", "TSM", "AVGO",
]

if __name__ == "__main__":
    # Fetch per-company news
    log.info("Fetching US news...")
    us_news = fetch_us_news_batch(US_NEWS_TICKERS)

    log.info("Fetching CN news...")
    cn_news = fetch_cn_news_batch(CN_NEWS_TICKERS)

    all_news = pd.concat([us_news, cn_news], ignore_index=True)

    if not all_news.empty:
        all_news = filter_by_keywords(all_news)
        counts = compute_news_counts(all_news)

        path = get_data_path("processed", "news_feed.parquet")
        all_news.to_parquet(path, index=False)

        print(f"\nNews: {len(all_news)} items, {all_news['ticker'].nunique()} tickers")
        if not counts.empty:
            print("\nNews counts:")
            print(counts.sort_values("news_count_keyword", ascending=False).head(10).to_string())

    # Fetch World Monitor geopolitical context
    log.info("Fetching World Monitor context...")
    context = fetch_geopolitical_context()
    if context:
        path = get_data_path("processed", "geopolitical_context.json")
        with open(path, "w") as f:
            json.dump(context, f, indent=2, default=str)
        log.info(f"Saved geopolitical context to {path}")
