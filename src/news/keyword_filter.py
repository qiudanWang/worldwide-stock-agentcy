import pandas as pd
from src.common.config import load_yaml
from src.common.logger import get_logger

log = get_logger("news.keyword")


def filter_by_keywords(news_df):
    """Filter news items that match configured keywords.

    Adds 'keyword_hits' column with list of matched keywords.
    """
    if news_df.empty:
        return news_df

    cfg = load_yaml("keywords.yaml")
    keywords = [k.lower() for k in cfg["keywords"]]

    def find_hits(title):
        title_lower = str(title).lower()
        return [k for k in keywords if k in title_lower]

    news_df = news_df.copy()
    news_df["keyword_hits"] = news_df["title"].apply(find_hits)
    news_df["hit_count"] = news_df["keyword_hits"].apply(len)

    matched = news_df[news_df["hit_count"] > 0]
    log.info(f"Keyword filter: {len(matched)}/{len(news_df)} news items matched")
    return news_df


def compute_news_counts(news_df):
    """Compute per-ticker news counts.

    Returns DataFrame with: ticker, market, news_count_total, news_count_keyword_matched
    """
    if news_df.empty:
        return pd.DataFrame()

    total = news_df.groupby(["ticker", "market"]).size().reset_index(name="news_count_total")

    if "hit_count" in news_df.columns:
        matched = (
            news_df[news_df["hit_count"] > 0]
            .groupby(["ticker", "market"])
            .size()
            .reset_index(name="news_count_keyword")
        )
        result = total.merge(matched, on=["ticker", "market"], how="left")
        result["news_count_keyword"] = result["news_count_keyword"].fillna(0).astype(int)
    else:
        result = total
        result["news_count_keyword"] = 0

    return result
