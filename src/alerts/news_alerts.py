import pandas as pd
from src.common.config import load_yaml
from src.common.logger import get_logger

log = get_logger("alerts.news")


def check_news_alerts(news_counts_df):
    """Find stocks with unusual news activity."""
    rules = load_yaml("alert_rules.yaml")
    threshold = rules["news_alert"]["spike_threshold_7d"]

    if news_counts_df.empty:
        return pd.DataFrame()

    col = "news_count_keyword" if "news_count_keyword" in news_counts_df.columns else "news_count_total"
    alerts = news_counts_df[news_counts_df[col] >= threshold].copy()
    alerts = alerts.sort_values(col, ascending=False)

    if not alerts.empty:
        log.info(f"News alerts: {len(alerts)} stocks above {threshold} keyword hits")
    else:
        log.info("No news alerts")

    return alerts
