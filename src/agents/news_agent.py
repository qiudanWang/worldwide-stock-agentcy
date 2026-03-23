"""News Agent: per-market company news + keyword filtering."""

import pandas as pd
from datetime import datetime

from src.agents.base import BaseAgent, AgentResult
from src.common.config import get_data_path, get_settings
from src.common.logger import get_logger
from src.news.keyword_filter import filter_by_keywords, compute_news_counts

log = get_logger("agent.news")


class NewsAgent(BaseAgent):
    """Fetches company news and applies keyword filtering for a single market."""

    def __init__(self, name, market, market_config, depends_on=None):
        super().__init__(name, agent_type="news", market=market,
                         depends_on=depends_on)
        self.market_config = market_config

    def run(self) -> AgentResult:
        market = self.market
        errors = []

        # Get news tickers from settings (key tickers, not all stocks)
        settings = get_settings()
        news_tickers = settings.get("news_tickers", {}).get(market, [])
        news_tickers = [str(t) for t in news_tickers]

        if not news_tickers:
            # Fall back: load first 15 tickers from universe
            try:
                uni_path = get_data_path("markets", market, "universe.parquet")
                uni = pd.read_parquet(uni_path)
                news_tickers = uni["ticker"].head(15).tolist()
            except Exception:
                return AgentResult(success=True, records_written=0,
                                   errors=["No news tickers configured"])

        self.update_status(progress=f"Scraping {len(news_tickers)} tickers...")

        # Fetch news based on market source
        news_source = self.market_config.get("news_source", "yfinance")
        suffix = self.market_config.get("ticker_suffix", "")

        if news_source == "akshare" and market == "CN":
            from src.news.cn_news import fetch_cn_news_batch
            news_df = fetch_cn_news_batch(news_tickers)
        else:
            from src.news.yf_news import fetch_yf_news_batch
            news_df = fetch_yf_news_batch(news_tickers, market=market,
                                           ticker_suffix=suffix)

        if news_df.empty:
            news_df = pd.DataFrame()

        # Apply keyword filtering
        if not news_df.empty:
            news_df = filter_by_keywords(news_df)

        # Web search for top movers (volume spikes / price movers)
        try:
            self.update_status(progress="Web searching top movers...")
            from src.news.web_search import search_top_movers_news
            market_path = get_data_path("markets", market,
                                        "alerts.json")
            import json, os
            top_movers = []
            if os.path.exists(market_path):
                with open(market_path) as f:
                    alerts = json.load(f)
                seen = set()
                for a in alerts:
                    t = a.get("ticker")
                    if t and t not in seen and a.get("alert_type") in (
                        "volume_spike", "volume_surge", "price_alert"
                    ):
                        top_movers.append({
                            "ticker": t,
                            "market": market,
                            "name": a.get("name", ""),
                        })
                        seen.add(t)
            # Fallback: top 5 by volume ratio from universe
            if not top_movers:
                top_movers = [{"ticker": t, "market": market, "name": ""}
                              for t in news_tickers[:5]]
            web_news = search_top_movers_news(top_movers, max_per_stock=3)
            if web_news:
                web_df = pd.DataFrame(web_news)
                web_df = filter_by_keywords(web_df)
                news_df = pd.concat([news_df, web_df], ignore_index=True) if not news_df.empty else web_df
                log.info(f"[{market}] Web search added {len(web_news)} articles")
        except Exception as e:
            log.warning(f"[{market}] Web search step failed: {e}")

        if news_df.empty:
            return AgentResult(success=True, records_written=0)

        # Save
        news_path = get_data_path("markets", market, "news.parquet")
        news_df.to_parquet(news_path, index=False)

        matched = len(news_df[news_df["hit_count"] > 0]) if "hit_count" in news_df.columns else 0
        self.update_status(
            progress=f"{len(news_df)} articles, {matched} matched"
        )

        return AgentResult(
            success=True,
            records_written=len(news_df),
            errors=errors,
        )
