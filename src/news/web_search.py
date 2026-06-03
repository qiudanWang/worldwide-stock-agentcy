"""Web search for stock news via DuckDuckGo (no API key required)."""

import time
import concurrent.futures
from src.common.logger import get_logger
from src.common.timeout import default_timeout
from src.common.tracing import observe

log = get_logger("news.web_search")


@observe(name="search_stock_news", type="tool")
def search_stock_news(ticker: str, company_name: str = "", market: str = "",
                      max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo for recent news about a stock.

    Args:
        ticker: Stock ticker code.
        company_name: Optional company name for better search results.
        market: Market code (CN, US, HK, etc.) for context.
        max_results: Maximum number of results to return.

    Returns:
        List of dicts with keys: ticker, title, link, publisher, date, source.
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
    except ImportError:
        log.warning("ddgs not installed. Run: pip install ddgs")
        return []

    query = company_name if company_name else ticker
    if market == "CN":
        query = f"{query} {ticker} 股票 新闻"
    else:
        query = f"{query} {ticker} stock news"

    def _do_search():
        results = []
        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=max_results):
                results.append({
                    "ticker": ticker,
                    "market": market,
                    "title": r.get("title", ""),
                    "link": r.get("url", ""),
                    "publisher": r.get("source", ""),
                    "date": r.get("date", "")[:10] if r.get("date") else "",
                    "source": "web_search",
                })
        return results

    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        return ex.submit(_do_search).result(timeout=default_timeout())
    except concurrent.futures.TimeoutError:
        log.warning(f"Web search timeout for {ticker}, skipping")
        return []
    except Exception as e:
        log.warning(f"Web search failed for {ticker}: {e}")
        return []
    finally:
        ex.shutdown(wait=False)


@observe(name="search_top_movers_news", type="tool")
def search_top_movers_news(movers: list[dict], max_per_stock: int = 3) -> list[dict]:
    """Search news for a list of top movers.

    Args:
        movers: List of dicts with keys: ticker, market, name (optional).
        max_per_stock: Max news articles per stock.

    Returns:
        Combined list of news articles.
    """
    all_news = []
    for i, mover in enumerate(movers):
        ticker = mover.get("ticker", "")
        market = mover.get("market", "")
        name = mover.get("name", "")
        if not ticker:
            continue
        if i > 0:
            time.sleep(0.5)  # be polite to DDG
        news = search_stock_news(ticker, name, market, max_per_stock)
        all_news.extend(news)
        log.info(f"Web search [{market}] {ticker}: {len(news)} articles")
    return all_news
