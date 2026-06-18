"""Tool definitions for the openai-agents SDK.

Each tool is a plain function decorated with @function_tool.
Business logic lives in local_trading_agent.py / openbb_tools.py — not here.
"""

import os

from agents import function_tool
from src.common.tracing import observe

# data_dir is resolved once at import time
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@function_tool
@observe(name="get_stock_data", type="tool")
def get_stock_data(ticker: str, market: str) -> str:
    """Get raw price, returns (1d/5d/20d), volume, market cap, sector, and financials for a stock.
    Use only when you need quick data lookup for multiple stocks or as input to other tools.
    For single-stock questions from the user, use deep_analysis instead."""
    from web.agent_llm import _load_ticker_context
    return _load_ticker_context(ticker, market, _DATA_DIR)


@function_tool
@observe(name="fetch_url", type="tool")
def fetch_url(url: str) -> str:
    """Fetch and read the full text of a specific URL (news article, announcement, report)."""
    from web.agent_llm import _fetch_url
    return _fetch_url(url)


@function_tool
@observe(name="get_peers", type="tool")
def get_peers(ticker: str, market: str) -> str:
    """Get peer company comparison for a stock — peer list with price performance and financials."""
    from web.local_trading_agent import local_peers_analyze
    return local_peers_analyze(ticker, market, _DATA_DIR)


@function_tool
@observe(name="get_sector", type="tool")
def get_sector(sector: str, market: str) -> str:
    """Analyze a market sector — top stocks, performance ranking, sector trends."""
    from web.local_trading_agent import local_sector_analyze
    return local_sector_analyze(sector, market, _DATA_DIR)


@function_tool
@observe(name="get_sector_ranking", type="tool")
def get_sector_ranking(market: str, period: str = "1d") -> str:
    """Rank ALL sectors in a market by average return.
    Use this when the user asks which sector rose the most, sector leaderboard,
    or best/worst performing sector. period: 1d, 5d, or 20d."""
    from web.local_trading_agent import local_sector_ranking
    return local_sector_ranking(market, _DATA_DIR, period)


@function_tool
@observe(name="deep_analysis", type="tool")
def deep_analysis(ticker: str, market: str) -> str:
    """Run multi-agent analysis on a single stock: technicals, fundamentals, sentiment, and bull/bear debate.
    Use this as the default tool whenever the user asks about a specific stock — e.g. 'how is AAPL doing',
    'analyze 688256', 'what do you think about Tesla'. Takes 30-60s."""
    from web.agent_llm import trading_agents_analyze
    from web.local_trading_agent import local_trading_analyze
    if market == "US":
        return trading_agents_analyze(ticker)
    return local_trading_analyze(ticker, _DATA_DIR)


@function_tool
@observe(name="get_live_quote", type="tool")
def get_live_quote(ticker: str) -> str:
    """Get live real-time price quote. For non-US add exchange suffix: 0700.HK, 7203.T, 005930.KS"""
    from web.openbb_tools import get_stock_quote
    return get_stock_quote(ticker)


@function_tool
@observe(name="get_market_movers", type="tool")
def get_market_movers(type: str = "gainers") -> str:
    """Get top gaining, losing, or most active stocks in the US market right now.
    type: gainers, losers, or active."""
    from web.openbb_tools import get_gainers, get_losers, get_most_active
    if type == "losers":
        return get_losers()
    elif type == "active":
        return get_most_active()
    return get_gainers()


@function_tool
@observe(name="get_indices", type="tool")
def get_indices() -> str:
    """Get current levels of major global indices: S&P 500, Nasdaq, Nikkei, HSI, CSI 300, etc."""
    from web.openbb_tools import get_major_indices
    return get_major_indices()


@function_tool
@observe(name="get_news", type="tool")
def get_news(ticker: str) -> str:
    """Get latest news articles for a company."""
    from web.openbb_tools import get_company_news
    return get_company_news(ticker)


@function_tool
@observe(name="get_fundamentals", type="tool")
def get_fundamentals(ticker: str) -> str:
    """Get key financial ratios: P/E, EPS, revenue, profit margin, ROE, debt/equity."""
    from web.openbb_tools import get_fundamentals
    return get_fundamentals(ticker)


# ---------------------------------------------------------------------------
# Tool list (passed to Agent)
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    get_stock_data,
    fetch_url,
    get_peers,
    get_sector,
    get_sector_ranking,
    deep_analysis,
    get_live_quote,
    get_market_movers,
    get_indices,
    get_news,
    get_fundamentals,
]
