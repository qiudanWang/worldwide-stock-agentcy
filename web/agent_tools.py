"""Tool schemas and executor for the LiteLLM agent loop.

Each tool maps 1-to-1 to an existing implementation — no logic is duplicated here,
this file is purely dispatch + schema definitions.
"""

import json
import os
import requests

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080")

# ---------------------------------------------------------------------------
# Web search via local SearXNG
# ---------------------------------------------------------------------------

def _searxng_search(query: str, language: str = "en-US", max_results: int = 5) -> str:
    try:
        r = requests.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json", "language": language},
            timeout=10,
        )
        r.raise_for_status()
        results = r.json().get("results", [])[:max_results]
        if not results:
            return "(no results found)"
        parts = []
        for item in results:
            title   = item.get("title", "")
            content = item.get("content", "")[:400]
            url     = item.get("url", "")
            parts.append(f"• {title}\n  {content}\n  {url}")
        return f"[Search: {query}]\n\n" + "\n\n".join(parts)
    except Exception as e:
        return f"(search failed: {e})"


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

def execute_tool(name: str, inputs: dict, context: dict, data_dir: str) -> str:
    """Dispatch a tool call to its implementation. Returns a string for the LLM."""
    try:
        if name == "get_stock_data":
            from web.agent_llm import _load_ticker_context
            ticker = inputs.get("ticker", "")
            market = inputs.get("market") or context.get("market", "")
            return _load_ticker_context(ticker, market, data_dir)

        elif name == "web_search":
            query = inputs["query"]
            lang  = "zh-CN" if inputs.get("language") == "zh" else "en-US"
            return _searxng_search(query, language=lang)

        elif name == "fetch_url":
            from web.agent_llm import _fetch_url
            return _fetch_url(inputs["url"])

        elif name == "get_peers":
            from web.local_trading_agent import local_peers_analyze
            ticker = inputs["ticker"]
            market = inputs.get("market") or context.get("market", "CN")
            return local_peers_analyze(ticker, market, data_dir)

        elif name == "get_sector":
            from web.local_trading_agent import local_sector_analyze
            market = inputs.get("market") or context.get("market", "CN")
            return local_sector_analyze(inputs["sector"], market, data_dir)

        elif name == "deep_analysis":
            from web.local_trading_agent import local_trading_analyze
            return local_trading_analyze(inputs["ticker"], data_dir)

        elif name == "get_live_quote":
            from web.openbb_tools import get_stock_quote
            return get_stock_quote(inputs["ticker"])

        elif name == "get_market_movers":
            from web.openbb_tools import get_gainers, get_losers, get_most_active
            t = inputs.get("type", "gainers")
            if t == "losers":
                return get_losers()
            elif t == "active":
                return get_most_active()
            else:
                return get_gainers()

        elif name == "get_indices":
            from web.openbb_tools import get_major_indices
            return get_major_indices()

        elif name == "get_news":
            from web.openbb_tools import get_company_news
            return get_company_news(inputs["ticker"])

        elif name == "get_fundamentals":
            from web.openbb_tools import get_fundamentals
            return get_fundamentals(inputs["ticker"])

        else:
            return f"Unknown tool: {name}"

    except Exception as e:
        return f"Tool '{name}' failed: {e}"


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI / LiteLLM format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_data",
            "description": (
                "Get price, returns (1d/5d/20d), volume, market cap, sector, and local financials "
                "for a stock from the local database. Use this first for any stock question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker e.g. 600519, AAPL, 0700"},
                    "market": {"type": "string", "description": "Market code: CN, US, HK, JP, KR, TW, IN, UK, DE, FR"},
                },
                "required": ["ticker", "market"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the internet for real-time news, recent financials, analyst estimates, "
                "or any information not in local data. "
                "For CN stocks use the 6-digit ticker code as the query (e.g. '603290'). "
                "Set language='zh' for Chinese market searches."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":    {"type": "string"},
                    "language": {"type": "string", "enum": ["en", "zh"],
                                 "description": "Use 'zh' for Chinese stocks/news, 'en' otherwise"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch and read the full text of a specific URL (news article, announcement, report).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_peers",
            "description": "Get peer company comparison for a stock — peer list with price performance and financials.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "market": {"type": "string"},
                },
                "required": ["ticker", "market"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sector",
            "description": "Analyze a market sector — top stocks, performance ranking, sector trends.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sector": {"type": "string", "description": "Sector name e.g. '半导体', 'Technology', 'Consumer'"},
                    "market": {"type": "string"},
                },
                "required": ["sector", "market"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deep_analysis",
            "description": (
                "Run deep multi-agent analysis: fundamentals + sentiment + technicals + bull/bear debate. "
                "Takes 30-60s. Only use when the user explicitly asks for deep/full analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_live_quote",
            "description": "Get live real-time price quote. For non-US add exchange suffix: 0700.HK, 7203.T, 005930.KS",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_movers",
            "description": "Get top gaining, losing, or most active stocks in the US market right now.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["gainers", "losers", "active"]},
                },
                "required": ["type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_indices",
            "description": "Get current levels of major global indices: S&P 500, Nasdaq, Nikkei, HSI, CSI 300, etc.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Get latest news articles for a company.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fundamentals",
            "description": "Get key financial ratios: P/E, EPS, revenue, profit margin, ROE, debt/equity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                },
                "required": ["ticker"],
            },
        },
    },
]
