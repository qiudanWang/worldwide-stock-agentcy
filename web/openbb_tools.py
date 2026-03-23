"""OpenBB-powered real-time market data tools for LLM agents.

These functions fetch live data via OpenBB (yfinance provider by default,
no API key required) and return compact text summaries ready for LLM context.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

_obb = None

def _get_obb():
    global _obb
    if _obb is None:
        from openbb import obb
        _obb = obb
    return _obb


def _safe(fn, *args, **kwargs):
    """Call an OpenBB function, return empty DataFrame on failure."""
    try:
        result = fn(*args, **kwargs)
        return result.to_df()
    except Exception:
        return pd.DataFrame()


def _df_text(df, max_rows=20):
    if df.empty:
        return "(no data)"
    return df.head(max_rows).to_string(index=False, max_colwidth=50)


# ── Real-time tools ─────────────────────────────────────────────────

def get_gainers(limit=10) -> str:
    """Top gaining stocks right now (US market)."""
    obb = _get_obb()
    df = _safe(obb.equity.discovery.gainers, sort="desc", provider="yfinance")
    if df.empty:
        return "No gainer data available."
    cols = [c for c in ["symbol", "name", "price", "percent_change", "volume"] if c in df.columns]
    return "TOP GAINERS (live):\n" + _df_text(df[cols], limit)


def get_losers(limit=10) -> str:
    """Top losing stocks right now (US market)."""
    obb = _get_obb()
    df = _safe(obb.equity.discovery.losers, sort="desc", provider="yfinance")
    if df.empty:
        return "No loser data available."
    cols = [c for c in ["symbol", "name", "price", "percent_change", "volume"] if c in df.columns]
    return "TOP LOSERS (live):\n" + _df_text(df[cols], limit)


def get_most_active(limit=10) -> str:
    """Most actively traded stocks right now."""
    obb = _get_obb()
    df = _safe(obb.equity.discovery.active, sort="desc", provider="yfinance")
    if df.empty:
        return "No activity data available."
    cols = [c for c in ["symbol", "name", "price", "percent_change", "volume"] if c in df.columns]
    return "MOST ACTIVE (live):\n" + _df_text(df[cols], limit)


def get_stock_quote(tickers: str) -> str:
    """Real-time quote for one or more tickers (comma-separated)."""
    obb = _get_obb()
    df = _safe(obb.equity.price.quote, tickers, provider="yfinance")
    if df.empty:
        return f"No quote data for {tickers}."
    cols = [c for c in ["symbol", "name", "last_price", "prev_close", "open",
                         "high", "low", "volume", "year_high", "year_low"] if c in df.columns]
    return f"QUOTE ({tickers}):\n" + _df_text(df[cols])


def get_stock_history(ticker: str, days: int = 30) -> str:
    """Recent price history for a ticker."""
    obb = _get_obb()
    df = _safe(obb.equity.price.historical, ticker, provider="yfinance")
    if df.empty:
        return f"No history for {ticker}."
    cols = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    if "date" not in df.columns:
        df = df.reset_index()
    return f"PRICE HISTORY {ticker} (last {days}d):\n" + _df_text(df[cols].tail(days))


def get_company_profile(ticker: str) -> str:
    """Company profile and key stats."""
    obb = _get_obb()
    df = _safe(obb.equity.profile, ticker, provider="yfinance")
    if df.empty:
        return f"No profile data for {ticker}."
    return f"PROFILE ({ticker}):\n" + _df_text(df)


def get_company_news(ticker: str, limit: int = 8) -> str:
    """Latest news for a company."""
    obb = _get_obb()
    df = _safe(obb.news.company, ticker, limit=limit, provider="yfinance")
    if df.empty:
        return f"No news for {ticker}."
    cols = [c for c in ["date", "title", "source", "url"] if c in df.columns]
    return f"NEWS ({ticker}, latest {limit}):\n" + _df_text(df[cols], limit)


def get_major_indices() -> str:
    """Current levels of major global indices."""
    obb = _get_obb()
    symbols = "^GSPC,^IXIC,^DJI,^RUT,^FTSE,^N225,^HSI,^STOXX50E"
    df = _safe(obb.index.price.historical, symbols, provider="yfinance", limit=1)
    if df.empty:
        return "No index data available."
    if "date" not in df.columns:
        df = df.reset_index()
    cols = [c for c in ["symbol", "date", "close", "volume"] if c in df.columns]
    return "MAJOR INDICES (latest):\n" + _df_text(df[cols])


def get_fundamentals(ticker: str) -> str:
    """Key financial ratios and fundamentals."""
    obb = _get_obb()
    df = _safe(obb.equity.fundamental.ratios, ticker, provider="yfinance", limit=1)
    if df.empty:
        return f"No fundamental data for {ticker}."
    return f"FUNDAMENTALS ({ticker}):\n" + _df_text(df)


def get_earnings_calendar(limit: int = 10) -> str:
    """Upcoming earnings announcements."""
    obb = _get_obb()
    df = _safe(obb.equity.calendar.earnings, provider="yfinance")
    if df.empty:
        return "No earnings calendar data."
    cols = [c for c in ["symbol", "name", "report_date", "eps_estimate",
                         "revenue_estimate"] if c in df.columns]
    return "EARNINGS CALENDAR (upcoming):\n" + _df_text(df[cols], limit)


# ── Intent detection → tool dispatch ────────────────────────────────

_TOOL_MAP = [
    # (keywords, function, description)
    (["gainer", "top mover", "top gain", "best perform", "biggest gain"], get_gainers,
     "live top gainers"),
    (["loser", "worst perform", "biggest loss", "biggest drop", "biggest fall"], get_losers,
     "live top losers"),
    (["most active", "highest volume", "most traded"], get_most_active,
     "most actively traded stocks"),
    (["index", "indices", "s&p", "nasdaq", "dow", "nikkei", "ftse", "hang seng"], get_major_indices,
     "major global indices"),
    (["earnings calendar", "earnings upcoming", "earnings schedule", "report date"], get_earnings_calendar,
     "upcoming earnings"),
]

_TICKER_TOOLS = [
    (["news", "headline", "article"], get_company_news, "company news"),
    (["fundamental", "ratio", "pe ", "p/e", "revenue", "profit", "balance", "valuation"], get_fundamentals,
     "fundamentals"),
    (["price", "quote", "stock price", "current price", "trading at"], get_stock_quote,
     "live price quote"),
    (["history", "historical", "chart", "past performance", "last month", "last week"], get_stock_history,
     "price history"),
    (["profile", "about", "what is", "who is", "company info", "sector", "industry"], get_company_profile,
     "company profile"),
]


def fetch_realtime_context(message: str, market: str = None) -> str:
    """
    Detect what real-time data the message needs and fetch it via OpenBB.
    Returns a compact text block to prepend to LLM context, or empty string.
    """
    msg_lower = message.lower()
    sections = []

    # Check for general market tools
    for keywords, fn, desc in _TOOL_MAP:
        if any(kw in msg_lower for kw in keywords):
            try:
                sections.append(fn())
            except Exception:
                pass

    # Check for ticker-specific tools (look for uppercase ticker pattern)
    import re
    tickers = re.findall(r'\b([A-Z]{1,5})\b', message)
    # Filter out common non-ticker words
    _IGNORE = {"I", "A", "AN", "THE", "FOR", "IN", "ON", "AT", "TO", "OF", "IS",
                "AND", "OR", "BY", "AS", "IT", "BE", "DO", "IF", "NO", "UP", "SO",
                "MY", "ME", "US", "UK", "CN", "JP", "HK", "DE", "FR", "KR", "TW",
                "AU", "BR", "SA", "IN", "EU", "LLM", "API", "ETF", "IPO", "PE",
                "EPS", "CEO", "CFO", "AI", "ML", "USD", "EUR", "GBP", "JPY"}
    tickers = [t for t in tickers if t not in _IGNORE]

    if tickers:
        ticker_str = ",".join(tickers[:3])  # limit to 3 tickers
        for keywords, fn, desc in _TICKER_TOOLS:
            if any(kw in msg_lower for kw in keywords):
                try:
                    if fn == get_company_news or fn == get_stock_history or \
                       fn == get_fundamentals or fn == get_company_profile:
                        sections.append(fn(tickers[0]))
                    else:
                        sections.append(fn(ticker_str))
                except Exception:
                    pass
        # Always get quote for mentioned tickers
        if not any(kw in msg_lower for kw in ["price", "quote", "trading at"]):
            try:
                sections.append(get_stock_quote(ticker_str))
            except Exception:
                pass

    if not sections:
        return ""

    return "\n\n=== LIVE MARKET DATA (OpenBB) ===\n" + "\n\n".join(sections) + "\n=== END LIVE DATA ===\n"
