"""LLM-backed agent chat — each agent loads its data and uses an LLM to answer."""

import os
import json
from datetime import datetime

import pandas as pd


def _load_parquet(path, max_rows=200):
    """Load parquet, return truncated DataFrame."""
    try:
        df = pd.read_parquet(path)
        return df.head(max_rows)
    except Exception:
        return pd.DataFrame()


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _df_to_text(df, max_rows=50):
    """Convert DataFrame to compact CSV-style text for LLM context."""
    if df.empty:
        return "(no data)"
    df = df.head(max_rows)
    return df.to_csv(index=False, float_format="%.4f")


# ── Data loaders per agent type ──────────────────────────────────────

def _name_lookup(base):
    """Load ticker→name mapping from universe (full, not truncated)."""
    try:
        uni = pd.read_parquet(os.path.join(base, "universe.parquet"))
    except Exception:
        return pd.DataFrame(columns=["ticker", "name"])
    if not uni.empty and "name" in uni.columns:
        return uni[["ticker", "name"]].drop_duplicates("ticker")
    return pd.DataFrame(columns=["ticker", "name"])


def _merge_names(df, names):
    """Merge name column into df from universe lookup."""
    if names.empty or "name" in df.columns:
        return df
    return df.merge(names, on="ticker", how="left")


def _latest_per_ticker(df):
    """Keep only the latest date row per ticker."""
    if "date" in df.columns and "ticker" in df.columns:
        df = df.sort_values("date").drop_duplicates("ticker", keep="last")
    return df


def _load_data_context(market, data_dir):
    """Load all data relevant to the Data Agent."""
    base = os.path.join(data_dir, "markets", market)
    sections = []
    names = _name_lookup(base)

    # Daily market data FIRST — gainers/losers must not be truncated
    snap = _latest_daily_parquet(base)
    if not snap.empty:
        snap = _latest_per_ticker(snap)
        snap = _merge_names(snap, names)
        cols = [c for c in ["ticker", "name", "close", "return_1d", "return_5d",
                            "volume", "volume_ratio", "turnover"] if c in snap.columns]
        if "return_1d" in snap.columns:
            gainers = snap.nlargest(30, "return_1d")[cols]
            losers  = snap.nsmallest(30, "return_1d")[cols]
            avg = snap["return_1d"].mean()
            up  = (snap["return_1d"] > 0).sum()
            dn  = (snap["return_1d"] < 0).sum()
            sections.append(f"DAILY DATA ({len(snap)} stocks, avg {avg:+.2%}, {up} up / {dn} down)")
            sections.append(f"TOP GAINERS (30):\n" + _df_to_text(gainers))
            sections.append(f"TOP LOSERS (30):\n"  + _df_to_text(losers))
        else:
            sections.append(f"DAILY DATA ({len(snap)} stocks):\n" + _df_to_text(snap[cols], 50))

    # Universe (after price data so it doesn't push losers out of truncation window)
    if not names.empty:
        sections.append(f"UNIVERSE: {len(names)} stocks")

    # Market cap
    cap = _load_parquet(os.path.join(base, "market_cap.parquet"))
    if not cap.empty:
        cap = _merge_names(cap, names)
        cols = [c for c in ["ticker", "name", "market_cap", "market_cap_usd"] if c in cap.columns]
        if "market_cap" in cap.columns:
            cap = cap.sort_values("market_cap", ascending=False)
        sections.append(f"MARKET CAP ({len(cap)} stocks):\n" + _df_to_text(cap[cols], 30))

    # Indices
    idx = _load_parquet(os.path.join(base, "indices.parquet"))
    if not idx.empty:
        sections.append(f"INDICES:\n" + _df_to_text(idx, 20))

    return "\n\n".join(sections) if sections else "No data available yet. Run the pipeline first."


def _load_news_context(market, data_dir):
    """Load all data relevant to the News Agent."""
    base = os.path.join(data_dir, "markets", market)
    sections = []
    names = _name_lookup(base)

    news = _load_parquet(os.path.join(base, "news.parquet"), max_rows=100)
    if not news.empty:
        news = _merge_names(news, names)
        cols = [c for c in ["ticker", "name", "title", "publisher", "hit_count",
                            "keywords_matched", "link"] if c in news.columns]
        if "hit_count" in news.columns:
            news = news.sort_values("hit_count", ascending=False)
        sections.append(f"NEWS ({len(news)} articles):\n" + _df_to_text(news[cols], 60))

    if not names.empty:
        sections.append("Ticker/Name lookup:\n" + _df_to_text(names.head(100), 100))

    return "\n\n".join(sections) if sections else "No news data yet. Run the pipeline first."


def _load_signal_context(market, data_dir):
    """Load all data relevant to the Signal Agent (data + news + alerts)."""
    base = os.path.join(data_dir, "markets", market)
    sections = []
    names = _name_lookup(base)

    # Alerts
    alerts = _load_json(os.path.join(base, "alerts.json"))
    if alerts and isinstance(alerts, list):
        sections.append(f"ALERTS ({len(alerts)} active):\n" + json.dumps(alerts[:30], indent=1, default=str))
    else:
        sections.append("ALERTS: None active")

    # Daily data summary — most recent file
    snap = _latest_daily_parquet(base)
    if not snap.empty and "return_1d" in snap.columns:
        snap = _latest_per_ticker(snap)
        snap = _merge_names(snap, names)
        cols = [c for c in ["ticker", "name", "close", "return_1d", "volume_ratio"] if c in snap.columns]
        top = snap.nlargest(10, "return_1d")[cols]
        bot = snap.nsmallest(10, "return_1d")[cols]
        avg = snap["return_1d"].mean()
        up = (snap["return_1d"] > 0).sum()
        down = (snap["return_1d"] < 0).sum()
        sections.append(f"MARKET SUMMARY: {len(snap)} stocks, avg {avg:+.2%}, {up} up / {down} down")
        sections.append(f"TOP GAINERS:\n" + _df_to_text(top))
        sections.append(f"TOP LOSERS:\n" + _df_to_text(bot))

    # News summary
    news = _load_parquet(os.path.join(base, "news.parquet"), max_rows=50)
    if not news.empty:
        news = _merge_names(news, names)
        if "hit_count" in news.columns:
            hot = news[news["hit_count"] > 0].sort_values("hit_count", ascending=False).head(10)
        else:
            hot = news.head(10)
        cols = [c for c in ["ticker", "name", "title", "hit_count", "keywords_matched"] if c in hot.columns]
        sections.append(f"NEWS HIGHLIGHTS ({len(news)} articles, showing top keyword matches):\n" + _df_to_text(hot[cols], 10))

    # Capital flow
    flow = _load_parquet(os.path.join(base, "capital_flow.parquet"))
    if not flow.empty:
        sections.append(f"CAPITAL FLOW:\n" + _df_to_text(flow.tail(5)))

    if not names.empty:
        sections.append("Ticker/Name lookup:\n" + _df_to_text(names.head(100), 100))

    return "\n\n".join(sections) if sections else "No data yet. Run the pipeline first."


def _latest_daily_parquet(base):
    """Return one row per ticker (most recent date) with computed return signals.

    Reads the FULL multi-day parquet so pct_change() has historical context,
    then keeps only the latest row per ticker. Computes return_1d/5d/20d and
    volume_ratio on-the-fly if the pipeline didn't save them.
    """
    import glob
    files = sorted(glob.glob(os.path.join(base, "market_daily_*.parquet")))
    if not files:
        return pd.DataFrame()
    try:
        df = pd.read_parquet(files[-1])
        if df.empty:
            return df

        # Compute signals if missing (old pipeline saves raw OHLCV only)
        if "return_1d" not in df.columns:
            parts = []
            for ticker, grp in df.groupby("ticker"):
                g = grp.sort_values("date").copy()
                g["return_1d"]  = g["close"].pct_change(1)
                g["return_5d"]  = g["close"].pct_change(5)
                g["return_20d"] = g["close"].pct_change(20)
                vol_avg = g["volume"].rolling(20, min_periods=3).mean()
                g["volume_ratio"] = g["volume"] / vol_avg
                parts.append(g)
            df = pd.concat(parts, ignore_index=True)

        return _latest_per_ticker(df)
    except Exception:
        return pd.DataFrame()


def _load_market_context(market, data_dir):
    """Load unified context combining data+news+signal for a market agent."""
    base = os.path.join(data_dir, "markets", market)
    sections = []
    names = _name_lookup(base)
    referenced_tickers = set()

    # Daily market data — use most recent file regardless of date
    snap = _latest_daily_parquet(base)
    if not snap.empty:
        snap = _latest_per_ticker(snap)
        snap = _merge_names(snap, names)
        cols = [c for c in ["ticker", "name", "close", "return_1d", "return_5d",
                            "volume", "volume_ratio", "turnover"] if c in snap.columns]
        if "return_1d" in snap.columns:
            snap = snap.sort_values("return_1d", ascending=False)
        sections.append(f"DAILY DATA ({len(snap)} stocks):\n" + _df_to_text(snap[cols], 40))
        if "ticker" in snap.columns:
            referenced_tickers.update(snap["ticker"].tolist())

    # Market cap
    cap = _load_parquet(os.path.join(base, "market_cap.parquet"))
    if not cap.empty:
        cap = _merge_names(cap, names)
        cols = [c for c in ["ticker", "name", "market_cap", "market_cap_usd"] if c in cap.columns]
        if "market_cap" in cap.columns:
            cap = cap.sort_values("market_cap", ascending=False)
        sections.append(f"MARKET CAP (top 20):\n" + _df_to_text(cap[cols], 20))
        if "ticker" in cap.columns:
            referenced_tickers.update(cap.head(20)["ticker"].tolist())

    # Indices
    idx = _load_parquet(os.path.join(base, "indices.parquet"))
    if not idx.empty:
        sections.append(f"INDICES:\n" + _df_to_text(idx, 15))

    # News
    news = _load_parquet(os.path.join(base, "news.parquet"), max_rows=80)
    if not news.empty:
        news = _merge_names(news, names)
        if "hit_count" in news.columns:
            news = news.sort_values("hit_count", ascending=False)
        cols = [c for c in ["ticker", "name", "title", "publisher", "hit_count",
                            "keywords_matched"] if c in news.columns]
        sections.append(f"NEWS ({len(news)} articles, top by keyword hits):\n" + _df_to_text(news[cols], 25))
        if "ticker" in news.columns:
            referenced_tickers.update(news.head(25)["ticker"].tolist())

    # Alerts
    alerts = _load_json(os.path.join(base, "alerts.json"))
    if alerts and isinstance(alerts, list):
        sections.append(f"ALERTS ({len(alerts)} active):\n" + json.dumps(alerts[:20], indent=1, default=str))
        for a in alerts[:20]:
            if isinstance(a, dict) and "ticker" in a:
                referenced_tickers.add(a["ticker"])
    else:
        sections.append("ALERTS: None active")

    # Capital flow
    flow = _load_parquet(os.path.join(base, "capital_flow.parquet"))
    if not flow.empty:
        sections.append(f"CAPITAL FLOW:\n" + _df_to_text(flow.tail(5)))

    # Complete ticker→name map for ALL referenced tickers
    if not names.empty and referenced_tickers:
        name_map = names[names["ticker"].isin(referenced_tickers)]
        if not name_map.empty:
            sections.append(f"TICKER→NAME MAP ({len(name_map)} entries):\n" + _df_to_text(name_map, 200))

    return "\n\n".join(sections) if sections else "No data available yet. Run the pipeline first."


def _load_global_context(data_dir):
    """Load all data relevant to the Global Strategist."""
    gdir = os.path.join(data_dir, "global")
    sections = []

    # Macro
    macro = _load_json(os.path.join(gdir, "macro_latest.json"))
    if macro:
        sections.append("MACRO INDICATORS:\n" + json.dumps(macro, indent=1, default=str))

    # Sector performance
    sec = _load_parquet(os.path.join(gdir, "sector_performance.parquet"))
    if not sec.empty:
        sections.append(f"SECTOR PERFORMANCE ({len(sec)} entries):\n" + _df_to_text(sec, 40))

    # Correlations
    corr = _load_json(os.path.join(gdir, "correlations.json"))
    if corr:
        sections.append("CORRELATIONS:\n" + json.dumps(corr, indent=1, default=str)[:2000])

    # Global alerts
    alerts = _load_json(os.path.join(gdir, "alerts.json"))
    if alerts and isinstance(alerts, list):
        sections.append(f"GLOBAL ALERTS ({len(alerts)}):\n" + json.dumps(alerts[:20], indent=1, default=str))

    # Peer groups
    peers = _load_json(os.path.join(gdir, "peer_groups.json"))
    if peers and isinstance(peers, list):
        sections.append(f"PEER GROUPS ({len(peers)} groups):\n" + json.dumps(peers[:10], indent=1, default=str)[:2000])

    # Geopolitical
    geo = _load_json(os.path.join(gdir, "geopolitical_context.json"))
    if geo:
        items = geo if isinstance(geo, list) else geo.get("items", [])
        if items:
            sections.append(f"GEOPOLITICAL NEWS ({len(items)} items):\n" + json.dumps(items[:15], indent=1, default=str)[:2000])

    return "\n\n".join(sections) if sections else "No global data yet. Run the pipeline first."


# ── Tool registry ────────────────────────────────────────────────────

TOOLS = {
    "local_data":         "Local OHLCV prices, returns (1d/5d/20d), volume ratios, market cap, indices from most recent pipeline run",
    "local_news":         "Local scraped news articles with keyword relevance scores",
    "local_signals":      "Local alerts: volume spikes, capital flow anomalies, unusual activity signals",
    "openbb_gainers":     "Live top gaining stocks today — US STOCKS ONLY, do NOT use for CN/HK/JP/KR/TW/IN/UK/DE/FR/AU/BR/SA markets",
    "openbb_losers":      "Live top losing stocks today — US STOCKS ONLY, do NOT use for non-US markets",
    "openbb_active":      "Live most actively traded stocks — US STOCKS ONLY, do NOT use for non-US markets",
    "openbb_indices":     "Live major global indices (S&P 500, Nasdaq, Nikkei, FTSE, Hang Seng, etc.)",
    "openbb_quote":       "Live real-time price quote for specific tickers [needs tickers]",
    "openbb_news":        "Latest news articles for a specific company [needs ticker]",
    "openbb_history":     "30-day price history for a specific stock [needs ticker]",
    "openbb_profile":     "Company profile: sector, industry, description, key stats [needs ticker]",
    "openbb_fundamentals":"Financial ratios: P/E, EPS, revenue, profit margins [needs ticker]",
    "trading_agent":      "Deep multi-agent analysis: fundamentals + sentiment + technicals + bull/bear debate (slow, 30-60s). Supports US stocks (e.g. NVDA) AND CN A-shares (6-digit codes e.g. 688031). Use ONLY for explicit 'analyze TICKER' requests [needs ticker]",
    "sector_agent":       "Multi-agent sector analysis: performance overview, sentiment, bull/bear debate, outlook for a whole sector (e.g. 'tech sector', '半导体', 'finance'). Use when user asks to analyze a sector, industry, or market segment [needs sector name]",
    "web_search":         "Search the internet for real-time news, recent events, or any information not available in local/OpenBB data. Use when other tools lack the needed information.",
}

_ROUTER_IGNORE = {
    "I", "A", "AN", "THE", "FOR", "IN", "ON", "AT", "TO", "OF", "IS",
    "AND", "OR", "BY", "AS", "IT", "BE", "DO", "IF", "NO", "UP", "SO",
    "MY", "ME", "US", "UK", "CN", "JP", "HK", "DE", "FR", "KR", "TW",
    "AU", "BR", "SA", "EU", "LLM", "API", "ETF", "IPO", "AI", "ML",
    "USD", "EUR", "GBP", "JPY", "CNY", "HKD", "KRW", "TWD", "SGD",
    "PE", "EPS", "CEO", "CFO", "CTO", "IPO", "YOY", "QOQ", "WTI",
}


def _route_message(message, market, agent_type, provider, api_key):
    """Fast LLM router: returns {"tools": [...], "tickers": [...]}."""
    import re as _re

    tools_list = "\n".join(f"- {k}: {v}" for k, v in TOOLS.items())
    router_system = "You are a routing agent. Output ONLY valid JSON, no explanation, no markdown."
    router_user = f"""Market: {market or "global"}  Agent: {agent_type}
Question: {message}

Tools:
{tools_list}

Return: {{"tools": ["tool1", "tool2"], "tickers": ["TICKER"], "company_name": ""}}

- tickers: ONLY ticker codes that the user explicitly typed (e.g. "NVDA", "688031", "9988"). Do NOT infer or convert company names to tickers — if user typed "阿里巴巴" or "Tesla", do NOT put "BABA" or "TSLA" in tickers.
- company_name: if user mentions a company by name rather than by ticker code (e.g. "阿里巴巴", "Tesla", "腾讯"), put the name here and leave tickers empty.

Rules:
- Pick 1-4 tools. local tools are fast (cached), OpenBB tools fetch live data.
- Extract stock tickers: 1-5 uppercase letters for US (e.g. NVDA, AAPL) OR 6-digit numbers for CN A-shares (e.g. 688031, 000001) OR 4-5 digits for HK (e.g. 9988, 0700).
- Use trading_agent if user asks to analyze a specific stock or company, whether by ticker (e.g. "analyze NVDA", "分析9988") or by company name (e.g. "帮我分析一下阿里巴巴", "analyze Tesla", "分析腾讯"). Works for all markets.
- Use sector_agent if user mentions a sector/industry/subsector (e.g. "analyze tech sector", "分析半导体板块", "semiconductor industry outlook", "software application companies", "医疗器械板块"). Extract the sector or subsector name as a ticker.
- CRITICAL: openbb_gainers/losers/active return US stocks ONLY. NEVER use for CN/HK/JP/KR/TW/IN/UK/DE/FR/AU/BR/SA markets.
- Non-US market movers (gainers/losers/active) → local_data + openbb_indices (has return_1d for ranking, plus live index).
- US market movers → local_data + openbb_gainers + openbb_losers + openbb_indices.
- Global overview → local_data + openbb_indices.
- News → local_news + openbb_news if ticker present.
- Specific stock → openbb_quote + local_data.
- Indices/macro → openbb_indices.
- If user mentions "internet", "search", "browse", "web", "online", "latest news", ALWAYS include web_search.
- For "top gainers news" / "news about movers" with no tickers → use local_data + web_search (web_search will auto-find the tickers).
- FALLBACK: If the question is about current events, general finance topics, economic data, company news, or ANY topic where local pipeline data would be insufficient or out-of-date, ALWAYS add web_search.
- If needed data is unavailable in local/OpenBB tools, add web_search to find it online.
- When in doubt about whether local data covers the question, include web_search as a safety net."""

    try:
        if provider == "anthropic":
            raw = _call_anthropic(api_key, router_system, [], router_user, max_tokens=150)
        else:
            raw = _call_openai(api_key, router_system, [], router_user, max_tokens=150)
        match = _re.search(r'\{.*?\}', raw, _re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            tools = [t for t in parsed.get("tools", []) if t in TOOLS]
            tickers = [t for t in parsed.get("tickers", []) if isinstance(t, str)]
            company_name = parsed.get("company_name", "").strip()
            if tools:
                return {"tools": tools, "tickers": tickers[:4], "company_name": company_name}
    except Exception:
        pass

    # Fallback: rule-based routing
    msg_lower = message.lower()
    # Extract US tickers (1-5 uppercase letters) and CN tickers (6-digit codes)
    us_tickers = [t for t in _re.findall(r'\b([A-Z]{1,5})\b', message)
                  if t not in _ROUTER_IGNORE]
    cn_tickers = _re.findall(r'\b(\d{6})\b', message)
    tickers = (cn_tickers + us_tickers)[:4]

    # Sector analysis detection
    _SECTOR_KEYWORDS = ["sector", "industry", "板块", "行业", "赛道"]
    is_sector_request = (
        any(k in msg_lower for k in _SECTOR_KEYWORDS)
    ) and ("analyze" in msg_lower or "分析" in message or "outlook" in msg_lower or
           "how is" in msg_lower or "怎么样" in message)

    if is_sector_request:
        # Extract sector name by stripping trigger words from the message
        import re as _re3
        # Remove intent words, keep the sector name
        clean = message
        for w in ["分析", "analyze", "analysis", "outlook", "how is", "怎么样", "tell me about"]:
            clean = clean.replace(w, " ")
        for w in _SECTOR_KEYWORDS:
            clean = clean.replace(w, " ")
        sector = clean.strip().strip("?？ \t\n") or "tech"
        return {"tools": ["sector_agent"], "tickers": [sector]}

    _ANALYZE_KEYWORDS = ["analyze", "analysis", "分析", "帮我看看", "帮我分析", "研究一下"]
    is_analyze = any(k in message for k in _ANALYZE_KEYWORDS)
    if is_analyze and tickers:
        return {"tools": ["trading_agent"], "tickers": tickers}
    # Analyze by company name (no ticker in message) — resolve via alias map later
    if is_analyze and not tickers:
        # Extract company name: remove intent words and punctuation
        clean = message
        for w in _ANALYZE_KEYWORDS + ["一下", "帮我", "请", "?", "？"]:
            clean = clean.replace(w, " ")
        company_name = clean.strip()
        if company_name:
            return {"tools": ["trading_agent"], "tickers": [], "company_name": company_name}

    tools = []
    if agent_type == "global" or market in (None, "ALL"):
        tools = ["local_data", "openbb_indices"]
    else:
        tools = ["local_data", "local_signals", "openbb_indices"]
    if any(w in msg_lower for w in ["news", "headline", "article"]):
        tools.append("local_news")
        if tickers:
            tools.append("openbb_news")
    is_us = market in ("US", None, "ALL")
    if any(w in msg_lower for w in ["gainer", "top gain", "best perform", "biggest gain"]):
        tools.append("openbb_gainers") if is_us else None
    if any(w in msg_lower for w in ["loser", "worst", "biggest drop", "biggest fall"]):
        tools.append("openbb_losers") if is_us else None
    if any(w in msg_lower for w in ["active", "most traded", "high volume"]):
        tools.append("openbb_active") if is_us else None
    if tickers:
        tools.append("openbb_quote")
    if any(w in msg_lower for w in ["internet", "search", "browse", "web", "online",
                                     "google", "latest", "recent news"]):
        tools.append("web_search")

    return {"tools": list(dict.fromkeys(tools)), "tickers": tickers}


def _web_search(query: str, max_results: int = 6) -> str:
    """Search the web via DuckDuckGo. Returns compact text for LLM context."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = r.get("title", "")
                body  = r.get("body", "")[:200]
                href  = r.get("href", "")
                results.append(f"• {title}\n  {body}\n  {href}")
        if not results:
            return "(no web results)"
        return f"WEB SEARCH: {query!r}\n" + "\n\n".join(results)
    except Exception as e:
        return f"(web search failed: {e})"


def _extract_sources(web_result: str) -> str:
    """Extract source URLs and titles from _web_search output. Returns a markdown footer."""
    sources = []
    current_title = ""
    for line in web_result.split("\n"):
        stripped = line.strip()
        if stripped.startswith("•"):
            current_title = stripped[1:].strip()
        elif stripped.startswith("http"):
            sources.append((current_title[:70] if current_title else stripped[:70], stripped))
            current_title = ""
    if not sources:
        return ""
    parts = [f"- [{title}]({url})" if title else f"- {url}" for title, url in sources[:6]]
    return "\n\n---\n**🔍 Sources:**\n" + "\n".join(parts)


def _execute_tools(tools, tickers, market, data_dir, message=""):
    """Execute selected tools in parallel. Returns dict of tool_name → result text."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    is_global = market in (None, "ALL", "global")

    def _run(name):
        try:
            # Local data tools
            if name == "local_data":
                return name, _load_global_context(data_dir) if is_global else _load_data_context(market, data_dir)
            if name == "local_news":
                return name, "(no per-market news for global agent)" if is_global else _load_news_context(market, data_dir)
            if name == "local_signals":
                return name, _load_global_context(data_dir) if is_global else _load_signal_context(market, data_dir)

            # Web search
            if name == "web_search":
                search_tickers = list(tickers)
                # If no tickers specified but user asks about movers, auto-load top gainers
                if not search_tickers and any(w in message.lower() for w in
                        ["gainer", "mover", "top gain", "best perform", "surge", "jump"]):
                    try:
                        import glob as _glob, os as _os, pandas as _pd
                        rows = []
                        for p in _glob.glob(_os.path.join(data_dir, "markets", "*", "market_daily_*.parquet")):
                            df = _pd.read_parquet(p)
                            if not df.empty:
                                rows.append(df.sort_values("date").groupby("ticker").tail(1))
                        if rows:
                            combined = _pd.concat(rows).dropna(subset=["return_1d"])
                            if market and market not in (None, "ALL", "global"):
                                combined = combined[combined.get("market", combined.get("ticker", combined)) == market] if "market" in combined.columns else combined
                            top = combined.nlargest(5, "return_1d")["ticker"].tolist()
                            search_tickers = top
                    except Exception:
                        pass
                if search_tickers:
                    results = []
                    for t in search_tickers[:5]:
                        r = _web_search(f"{t} stock news today", max_results=3)
                        results.append(r)
                    return name, "\n\n".join(results)
                else:
                    query = f"{market or ''} {message}".strip()
                    return name, _web_search(query)

            # OpenBB live tools
            try:
                from web.openbb_tools import (get_gainers, get_losers, get_most_active,
                                               get_major_indices, get_stock_quote,
                                               get_company_news, get_stock_history,
                                               get_company_profile, get_fundamentals)
            except ImportError:
                from openbb_tools import (get_gainers, get_losers, get_most_active,
                                          get_major_indices, get_stock_quote,
                                          get_company_news, get_stock_history,
                                          get_company_profile, get_fundamentals)

            if name == "openbb_gainers":  return name, get_gainers()
            if name == "openbb_losers":   return name, get_losers()
            if name == "openbb_active":   return name, get_most_active()
            if name == "openbb_indices":  return name, get_major_indices()
            if name == "openbb_quote" and tickers:
                return name, get_stock_quote(",".join(tickers[:3]))
            if name == "openbb_news" and tickers:
                return name, get_company_news(tickers[0])
            if name == "openbb_history" and tickers:
                return name, get_stock_history(tickers[0])
            if name == "openbb_profile" and tickers:
                return name, get_company_profile(tickers[0])
            if name == "openbb_fundamentals" and tickers:
                return name, get_fundamentals(tickers[0])

            return name, "(skipped — no tickers provided)"
        except Exception as e:
            return name, f"(tool error: {e})"

    results = {}
    non_ta = [t for t in tools if t != "trading_agent"]
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(_run, t): t for t in non_ta}
        for future in futures:
            name, result = future.result()
            results[name] = result
    return results


# ── System prompts per agent type ────────────────────────────────────

_BEGINNER_STYLE = """
## Your audience
You are talking to stock market beginners who may not know financial jargon.
Your responses must be educational, clear, and easy to understand.

## How to write responses
1. **Walk through your reasoning step by step** — don't just state conclusions, show how you got there.
   Example: "First, I look at the price change: it went up +8%. That's a big single-day move.
   Next, I check WHY — the news shows an earnings beat. This explains the jump."

2. **Always explain financial terms** the first time you use them.
   Example: "volume ratio (this measures how much more than usual people are trading this stock —
   a ratio of 3× means 3 times the normal daily volume, which signals unusual interest)"

3. **Use simple analogies** to make numbers meaningful.
   Example: don't just say "market cap ¥50B" — say "market cap ¥50B (roughly the size of a mid-sized regional bank)"

4. **Explain what signals mean in plain language.**
   - High volume → "more people than usual are buying/selling this stock today"
   - Price gap up → "the stock opened much higher than yesterday's close — usually triggered by overnight news"
   - RSI > 70 → "the stock has risen very fast recently and may be 'overbought' — meaning it might slow down soon"
   - Capital inflow → "big institutional investors are putting money into this stock"

5. **End with a plain-language takeaway** — what should a beginner actually understand from this?

6. **Format clearly** with headers, bullet points, and short paragraphs. Never write a wall of text."""


SYSTEM_PROMPTS = {
    "data": """You are the {market} Market Data Guide — a friendly teacher helping stock market beginners understand market data.
You have access to the stock universe, daily prices, market cap, and index data for {market}.

{beginner_style}

When showing stock data, always include: ticker code + full company name + what the number means.""".replace("{beginner_style}", _BEGINNER_STYLE),

    "news": """You are the {market} Market News Guide — a friendly teacher helping stock market beginners understand financial news.
You have access to company news articles for {market}.

{beginner_style}

When explaining news, always cover: what happened → why it matters → what effect it could have on the stock price.""".replace("{beginner_style}", _BEGINNER_STYLE),

    "signal": """You are the {market} Market Signal Guide — a friendly teacher helping stock market beginners understand market signals and alerts.
You synthesize prices, news, and alerts to explain what's happening in the market.

{beginner_style}

When explaining signals, always follow this structure:
1. **What I see** (the raw data/alert)
2. **What it means** (explanation in plain language)
3. **Why it might be happening** (connect to news or broader context)
4. **What to watch** (what a beginner should pay attention to next)""".replace("{beginner_style}", _BEGINNER_STYLE),

    "market": """You are the {market} Market Guide — a knowledgeable but approachable teacher helping stock market beginners understand what's happening in the {market} market.

You have comprehensive data: stock prices, news, market signals, alerts, capital flow, live real-time data from OpenBB.
Prefer LIVE MARKET DATA (labeled "LIVE MARKET DATA") over local data when both are available — it's more current.

{beginner_style}

## Response structure for market questions
For questions like "what's happening" or "top movers", use this structure:
### 📊 Market Overview
(1-2 sentences on the overall market mood today — is it up or down, calm or volatile?)

### 🚀 Notable Movers
(List 3-5 stocks with: name, price change, AND a plain-language explanation of why)

### 📰 What the News Says
(Key news stories and what they mean for investors)

### 🔍 Signals Worth Watching
(Any unusual patterns — explain each one in plain language)

### 💡 Key Takeaway
(1-2 sentences: what should a beginner take away from today's market?)""".replace("{beginner_style}", _BEGINNER_STYLE),

    "global": """You are the Global Market Guide — a knowledgeable but approachable teacher helping stock market beginners understand global financial markets.

You cover 13 markets worldwide. You have access to macro indicators, sector data, cross-market correlations, and live real-time data from OpenBB (indices, top movers, company data).
Prefer LIVE MARKET DATA over local data when both are available.

{beginner_style}

## Response structure for global questions
### 🌍 Global Snapshot
(What's the overall mood across world markets today?)

### 📈 Index Movements
(Key indices with plain-language context — e.g., "S&P 500 is up 0.5%, meaning most large US companies gained today")

### 🔗 Cross-Market Story
(Is there a theme connecting markets? e.g., "Tech sold off globally after US inflation data surprised")

### 💡 What This Means for a Beginner
(Plain-language summary of the most important thing to understand)""".replace("{beginner_style}", _BEGINNER_STYLE),
}


# ── Main LLM chat function ──────────────────────────────────────────

def _load_llm_config():
    """Load LLM config from data/llm_config.json."""
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "llm_config.json")
        with open(config_path) as f:
            return json.load(f)
    except Exception:
        return {}

def get_llm_provider():
    """Returns ("openai", api_key) using config file, or (None, None) if not configured."""
    cfg = _load_llm_config()
    key = cfg.get("api_key", "").strip()
    if key:
        return "openai", key
    return None, None


def _resolve_cn_name_to_ticker(name: str, data_dir: str) -> str | None:
    """Look up a company name in the CN A-share universe and return its 6-digit ticker."""
    import re as _re
    if _re.match(r'^\d{6}$', name.strip()):
        return name.strip()
    try:
        uni_path = os.path.join(data_dir, "markets", "CN", "universe.parquet")
        uni = pd.read_parquet(uni_path)
        if "name" in uni.columns and "ticker" in uni.columns:
            exact = uni[uni["name"] == name]
            if not exact.empty:
                return str(exact.iloc[0]["ticker"])
            partial = uni[uni["name"].str.contains(name, na=False, regex=False)]
            if not partial.empty:
                return str(partial.iloc[0]["ticker"])
    except Exception:
        pass
    return None


# Known Chinese name → (ticker, market) for well-known stocks whose yfinance name is English-only.
# Used as a last-resort fallback when name lookup against the stored English names fails.
_CN_NAME_ALIASES: dict[str, tuple[str, str]] = {
    # HK
    "阿里巴巴": ("9988", "HK"), "阿里": ("9988", "HK"),
    "腾讯": ("700", "HK"),
    "美团": ("3690", "HK"),
    "京东": ("9618", "HK"),
    "百度": ("9888", "HK"),
    "网易": ("9999", "HK"),
    "快手": ("1024", "HK"),
    "哔哩哔哩": ("9626", "HK"), "b站": ("9626", "HK"),
    "携程": ("9961", "HK"),
    "小米": ("1810", "HK"),
    "联想": ("992", "HK"),
    "中芯国际": ("981", "HK"),
    "华虹": ("1347", "HK"),
    "比亚迪": ("1211", "HK"),
    "蔚来": ("9866", "HK"),
    "小鹏": ("9868", "HK"), "小鹏汽车": ("9868", "HK"),
    "理想": ("2015", "HK"), "理想汽车": ("2015", "HK"),
    "金蝶": ("268", "HK"),
    "中兴": ("763", "HK"), "中兴通讯": ("763", "HK"),
    "舜宇": ("2382", "HK"), "舜宇光学": ("2382", "HK"),
    "瑞声": ("2018", "HK"), "瑞声科技": ("2018", "HK"),
    "吉利": ("175", "HK"), "吉利汽车": ("175", "HK"),
    "港交所": ("388", "HK"),
    "阿里健康": ("241", "HK"),
    "京东健康": ("6618", "HK"),
    "商汤": ("20", "HK"), "商汤科技": ("20", "HK"),
    "药明生物": ("2269", "HK"),
    "平安好医生": ("1833", "HK"),
    "金山软件": ("3888", "HK"),
    "联发科": ("2454", "TW"),
    # US
    "英伟达": ("NVDA", "US"), "英达": ("NVDA", "US"),
    "苹果": ("AAPL", "US"),
    "微软": ("MSFT", "US"),
    "谷歌": ("GOOGL", "US"),
    "亚马逊": ("AMZN", "US"),
    "特斯拉": ("TSLA", "US"),
    "脸书": ("META", "US"), "元宇宙": ("META", "US"),
    "奈飞": ("NFLX", "US"),
}


def _find_company_in_market(name: str, market: str, data_dir: str) -> str | None:
    """Look up a company name in a single market's universe. Returns ticker or None."""
    try:
        uni_path = os.path.join(data_dir, "markets", market, "universe.parquet")
        uni = pd.read_parquet(uni_path)
        if "name" not in uni.columns or "ticker" not in uni.columns:
            return None
        exact = uni[uni["name"] == name]
        if not exact.empty:
            return str(exact.iloc[0]["ticker"])
        partial = uni[uni["name"].str.contains(name, na=False, regex=False)]
        if not partial.empty:
            return str(partial.iloc[0]["ticker"])
    except Exception:
        pass
    # Fallback: check known Chinese name aliases for this market
    alias = _CN_NAME_ALIASES.get(name)
    if alias and alias[1] == market:
        return alias[0]
    return None


def _find_company_in_markets(name: str, data_dir: str) -> tuple[str, str] | tuple[None, None]:
    """Search all market universes for a company name. Returns (ticker, market) or (None, None)."""
    _MARKET_LABELS = {
        "CN": "A股", "HK": "港股", "US": "美股", "JP": "日股",
        "KR": "韩股", "TW": "台股", "IN": "印度", "UK": "英股",
        "DE": "德股", "FR": "法股", "AU": "澳股", "BR": "巴西", "SA": "沙特",
    }
    for market in _MARKET_LABELS:
        try:
            uni_path = os.path.join(data_dir, "markets", market, "universe.parquet")
            uni = pd.read_parquet(uni_path)
            if "name" not in uni.columns or "ticker" not in uni.columns:
                continue
            exact = uni[uni["name"] == name]
            if not exact.empty:
                return str(exact.iloc[0]["ticker"]), market
            partial = uni[uni["name"].str.contains(name, na=False, regex=False)]
            if not partial.empty:
                return str(partial.iloc[0]["ticker"]), market
        except Exception:
            continue
    # Fallback: check known Chinese name aliases across all markets
    alias = _CN_NAME_ALIASES.get(name)
    if alias:
        return alias[0], alias[1]
    return None, None


def _web_to_local_enrichment(web_result: str, user_question: str,
                              data_dir: str, market: str,
                              provider: str, api_key: str) -> str:
    """
    Two-step post-web-search local enrichment:

    Step 1 — LLM planner: given web results + user question, decide what local
              data to fetch (which tickers, which sectors, or skip entirely).
    Step 2 — Execute: load local pipeline data for the planned queries.
    Fallback — regex ticker extraction if the LLM planner fails.

    Returns a markdown block of local data, or "" if nothing useful found.
    """
    # ── Step 1: LLM planner ──────────────────────────────────────────────────
    planner_system = (
        "You are a data planning agent. Output ONLY valid JSON, no explanation, no markdown."
    )
    planner_user = (
        f"User question: {user_question}\n\n"
        f"Web search results (excerpt):\n{web_result[:1500]}\n\n"
        "Based on the web results and the user's question, decide what local stock data to fetch.\n"
        "Return JSON:\n"
        '{"tickers": ["6-digit CN codes or US uppercase symbols mentioned that are relevant"], '
        '"sectors": ["sector or subsector name if user asked about a sector/industry"], '
        '"skip": false}\n'
        'Set "skip": true if web results are self-sufficient and no local data is needed.\n'
        "Only include tickers/sectors that are directly relevant to the user's question."
    )
    plan = {}
    try:
        if provider == "anthropic":
            raw = _call_anthropic(api_key, planner_system, [], planner_user, max_tokens=120)
        else:
            raw = _call_openai(api_key, planner_system, [], planner_user, max_tokens=120)
        import re as _re2
        m = _re2.search(r'\{.*\}', raw, _re2.DOTALL)
        if m:
            plan = json.loads(m.group())
    except Exception:
        pass  # fall through to regex fallback

    if plan.get("skip"):
        return ""

    planned_tickers = [str(t).strip() for t in plan.get("tickers", []) if t]
    planned_sectors  = [str(s).strip() for s in plan.get("sectors",  []) if s]

    # ── Step 2: Execute local queries ────────────────────────────────────────
    parts = []

    # 2a. Ticker lookup from planner
    if planned_tickers:
        try:
            try:
                from web.local_trading_agent import _load_local_price, _load_local_universe, detect_market
            except ImportError:
                from local_trading_agent import _load_local_price, _load_local_universe, detect_market

            ticker_lines = ["**📊 Local pipeline data for relevant stocks:**"]
            for ticker in planned_tickers[:10]:
                mkt = market if market and market not in (None, "ALL", "global") else detect_market(ticker)
                if mkt in ("UNKNOWN", "US") and not market:
                    mkt = "US"
                price = _load_local_price(ticker, mkt, data_dir)
                info  = _load_local_universe(ticker, mkt, data_dir)
                if not price:
                    # try other markets if hinted market missed
                    for alt in ["CN", "US", "HK", "JP", "KR", "TW"]:
                        if alt != mkt:
                            price = _load_local_price(ticker, alt, data_dir)
                            info  = _load_local_universe(ticker, alt, data_dir)
                            if price:
                                mkt = alt
                                break
                if not price:
                    continue
                name     = info.get("name", ticker)
                subsect  = info.get("subsector") or info.get("industry") or info.get("sector", "")
                close    = f"{price['close']:.2f}" if price.get("close") else "?"
                r1d      = f"{price['return_1d']:+.1%}" if price.get("return_1d") is not None else "?"
                r5d      = f"{price['return_5d']:+.1%}" if price.get("return_5d") is not None else "?"
                vr       = f"{price['volume_ratio']:.1f}x" if price.get("volume_ratio") is not None else "?"
                sub_str  = f" [{subsect}]" if subsect else ""
                ticker_lines.append(
                    f"- **{ticker}** {name}{sub_str} ({mkt}): "
                    f"close={close}, 1d={r1d}, 5d={r5d}, vol={vr}"
                )
            if len(ticker_lines) > 1:
                parts.append("\n".join(ticker_lines))
        except Exception:
            pass

    # 2b. Sector lookup from planner
    if planned_sectors:
        try:
            try:
                from web.local_trading_agent import _gather_sector_data
            except ImportError:
                from local_trading_agent import _gather_sector_data

            hint_markets = [market] if market and market not in (None, "ALL", "global") else \
                           ["CN", "US", "JP", "KR", "TW", "HK"]
            for sector in planned_sectors[:2]:
                sector_lines = []
                for m_code in hint_markets:
                    sd = _gather_sector_data(sector, m_code, data_dir)
                    if "error" in sd or sd.get("stock_count", 0) == 0:
                        continue
                    s = sd.get("summary", {})
                    avg1 = s.get("avg_return_1d")
                    line = (f"**{sector} — {m_code}** ({sd['stock_count']} stocks"
                            f" via {sd.get('match_source','?')})")
                    if avg1 is not None:
                        line += f": avg 1d={avg1:+.2%}, {s.get('up_count',0)}↑ {s.get('down_count',0)}↓"
                    sector_lines.append(line)
                    for r in sd.get("top_gainers", [])[:3]:
                        if r.get("return_1d") is not None:
                            sector_lines.append(
                                f"  ↑ {r.get('name', r['ticker'])} ({r['ticker']}): {r['return_1d']:+.1%}"
                            )
                if sector_lines:
                    parts.append("**📈 Local sector data:**\n" + "\n".join(sector_lines))
        except Exception:
            pass

    # 2c. Regex fallback if planner found nothing
    if not parts and not plan:
        try:
            try:
                from web.local_trading_agent import _discover_local_from_web
            except ImportError:
                from local_trading_agent import _discover_local_from_web
            hint = [market] if market and market not in (None, "ALL", "global") else None
            fallback = _discover_local_from_web(web_result, data_dir, hint)
            if fallback:
                parts.append(fallback)
        except Exception:
            pass

    return "\n\n".join(parts) if parts else ""


def agent_chat(agent_type, market, message, data_dir, chat_history=None, language=None):
    """
    LLM-backed agent chat with LLM router.

    Flow:
      1. Router LLM  → decides which tools to invoke + extracts tickers
      2. Tool execution → runs selected tools in parallel
      3. Synthesizer LLM → produces final beginner-friendly response
    """
    import re as _re

    provider, api_key = get_llm_provider()
    if provider is None:
        return None

    # 1. Route: ask LLM which tools to call
    route = _route_message(message, market, agent_type, provider, api_key)
    tools = route.get("tools", ["local_data"])
    tickers = route.get("tickers", [])
    company_name = route.get("company_name", "")

    # If LLM extracted a company name but no ticker, resolve it.
    # For CN agent: only look up A-shares; if found elsewhere, tell user to ask the right agent.
    _MARKET_NAMES = {
        "CN": "A股", "HK": "港股", "US": "美股", "JP": "日股",
        "KR": "韩股", "TW": "台股", "IN": "印度股", "UK": "英股",
        "DE": "德股", "FR": "法股", "AU": "澳股", "BR": "巴西股", "SA": "沙特股",
    }
    # Single-market agents: check that every resolved ticker belongs to this market.
    # If a ticker belongs to a different market, redirect the user.
    # Global agent has no boundary — skip this check.
    if "trading_agent" in tools and market and market not in (None, "ALL", "global"):
        try:
            from web.local_trading_agent import detect_market as _detect
        except ImportError:
            from local_trading_agent import detect_market as _detect

        def _redirect_msg(name_or_ticker, ticker_market):
            current_label = _MARKET_NAMES.get(market, market)
            alt_label     = _MARKET_NAMES.get(ticker_market, ticker_market)
            return (f"**{name_or_ticker}** 不是{current_label}，"
                    f"它在**{alt_label}**上市。\n\n"
                    f"请切换到 **{ticker_market} Market Agent** 提问。")

        # Case 1: tickers already extracted — check each one
        if tickers:
            foreign = [t for t in tickers if _detect(t) != market]
            if foreign:
                tk = foreign[0]
                tk_market = _detect(tk)
                display = company_name or tk
                return _redirect_msg(display, tk_market)

        # Case 2: company name but no ticker — try local market first, then others
        elif company_name:
            local_ticker = _resolve_cn_name_to_ticker(company_name, data_dir) \
                if market == "CN" else _find_company_in_market(company_name, market, data_dir)
            if local_ticker:
                tickers = [local_ticker]
            else:
                alt_ticker, alt_market = _find_company_in_markets(company_name, data_dir)
                current_label = _MARKET_NAMES.get(market, market)
                if alt_ticker and alt_market:
                    return _redirect_msg(company_name, alt_market)
                else:
                    return (f"**{company_name}** 不在{current_label}市场，"
                            f"本 agent 只覆盖{current_label}。\n\n"
                            f"请切换到对应的 Market Agent 提问。")

    # For the global agent: resolve company_name → ticker across all markets.
    # Single-market agents handle this above (with redirect); global agent has no boundary.
    if "trading_agent" in tools and not tickers and company_name \
            and (not market or market in (None, "ALL", "global")):
        resolved_ticker, resolved_market = _find_company_in_markets(company_name, data_dir)
        if resolved_ticker:
            tickers = [resolved_ticker]
        # If still not found, let it fall through to web_search / LLM synthesis

    # Force web_search if user explicitly requests it (override LLM router)
    _WEB_TRIGGERS = ["internet", "search", "browse", "google", "web", "online",
                     "find online", "look up", "look it up", "搜索", "搜一下",
                     "网上", "网络", "查一查", "查找"]
    if any(w in message.lower() for w in _WEB_TRIGGERS) and "web_search" not in tools:
        tools = list(tools) + ["web_search"]
    # 2a. Sector agent
    if "sector_agent" in tools and tickers:
        sector_query = tickers[0]
        try:
            from web.local_trading_agent import local_sector_analyze
        except ImportError:
            from local_trading_agent import local_sector_analyze
        return local_sector_analyze(sector_query, market, data_dir)

    # 2b. If trading_agent selected: route by market
    #    US  → open-source TradingAgents (stable, yfinance-backed)
    #    All others → our local multi-agent system (local pipeline + akshare/yfinance)
    if "trading_agent" in tools and tickers:
        ticker = tickers[0]
        try:
            from web.local_trading_agent import detect_market, local_trading_analyze
        except ImportError:
            from local_trading_agent import detect_market, local_trading_analyze
        if detect_market(ticker) == "US":
            return trading_agents_analyze(ticker)
        else:
            return local_trading_analyze(ticker, data_dir)

    # 3. Execute all other tools in parallel
    # Enrich the search message with recent ticker context from chat history
    search_message = message
    if "web_search" in tools and not tickers and chat_history:
        # Pull tickers mentioned in the last few turns to give web search context
        import re as _re2
        recent_text = " ".join(m.get("content", "") for m in chat_history[-6:])
        ctx_tickers  = _re2.findall(r'\b(\d{6})\b', recent_text)           # CN
        ctx_tickers += [t for t in _re2.findall(r'\b([A-Z]{1,5})\b', recent_text)
                        if t not in _ROUTER_IGNORE]                         # US
        if ctx_tickers:
            tickers = ctx_tickers[:2]
        else:
            # Inject company/ticker context into the search query string
            search_message = recent_text[-200:] + " " + message
    tool_results = _execute_tools(tools, tickers, market, data_dir, message=search_message)

    # 4a. Web → local enrichment via LLM planner
    #     After web search, ask LLM: "given web results + user question, what local data to fetch?"
    #     LLM decides tickers/sectors to query; fallback to regex if planner fails.
    if "web_search" in tool_results:
        web_raw = tool_results["web_search"]
        if "(web search failed" not in web_raw and "(no web results)" not in web_raw:
            local_enrichment = _web_to_local_enrichment(
                web_raw, message, data_dir, market, provider, api_key
            )
            if local_enrichment:
                tool_results["local_data_from_web"] = local_enrichment

    # 4b. Build combined context from tool outputs.
    # Keep total chars low enough to stay under the model's token limit.
    # Budget: ~8192 tokens total; reserve ~2000 for system prompt + history + completion.
    # That leaves ~6192 tokens ≈ ~12000 chars for combined_context.
    context_parts = []
    for tool_name, result in tool_results.items():
        label = tool_name.replace("_", " ").upper()
        if len(result) > 2500:
            result = result[:2500] + "\n...(truncated)"
        context_parts.append(f"=== {label} ===\n{result}")

    combined_context = "\n\n".join(context_parts)
    if len(combined_context) > 12000:
        combined_context = combined_context[:12000] + "\n...(truncated)"

    # 5. Pick system prompt
    is_global = agent_type == "global" or market in (None, "ALL")
    if is_global:
        base_system = SYSTEM_PROMPTS["global"]
    else:
        base_system = SYSTEM_PROMPTS["market"].format(market=market)

    system = base_system + f"""

RULES:
1. Use Markdown: headers (##/###), bold (**text**), bullet points, tables.
2. When listing stocks: **TICKER (Company Name)** — details.
3. NEVER assume the reader knows financial terms — always explain in parentheses.
4. Show reasoning step by step. Don't just state conclusions.
5. End with a plain-language takeaway for beginners.
6. Tools used to answer this: {', '.join(tools)}. Mention if data is live (OpenBB) vs cached (local pipeline).
7. NEVER ask the user to paste or provide data. You have complete market data — use it directly.
8. You CAN browse the internet. If WEB SEARCH results appear in your data below, summarize them directly.
   NEVER say "I cannot browse the internet" or "I don't have real-time access" — you DO have web search.
   If the search returned no useful results, say "I searched but couldn't find that information" instead."""

    full_system = f"{system}\n\nCRITICAL: NEVER ask the user to provide or paste data. You already have COMPLETE access to all market data below. Use it directly to answer.\n\n--- YOUR COMPLETE MARKET DATA (use this to answer, do not ask user for more) ---\n{combined_context}"

    # 6. Build conversation history (last 10 turns)
    history_msgs = []
    if chat_history:
        for msg in chat_history[-10:]:
            role = "assistant" if msg.get("role") == "agent" else "user"
            history_msgs.append({"role": role, "content": msg["content"]})

    # If web search was run, inject results directly into the user message
    # so the LLM cannot fall back to "I can't browse the internet"
    web_result = tool_results.get("web_search", "")
    if web_result and "(web search failed" not in web_result and "(no web results)" not in web_result:
        wrapped_message = (
            f"[Web search was performed. Results below — answer based on these results.]\n\n"
            f"{web_result[:3000]}\n\n"
            f"---\nUser question: {message}"
        )
    else:
        wrapped_message = message

    try:
        if provider == "anthropic":
            result = _call_anthropic(api_key, full_system, history_msgs, wrapped_message, max_tokens=800)
        else:
            result = _call_openai(api_key, full_system, history_msgs, wrapped_message, max_tokens=800)

        # Fallback: if response admits inability and web search wasn't used, retry with web search
        _INABILITY_PHRASES = [
            "don't have access", "cannot access", "i don't have", "not in my data",
            "i'm unable to", "i cannot browse", "outside my knowledge",
            "i don't have information", "no data available", "i lack",
            "i cannot provide", "i do not have real-time",
        ]
        if (any(p in (result or "").lower() for p in _INABILITY_PHRASES)
                and "web_search" not in tools):
            fallback_results = _execute_tools(["web_search"], tickers, market, data_dir, message=search_message)
            fb_web = fallback_results.get("web_search", "")
            if fb_web and "(web search failed" not in fb_web and "(no web results)" not in fb_web:
                fb_message = (
                    f"[Fallback web search performed. Results below — answer based on these.]\n\n"
                    f"{fb_web[:3000]}\n\n"
                    f"---\nUser question: {message}"
                )
                tools = list(tools) + ["web_search"]
                tool_results["web_search"] = fb_web
                web_result = fb_web
                if provider == "anthropic":
                    result = _call_anthropic(api_key, full_system, history_msgs, fb_message)
                else:
                    result = _call_openai(api_key, full_system, history_msgs, fb_message)

        # Append web sources footer if web search was used
        if web_result and "(web search failed" not in web_result and "(no web results)" not in web_result:
            sources_footer = _extract_sources(web_result)
            if sources_footer:
                result = (result or "") + sources_footer

        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"(error: {e})"


def _call_anthropic(api_key, system, history, message, max_tokens=1000):
    """Call Anthropic Claude API."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    messages = list(history)
    messages.append({"role": "user", "content": message})

    # Ensure messages alternate user/assistant (Anthropic requirement)
    cleaned = []
    for msg in messages:
        if cleaned and cleaned[-1]["role"] == msg["role"]:
            cleaned[-1]["content"] += "\n" + msg["content"]
        else:
            cleaned.append(msg)

    try:
        resp = client.messages.create(
            model=model,
            system=system,
            messages=cleaned,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.content[0].text if resp.content else ""
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise


def _call_openai(api_key, system, history, message, max_tokens=1000):
    """Call OpenAI-compatible API using config file settings."""
    from openai import OpenAI
    cfg = _load_llm_config()
    key = api_key or cfg.get("api_key", "")
    base_url = cfg.get("base_url", "https://api.openai.com/v1") or "https://api.openai.com/v1"
    model = cfg.get("model", "gpt-4o-mini") or "gpt-4o-mini"
    import httpx
    client = OpenAI(api_key=key, base_url=base_url, http_client=httpx.Client(verify=False))

    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.choices[0].message.content


# ── TradingAgents deep analysis ───────────────────────────────────

_ta_instance = None
_openai_patched = False  # reset each server start so patch is always applied fresh


def _patch_openai_context_truncation(model_token_limit: int = 8192, max_completion_tokens: int = 1500):
    """Monkey-patch openai.Completions.create to ensure input + output never exceeds
    the model's context window.

    Two guards:
    1. Cap max_tokens (completion) so that message_tokens + max_tokens <= model_token_limit.
    2. Drop oldest non-system messages if messages alone are too long.

    At ~4 chars/token, leaves room for both input and a reasonable response.
    """
    global _openai_patched
    if _openai_patched:
        return
    try:
        import openai.resources.chat.completions as _comp_mod
        _orig_create = _comp_mod.Completions.create

        def _safe_create(self, *args, **kwargs):
            msgs = list(kwargs.get("messages") or [])

            # Estimate message tokens (rough: 4 chars ≈ 1 token)
            def _tokens(m):
                return len(str(m.get("content") or "")) // 4

            msg_tokens = sum(_tokens(m) for m in msgs)

            # 1. Cap completion tokens so input + output fits in context window.
            # Use whichever key is already set (max_completion_tokens takes precedence
            # over max_tokens; never set both — OpenAI rejects that combination).
            headroom = model_token_limit - msg_tokens - 50
            capped = max(200, min(max_completion_tokens, headroom))
            if "max_completion_tokens" in kwargs:
                kwargs["max_completion_tokens"] = min(kwargs["max_completion_tokens"], capped)
            else:
                # Remove max_tokens if present, then set max_completion_tokens
                kwargs.pop("max_tokens", None)
                kwargs["max_completion_tokens"] = capped

            # 2. If messages alone exceed limit, drop oldest non-system messages
            max_msg_tokens = model_token_limit - 200  # leave 200 for completion minimum
            while msg_tokens > max_msg_tokens and len(msgs) > 1:
                for i, m in enumerate(msgs):
                    if m.get("role") != "system":
                        msg_tokens -= _tokens(m)
                        msgs.pop(i)
                        break
                else:
                    msgs[0]["content"] = str(msgs[0].get("content") or "")[: max_msg_tokens * 4]
                    break
            kwargs["messages"] = msgs

            return _orig_create(self, *args, **kwargs)

        _comp_mod.Completions.create = _safe_create
        _openai_patched = True
    except Exception:
        pass


def _get_trading_agents():
    """Lazy-init TradingAgents using the user's configured LLM key."""
    global _ta_instance
    if _ta_instance is not None:
        return _ta_instance
    try:
        cfg = _load_llm_config()
        api_key  = cfg.get("api_key", "").strip()
        base_url = cfg.get("base_url", "").strip() or "https://api.openai.com/v1"
        model    = cfg.get("model", "gpt-4o-mini").strip() or "gpt-4o-mini"

        if not api_key:
            return None  # No key configured

        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["OPENAI_API_KEY"] = api_key
        # Fix macOS Python SSL certificate issue
        try:
            import certifi
            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
            os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        except ImportError:
            pass

        _patch_openai_context_truncation()

        from tradingagents.graph.trading_graph import TradingAgentsGraph
        from tradingagents.default_config import DEFAULT_CONFIG

        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = "openai"
        config["deep_think_llm"]  = model
        config["quick_think_llm"] = model
        # TradingAgents passes backend_url directly to ChatOpenAI(base_url=...) which needs the /v1 suffix
        config["backend_url"] = base_url.rstrip("/") or "https://api.openai.com/v1"
        config["max_debate_rounds"] = 1
        config["max_risk_discuss_rounds"] = 1

        _ta_instance = TradingAgentsGraph(debug=False, config=config)
        return _ta_instance
    except Exception:
        return None


def _local_cn_supplement(ticker: str) -> str:
    """Build a local-data supplement section for a CN A-share ticker.

    Reads from data/markets/CN/ — price history, volume signals, news, alerts,
    capital flow — and returns a formatted markdown string to append to the
    TradingAgents output when remote data sources are unavailable.
    """
    try:
        base = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "markets", "CN"
        )
        sections = ["---", "## 📊 Local Pipeline Data (CN A-share supplement)"]

        # ── Price & volume signals ──────────────────────────────────
        snap = _latest_daily_parquet(base)
        if not snap.empty and "ticker" in snap.columns:
            row = snap[snap["ticker"] == ticker]
            if not row.empty:
                r = row.iloc[0]
                close      = r.get("close", "N/A")
                ret_1d     = r.get("return_1d", None)
                ret_5d     = r.get("return_5d", None)
                ret_20d    = r.get("return_20d", None)
                vol_ratio  = r.get("volume_ratio", None)
                volume     = r.get("volume", None)
                lines = [f"**Price & Returns** (latest pipeline snapshot)"]
                lines.append(f"- Close: {close}")
                if ret_1d  is not None: lines.append(f"- 1-day return:  {ret_1d:+.2%}")
                if ret_5d  is not None: lines.append(f"- 5-day return:  {ret_5d:+.2%}")
                if ret_20d is not None: lines.append(f"- 20-day return: {ret_20d:+.2%}")
                if vol_ratio is not None:
                    flag = " ⚠️ unusual volume" if vol_ratio >= 2.0 else ""
                    lines.append(f"- Volume ratio:  {vol_ratio:.2f}x{flag}")
                if volume is not None: lines.append(f"- Volume:        {int(volume):,}")
                sections.append("\n".join(lines))

        # ── Alerts ─────────────────────────────────────────────────
        alerts_path = os.path.join(base, "alerts.json")
        alerts = _load_json(alerts_path)
        if alerts and isinstance(alerts, list):
            ticker_alerts = [a for a in alerts if isinstance(a, dict)
                             and a.get("ticker") == ticker]
            if ticker_alerts:
                lines = [f"**Active Alerts** ({len(ticker_alerts)})"]
                for a in ticker_alerts[:5]:
                    atype = a.get("alert_type", a.get("type", "alert"))
                    msg   = a.get("message", a.get("msg", str(a)))
                    lines.append(f"- `{atype}`: {msg}")
                sections.append("\n".join(lines))

        # ── News ───────────────────────────────────────────────────
        try:
            news = pd.read_parquet(os.path.join(base, "news.parquet"))
            if not news.empty and "ticker" in news.columns:
                tnews = news[news["ticker"] == ticker].copy()
                if not tnews.empty:
                    if "hit_count" in tnews.columns:
                        tnews = tnews.sort_values("hit_count", ascending=False)
                    lines = [f"**Recent News** ({len(tnews)} articles, top 5 by relevance)"]
                    for _, nr in tnews.head(5).iterrows():
                        title = nr.get("title", "")
                        pub   = nr.get("publisher", "")
                        hits  = nr.get("hit_count", "")
                        lines.append(f"- {title}" + (f" ({pub})" if pub else "")
                                     + (f" [hits:{hits}]" if hits else ""))
                    sections.append("\n".join(lines))
        except Exception:
            pass

        # ── Capital flow ───────────────────────────────────────────
        try:
            flow = pd.read_parquet(os.path.join(base, "capital_flow.parquet"))
            if not flow.empty:
                latest = flow.tail(3)
                lines = ["**Capital Flow** (last 3 entries)"]
                for _, fr in latest.iterrows():
                    parts_f = []
                    for col in ["date", "flow_type", "net_flow", "net_flow_usd"]:
                        if col in fr.index and fr[col] is not None:
                            parts_f.append(f"{col}: {fr[col]}")
                    lines.append("- " + ", ".join(parts_f))
                sections.append("\n".join(lines))
        except Exception:
            pass

        if len(sections) <= 2:  # only header lines, no real data
            return ""
        sections.append("*Source: local akshare pipeline data*")
        return "\n\n".join(sections)
    except Exception:
        return ""


def trading_agents_analyze(ticker, date=None):
    """Run TradingAgents multi-agent analysis on a single ticker.

    Returns a formatted analysis string combining the decision and
    detailed reasoning from the multi-agent state.
    """
    # Pre-check: key configured?
    cfg = _load_llm_config()
    if not cfg.get("api_key", "").strip():
        return ("⚙ LLM API key not configured.\n\n"
                "Open the chat panel → click ⚙ gear icon → enter your API key and save.")

    # Validate ticker format: US (1-5 letters), CN A-share (6 digits), HK (4-5 digits)
    import re as _re
    ticker = ticker.strip()
    if not _re.match(r'^(\d{4,6}|[A-Z]{1,5}(?:\.[A-Z]{1,2})?)$', ticker):
        return (f"**'{ticker}'** doesn't look like a stock ticker code.\n\n"
                f"Please use:\n"
                f"- **CN A-shares**: 6-digit code, e.g. `688031` (星环科技)\n"
                f"- **US stocks**: symbol, e.g. `NVDA`\n"
                f"- **HK stocks**: 4–5 digit code, e.g. `0700` (Tencent)\n\n"
                f"Tip: You can find the ticker code in the Companies or Market pages.")

    ta = _get_trading_agents()
    if ta is None:
        return ("TradingAgents could not be initialized.\n\n"
                "Make sure the `tradingagents` package is installed and your API key is correct.")
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    try:
        import time as _time
        last_err = None
        for _attempt in range(3):
            try:
                state, decision = ta.propagate(ticker, date)
                break
            except Exception as _e:
                last_err = _e
                if "429" in str(_e) or "rate_limit" in str(_e).lower() or "rate limit" in str(_e).lower():
                    wait = 15 * (_attempt + 1)
                    _time.sleep(wait)
                else:
                    raise
        else:
            raise last_err

        # Build rich output from state
        parts = [f"TradingAgents Analysis: {ticker} ({date})",
                 f"Decision: {decision}", ""]

        # Extract detailed reasoning from state dict
        if isinstance(state, dict):
            for key in ["market_report", "fundamentals_report",
                        "news_report", "sentiment_report",
                        "investment_plan", "trader_investment_plan",
                        "final_trade_decision"]:
                val = state.get(key)
                if val and isinstance(val, str) and len(val) > 10:
                    label = key.replace("_", " ").title()
                    parts.append(f"--- {label} ---")
                    parts.append(val.strip())
                    parts.append("")

        result = "\n".join(parts)
        result = result if len(result) > 50 else decision

        # Append local pipeline data for CN A-shares (supplements missing remote data)
        import re as _re2
        if _re2.match(r'^\d{6}$', ticker):
            local = _local_cn_supplement(ticker)
            if local:
                result += "\n\n" + local

        # If TradingAgents signals insufficient data, fall back to local multi-agent analysis.
        # This handles Chinese ADRs (BABA, BIDU, JD, PDD, etc.) which are US-listed but
        # have poor coverage in TradingAgents' remote fundamentals sources.
        insufficient = (
            isinstance(decision, dict) and "没有足够" in str(decision.get("reasoning", ""))
        ) or "没有足够" in result
        if insufficient:
            try:
                from web.local_trading_agent import local_trading_analyze
            except ImportError:
                from local_trading_agent import local_trading_analyze
            data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
            )
            return local_trading_analyze(ticker, data_dir, _allow_us=True)

        return result
    except Exception as e:
        err = str(e)
        if "connection" in err.lower() or "connect" in err.lower() or "timeout" in err.lower():
            return (f"Analysis failed for {ticker}: could not reach the LLM API.\n\n"
                    f"Check your Base URL and API key in ⚙ settings. Error: {err}")
        if "401" in err or "unauthorized" in err.lower() or "authentication" in err.lower():
            return (f"Analysis failed for {ticker}: API key rejected (401).\n\n"
                    f"Update your API key in ⚙ settings.")
        return f"Analysis failed for {ticker}: {err}"
