"""
Local Multi-Agent Trading Analysis
====================================
Mimics TradingAgents' multi-agent debate architecture using local pipeline
data and direct LLM calls.  Used for non-US markets (CN, HK, JP, KR, etc.)
where the open-source TradingAgents has poor data coverage.

Agent pipeline (sequential, to respect gpt-4 8192-token limit):
  1. Market Analyst      — price trend, volume signals, technical indicators
  2. Fundamentals Analyst — P/E, P/B, revenue, profitability
  3. News/Sentiment Analyst — recent news, keyword matches, market mood
  4. Bull Researcher     — builds bull case from the 3 reports
  5. Bear Researcher     — builds bear case from the 3 reports
  6. Risk Analyst        — assesses downside risks
  7. Decision Maker      — synthesizes debate into BUY / HOLD / SELL

Integration: called from agent_llm.py for non-US tickers.
"""

import os
import json
import glob
import re

import pandas as pd


# ── Market detection ──────────────────────────────────────────────

def detect_market(ticker: str) -> str:
    """Return market code from ticker format."""
    t = ticker.strip().upper()
    if re.match(r'^\d{6}$', t):
        return "CN"
    if re.match(r'^\d{4,5}(\.HK)?$', t, re.I):
        return "HK"
    if re.match(r'^[A-Z]{1,5}$', t):
        return "US"
    # Suffixed tickers: 7203.T → JP, 005930.KS → KR, etc.
    m = re.match(r'^[\dA-Z]+\.([A-Z]+)$', t)
    if m:
        suffix_map = {"T": "JP", "KS": "KR", "TW": "TW", "NS": "IN",
                      "L": "UK", "DE": "DE", "PA": "FR", "AX": "AU", "SA": "SA"}
        return suffix_map.get(m.group(1), "UNKNOWN")
    return "UNKNOWN"


# ── Data gathering ────────────────────────────────────────────────

def _load_local_price(ticker: str, market: str, data_dir: str) -> dict:
    """Load latest price snapshot + 20-day history from local pipeline."""
    base = os.path.join(data_dir, "markets", market)
    result = {}

    files = sorted(glob.glob(os.path.join(base, "market_daily_*.parquet")))
    if not files:
        return result
    try:
        df = pd.read_parquet(files[-1])
        if "ticker" in df.columns:
            tdf = df[df["ticker"] == ticker].copy()
            if tdf.empty:
                return result
            tdf = tdf.sort_values("date") if "date" in tdf.columns else tdf

            latest = tdf.iloc[-1]
            result["close"]       = latest.get("close")
            result["return_1d"]   = latest.get("return_1d")
            result["return_5d"]   = latest.get("return_5d")
            result["return_20d"]  = latest.get("return_20d")
            result["volume_ratio"]= latest.get("volume_ratio")
            result["volume"]      = latest.get("volume")

            # 20-day history as compact list
            cols = [c for c in ["date", "close", "volume", "return_1d"] if c in tdf.columns]
            result["history"] = tdf[cols].tail(20).to_dict("records")
    except Exception:
        pass
    return result


def _load_local_news(ticker: str, market: str, data_dir: str) -> list:
    try:
        path = os.path.join(data_dir, "markets", market, "news.parquet")
        news = pd.read_parquet(path)
        if "ticker" not in news.columns:
            return []
        tnews = news[news["ticker"] == ticker].copy()
        if tnews.empty:
            return []
        if "hit_count" in tnews.columns:
            tnews = tnews.sort_values("hit_count", ascending=False)
        cols = [c for c in ["title", "publisher", "hit_count", "keywords_matched"] if c in tnews.columns]
        return tnews[cols].head(8).to_dict("records")
    except Exception:
        return []


def _load_local_alerts(ticker: str, market: str, data_dir: str) -> list:
    try:
        path = os.path.join(data_dir, "markets", market, "alerts.json")
        with open(path) as f:
            alerts = json.load(f)
        return [a for a in (alerts or []) if isinstance(a, dict) and a.get("ticker") == ticker]
    except Exception:
        return []


def _load_local_universe(ticker: str, market: str, data_dir: str) -> dict:
    try:
        path = os.path.join(data_dir, "markets", market, "universe.parquet")
        uni = pd.read_parquet(path)
        row = uni[uni["ticker"] == ticker]
        if row.empty:
            return {}
        r = row.iloc[0]
        return {k: r.get(k) for k in ["name", "sector", "industry", "exchange"] if r.get(k) is not None}
    except Exception:
        return {}


def _load_cn_fundamentals(ticker: str) -> dict:
    """Fetch CN A-share valuation & financial data via akshare."""
    result = {}
    try:
        import akshare as ak
        # Valuation indicators (P/E, P/B, market cap, dividend yield)
        val = ak.stock_a_lg_indicator_push(stock=ticker)
        if not val.empty:
            row = val.iloc[-1]
            for col in val.columns:
                result[col] = row[col]
    except Exception:
        pass
    try:
        import akshare as ak
        # Latest financial summary
        fin = ak.stock_financial_abstract_ths(symbol=ticker, indicator="按年度")
        if not fin.empty:
            latest = fin.iloc[0]
            result["financials"] = latest.to_dict()
    except Exception:
        pass
    return result


def _load_yf_fundamentals(ticker: str) -> dict:
    """Fetch fundamentals via yfinance for HK/JP/KR/other markets."""
    result = {}
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info or {}
        wanted = ["shortName", "sector", "industry", "trailingPE", "priceToBook",
                  "returnOnEquity", "revenueGrowth", "profitMargins", "debtToEquity",
                  "currentPrice", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
                  "marketCap", "dividendYield", "earningsGrowth", "forwardPE"]
        for k in wanted:
            if info.get(k) is not None:
                result[k] = info[k]
    except Exception:
        pass
    return result


def gather_data(ticker: str, market: str, data_dir: str) -> dict:
    """Collect all available data for the ticker."""
    info    = _load_local_universe(ticker, market, data_dir)
    price   = _load_local_price(ticker, market, data_dir)
    news    = _load_local_news(ticker, market, data_dir)
    alerts  = _load_local_alerts(ticker, market, data_dir)

    if market == "CN":
        fundamentals = _load_cn_fundamentals(ticker)
    else:
        # Try yfinance for all non-CN non-US markets
        yf_ticker = ticker if "." in ticker else ticker  # suffix already on ticker for non-CN
        fundamentals = _load_yf_fundamentals(yf_ticker)

    return {
        "ticker":       ticker,
        "market":       market,
        "info":         info,
        "price":        price,
        "news":         news,
        "alerts":       alerts,
        "fundamentals": fundamentals,
    }


# ── Prompt builders ───────────────────────────────────────────────

def _fmt(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False, default=str, indent=2)


def _price_summary(price: dict) -> str:
    if not price:
        return "No price data available from local pipeline."
    lines = []
    if price.get("close"):      lines.append(f"Close: {price['close']}")
    if price.get("return_1d") is not None: lines.append(f"1d return: {price['return_1d']:+.2%}")
    if price.get("return_5d") is not None: lines.append(f"5d return: {price['return_5d']:+.2%}")
    if price.get("return_20d") is not None: lines.append(f"20d return: {price['return_20d']:+.2%}")
    if price.get("volume_ratio") is not None:
        vr = price["volume_ratio"]
        flag = " ⚠️ abnormal" if vr >= 2.0 else ""
        lines.append(f"Volume ratio: {vr:.2f}x{flag}")
    hist = price.get("history", [])
    if hist:
        closes = [str(round(h["close"], 2)) for h in hist if h.get("close")]
        lines.append(f"20d close history: [{', '.join(closes[-10:])}]")
    return "\n".join(lines)


# ── LLM analyst calls ─────────────────────────────────────────────

def _call_llm(system: str, user: str, provider: str, api_key: str,
              max_tokens: int = 600) -> str:
    """Thin wrapper — reuse the openai/anthropic callers from agent_llm."""
    try:
        # Import from sibling module at runtime to avoid circular imports
        if provider == "anthropic":
            from web.agent_llm import _call_anthropic
            return _call_anthropic(api_key, system, [], user, max_tokens=max_tokens)
        else:
            from web.agent_llm import _call_openai
            return _call_openai(api_key, system, [], user, max_tokens=max_tokens)
    except ImportError:
        from agent_llm import _call_anthropic, _call_openai
        if provider == "anthropic":
            return _call_anthropic(api_key, system, [], user, max_tokens=max_tokens)
        return _call_openai(api_key, system, [], user, max_tokens=max_tokens)


def run_market_analyst(data: dict, provider: str, api_key: str) -> str:
    ticker = data["ticker"]
    name   = data["info"].get("name", ticker)
    system = (
        "You are a technical market analyst. Analyze the provided price and volume data. "
        "Be concise. Output 3-5 bullet points covering: trend direction, volume signal, "
        "key support/resistance, and momentum. Plain language, beginner-friendly."
    )
    user = (
        f"Stock: {name} ({ticker})\n\n"
        f"PRICE & VOLUME DATA:\n{_price_summary(data['price'])}\n\n"
        f"ALERTS: {json.dumps(data['alerts'], ensure_ascii=False, default=str)[:500]}\n\n"
        "Write a concise technical analysis (max 200 words)."
    )
    return _call_llm(system, user, provider, api_key, max_tokens=400)


def run_fundamentals_analyst(data: dict, provider: str, api_key: str) -> str:
    ticker = data["ticker"]
    name   = data["info"].get("name", ticker)
    fund   = data["fundamentals"]
    info   = data["info"]

    system = (
        "You are a fundamental analyst. Assess the company's financial health. "
        "Cover valuation (P/E, P/B), profitability, growth, and debt. "
        "If data is missing say so briefly. Be concise and beginner-friendly."
    )
    fund_text = _fmt(fund)[:1500] if fund else "No fundamental data available."
    user = (
        f"Stock: {name} ({ticker})\n"
        f"Sector: {info.get('sector', 'N/A')}  Industry: {info.get('industry', 'N/A')}\n\n"
        f"FUNDAMENTAL DATA:\n{fund_text}\n\n"
        "Write a concise fundamentals analysis (max 200 words)."
    )
    return _call_llm(system, user, provider, api_key, max_tokens=400)


def run_news_analyst(data: dict, provider: str, api_key: str) -> str:
    ticker = data["ticker"]
    name   = data["info"].get("name", ticker)
    news   = data["news"]

    # If no local news, fall back to live web search
    source_label = "local pipeline"
    web_sources = ""
    if not news:
        try:
            from web.agent_llm import _web_search
        except ImportError:
            from agent_llm import _web_search
        query = f"{name} {ticker} stock news"
        web_raw = _web_search(query, max_results=6)
        news_text = web_raw[:1500]
        source_label = "live web search"
        # Extract source URLs
        urls = [l.strip() for l in web_raw.split("\n") if l.strip().startswith("http")]
        if urls:
            web_sources = "\n\n*Sources: " + " · ".join(urls[:4]) + "*"
    else:
        news_text = _fmt(news)[:1200]

    system = (
        "You are a news and sentiment analyst. Assess recent news for this stock. "
        "Identify positive/negative catalysts and overall market sentiment. "
        "Be concise and beginner-friendly."
    )
    user = (
        f"Stock: {name} ({ticker})\n"
        f"News source: {source_label}\n\n"
        f"RECENT NEWS:\n{news_text}\n\n"
        "Write a concise news/sentiment analysis (max 200 words)."
    )
    result = _call_llm(system, user, provider, api_key, max_tokens=400)
    return result + web_sources


def run_bull_researcher(ticker: str, name: str,
                        tech: str, fund: str, news: str,
                        provider: str, api_key: str) -> str:
    system = (
        "You are a bull researcher. Using the analyst reports provided, "
        "make the strongest possible BULL case for buying this stock. "
        "3-4 bullet points. Be persuasive but grounded in the data."
    )
    user = (
        f"Stock: {name} ({ticker})\n\n"
        f"TECHNICAL: {tech[:400]}\n\n"
        f"FUNDAMENTALS: {fund[:400]}\n\n"
        f"NEWS: {news[:400]}\n\n"
        "Make the bull case (max 150 words)."
    )
    return _call_llm(system, user, provider, api_key, max_tokens=300)


def run_bear_researcher(ticker: str, name: str,
                        tech: str, fund: str, news: str,
                        provider: str, api_key: str) -> str:
    system = (
        "You are a bear researcher. Using the analyst reports provided, "
        "make the strongest possible BEAR case against this stock. "
        "3-4 bullet points. Be critical but grounded in the data."
    )
    user = (
        f"Stock: {name} ({ticker})\n\n"
        f"TECHNICAL: {tech[:400]}\n\n"
        f"FUNDAMENTALS: {fund[:400]}\n\n"
        f"NEWS: {news[:400]}\n\n"
        "Make the bear case (max 150 words)."
    )
    return _call_llm(system, user, provider, api_key, max_tokens=300)


def run_risk_analyst(ticker: str, name: str,
                     bear: str, data: dict,
                     provider: str, api_key: str) -> str:
    system = (
        "You are a risk analyst. Identify the top 3 risks for this stock "
        "and rate overall risk as Low / Medium / High. Be brief and specific."
    )
    vr = data["price"].get("volume_ratio")
    vol_note = f"Volume ratio: {vr:.1f}x" if vr else ""
    user = (
        f"Stock: {name} ({ticker}). {vol_note}\n\n"
        f"BEAR CASE:\n{bear[:500]}\n\n"
        f"ALERTS: {json.dumps(data['alerts'], ensure_ascii=False, default=str)[:300]}\n\n"
        "List top 3 risks and overall risk rating (max 120 words)."
    )
    return _call_llm(system, user, provider, api_key, max_tokens=250)


def run_decision_maker(ticker: str, name: str,
                       tech: str, fund: str, news: str,
                       bull: str, bear: str, risk: str,
                       provider: str, api_key: str) -> str:
    system = (
        "You are a portfolio manager making a final trading decision. "
        "Weigh the bull and bear cases and all analyst reports. "
        "Output your decision in this exact format:\n"
        "**Decision: BUY / HOLD / SELL**\n"
        "**Confidence: High / Medium / Low**\n"
        "**Reasoning:** 2-3 sentences.\n"
        "**Key risk:** 1 sentence.\n"
        "**Beginner takeaway:** 1 plain-language sentence for a new investor."
    )
    user = (
        f"Stock: {name} ({ticker})\n\n"
        f"Technical: {tech[:300]}\n"
        f"Fundamentals: {fund[:300]}\n"
        f"News: {news[:300]}\n"
        f"Bull case: {bull[:250]}\n"
        f"Bear case: {bear[:250]}\n"
        f"Risk: {risk[:200]}\n\n"
        "Make your final decision."
    )
    return _call_llm(system, user, provider, api_key, max_tokens=350)


# ── Main entry point ──────────────────────────────────────────────

def local_trading_analyze(ticker: str, data_dir: str) -> str:
    """
    Run full multi-agent analysis for a non-US ticker.

    Returns formatted markdown analysis string.
    """
    from web.agent_llm import get_llm_provider, _local_cn_supplement
    try:
        from web.agent_llm import get_llm_provider, _local_cn_supplement
    except ImportError:
        from agent_llm import get_llm_provider, _local_cn_supplement

    provider, api_key = get_llm_provider()
    if not provider:
        return "⚙ LLM API key not configured. Open the chat panel → ⚙ gear icon → enter your API key."

    market = detect_market(ticker)
    if market == "US":
        return "US stocks are handled by TradingAgents. Use the standard analyze command."

    # 1. Gather all data
    data = gather_data(ticker, market, data_dir)
    name = data["info"].get("name", ticker)
    date = __import__("datetime").datetime.now().strftime("%Y-%m-%d")

    steps = []  # collect each agent's output

    def _step(label: str, fn, *args) -> str:
        try:
            result = fn(*args)
            return result or f"({label}: no output)"
        except Exception as e:
            return f"({label} unavailable: {e})"

    # 2. Run analysts
    tech  = _step("Technical Analyst",    run_market_analyst,       data, provider, api_key)
    fund  = _step("Fundamentals Analyst", run_fundamentals_analyst,  data, provider, api_key)
    news  = _step("News Analyst",         run_news_analyst,          data, provider, api_key)

    # 3. Bull / Bear debate
    bull  = _step("Bull Researcher",  run_bull_researcher, ticker, name, tech, fund, news, provider, api_key)
    bear  = _step("Bear Researcher",  run_bear_researcher, ticker, name, tech, fund, news, provider, api_key)

    # 4. Risk assessment
    risk  = _step("Risk Analyst",     run_risk_analyst, ticker, name, bear, data, provider, api_key)

    # 5. Final decision
    decision = _step("Decision Maker", run_decision_maker,
                     ticker, name, tech, fund, news, bull, bear, risk, provider, api_key)

    # 6. Format output
    market_label = {
        "CN": "CN A-share", "HK": "HK Stock", "JP": "Japan",
        "KR": "South Korea", "TW": "Taiwan", "IN": "India",
        "UK": "UK", "DE": "Germany", "FR": "France",
        "AU": "Australia", "BR": "Brazil", "SA": "Saudi Arabia",
    }.get(market, market)

    lines = [
        f"# Multi-Agent Analysis: {name} ({ticker})",
        f"*{market_label} · {date}*",
        "",
        "---",
        "## 📈 Technical Analysis",
        tech,
        "",
        "## 💼 Fundamentals",
        fund,
        "",
        "## 📰 News & Sentiment",
        news,
        "",
        "## 🐂 Bull Case",
        bull,
        "",
        "## 🐻 Bear Case",
        bear,
        "",
        "## ⚠️ Risk Assessment",
        risk,
        "",
        "---",
        "## 🏁 Final Decision",
        decision,
    ]

    # Append local pipeline data supplement for CN stocks
    if market == "CN":
        supplement = _local_cn_supplement(ticker)
        if supplement:
            lines += ["", supplement]

    return "\n".join(lines)


# ── Sector analysis ───────────────────────────────────────────────

# Maps user-facing keywords → sector values stored in universe.parquet
_SECTOR_ALIASES = {
    # English
    "tech": ["软件开发", "IT服务Ⅲ", "半导体", "计算机设备", "通信设备",
             "software", "semiconductor", "cloud_infrastructure", "internet_platforms",
             "AI_infrastructure", "cybersecurity", "networking_telecom", "hardware_devices", "IT_services"],
    "semiconductor": ["半导体", "semiconductor"],
    "software":      ["软件开发", "software"],
    "finance":       ["银行", "证券", "保险", "fintech", "finance"],
    "healthcare":    ["医疗器械", "医药生物", "healthcare", "biotech"],
    "energy":        ["能源", "石油石化", "energy", "oil"],
    "consumer":      ["食品饮料", "零售", "consumer", "retail"],
    "industrial":    ["机械设备", "建筑", "industrial"],
    "communication": ["通信设备", "传媒", "networking_telecom", "communication"],
    "ev":            ["汽车", "EV_tech", "ev", "electric vehicle"],
    "robotics":      ["Specialty Industrial Machinery", "Electrical Equipment & Parts",
                      "Scientific & Technical Instruments", "robotics", "automation"],
    "industrial":    ["Specialty Industrial Machinery", "Electrical Equipment & Parts",
                      "机械设备", "建筑", "industrial"],
    "electrical":    ["Electrical Equipment & Parts", "Electronic Components"],
    # Chinese keywords
    "科技": ["软件开发", "IT服务Ⅲ", "半导体", "计算机设备", "通信设备"],
    "半导体": ["半导体", "semiconductor", "Semiconductors"],
    "软件": ["软件开发", "software", "Software - Application", "Software - Infrastructure"],
    "金融": ["银行", "证券", "保险", "fintech"],
    "医疗": ["医疗器械", "医药生物", "Biotechnology", "Medical Devices", "Healthcare"],
    "能源": ["能源", "石油石化", "Solar", "Utilities - Renewable"],
    "消费": ["食品饮料", "零售"],
    "通信": ["通信设备", "传媒", "Telecom Services", "Communication Equipment"],
    "机器人": ["Specialty Industrial Machinery", "Electrical Equipment & Parts",
               "Scientific & Technical Instruments"],
    "工业": ["Specialty Industrial Machinery", "Electrical Equipment & Parts",
              "机械设备", "Tools & Accessories"],
    "新能源": ["Solar", "Utilities - Renewable", "Electrical Equipment & Parts"],
    "医疗器械": ["Medical Devices", "Scientific & Technical Instruments", "医疗器械"],
}


def resolve_sector(query: str) -> list[str]:
    """Return list of sector values to match for a user query string."""
    q = query.lower().strip()
    for key, vals in _SECTOR_ALIASES.items():
        if key in q:
            return vals
    # Fall back to direct substring match
    return [query]


def _gather_sector_data(sector_query: str, market: str, data_dir: str) -> dict:
    """Aggregate data for all stocks in a sector."""
    base    = os.path.join(data_dir, "markets", market)
    sectors = resolve_sector(sector_query)

    # Load universe and filter by sector
    try:
        uni = pd.read_parquet(os.path.join(base, "universe.parquet"))
    except Exception:
        return {"error": f"No universe data for {market}"}

    if "sector" not in uni.columns and "subsector" not in uni.columns:
        return {"error": "Universe has no sector or subsector column"}

    match_source = "sector"

    # Try sector column first
    if "sector" in uni.columns:
        mask = uni["sector"].apply(lambda s: any(sv.lower() in str(s).lower() for sv in sectors))
        sector_stocks = uni[mask].copy()
        if sector_stocks.empty:
            sector_stocks = uni[uni["sector"].str.contains(sector_query, case=False, na=False)]

    # Fall back to subsector column if sector search found nothing
    if sector_stocks.empty and "subsector" in uni.columns:
        match_source = "subsector"
        mask_sub = uni["subsector"].apply(lambda s: any(sv.lower() in str(s).lower() for sv in sectors))
        sector_stocks = uni[mask_sub].copy()
        if sector_stocks.empty:
            sector_stocks = uni[uni["subsector"].str.contains(sector_query, case=False, na=False)]

    if sector_stocks.empty:
        available_sectors  = uni["sector"].dropna().unique().tolist() if "sector" in uni.columns else []
        available_subsects = uni["subsector"].dropna().unique().tolist() if "subsector" in uni.columns else []
        return {"error": (
            f"No stocks found for '{sector_query}'. "
            f"Sectors: {available_sectors[:10]}  "
            f"Subsectors (sample): {available_subsects[:15]}"
        )}

    tickers = sector_stocks["ticker"].tolist()
    actual_sectors = sector_stocks["sector"].unique().tolist() if "sector" in sector_stocks.columns else []
    actual_subsectors = sector_stocks["subsector"].dropna().unique().tolist() if "subsector" in sector_stocks.columns else []

    # Price data for sector stocks
    snap = _load_local_price.__module__ and None  # just get the parquet
    files = sorted(glob.glob(os.path.join(base, "market_daily_*.parquet")))
    price_rows = pd.DataFrame()
    if files:
        try:
            df = pd.read_parquet(files[-1])
            if "ticker" in df.columns:
                df = df.sort_values("date") if "date" in df.columns else df
                df = df.groupby("ticker").last().reset_index()
                price_rows = df[df["ticker"].isin(tickers)].copy()
        except Exception:
            pass

    # Merge names
    if not price_rows.empty and "name" not in price_rows.columns:
        price_rows = price_rows.merge(
            sector_stocks[["ticker", "name", "sector"]].drop_duplicates("ticker"),
            on="ticker", how="left"
        )

    # Top gainers / losers
    top_gainers = pd.DataFrame()
    top_losers  = pd.DataFrame()
    summary     = {}
    if not price_rows.empty and "return_1d" in price_rows.columns:
        valid = price_rows.dropna(subset=["return_1d"])
        top_gainers = valid.nlargest(5, "return_1d")
        top_losers  = valid.nsmallest(5, "return_1d")
        summary = {
            "stock_count":  len(valid),
            "avg_return_1d": float(valid["return_1d"].mean()),
            "avg_return_5d": float(valid["return_5d"].mean()) if "return_5d" in valid else None,
            "up_count":     int((valid["return_1d"] > 0).sum()),
            "down_count":   int((valid["return_1d"] < 0).sum()),
            "high_volume_count": int((valid.get("volume_ratio", pd.Series()) >= 2.0).sum())
                                  if "volume_ratio" in valid.columns else 0,
        }

    # News for top tickers by volume
    news = []
    try:
        news_df = pd.read_parquet(os.path.join(base, "news.parquet"))
        if "ticker" in news_df.columns:
            snews = news_df[news_df["ticker"].isin(tickers)]
            if "hit_count" in snews.columns:
                snews = snews.sort_values("hit_count", ascending=False)
            cols = [c for c in ["ticker", "title", "publisher", "hit_count"] if c in snews.columns]
            news = snews[cols].head(10).to_dict("records")
    except Exception:
        pass

    # Alerts
    alerts = []
    try:
        with open(os.path.join(base, "alerts.json")) as f:
            all_alerts = json.load(f)
        alerts = [a for a in (all_alerts or [])
                  if isinstance(a, dict) and a.get("ticker") in tickers]
    except Exception:
        pass

    cols_g = [c for c in ["ticker", "name", "close", "return_1d", "return_5d", "volume_ratio"]
              if c in top_gainers.columns]
    cols_l = [c for c in ["ticker", "name", "close", "return_1d", "return_5d", "volume_ratio"]
              if c in top_losers.columns]

    return {
        "market":           market,
        "sector_query":     sector_query,
        "match_source":     match_source,
        "matched_sectors":  actual_sectors,
        "matched_subsectors": actual_subsectors,
        "stock_count":      len(tickers),
        "summary":          summary,
        "top_gainers":      top_gainers[cols_g].to_dict("records") if not top_gainers.empty else [],
        "top_losers":       top_losers[cols_l].to_dict("records")  if not top_losers.empty else [],
        "news":             news,
        "alerts":           alerts,
    }


def _discover_local_from_web(web_text: str, data_dir: str,
                             hint_markets: list = None) -> str:
    """
    Given web search result text, extract any stock tickers mentioned and load
    their local pipeline data (price, universe info).  Returns a compact markdown
    block, or empty string if nothing was found locally.
    """
    import re as _re

    # Extract ticker candidates from web text
    cn_cands = set(_re.findall(r'\b(\d{6})\b', web_text))
    # US/global: 2-5 uppercase letters — filter against known universe to cut noise
    raw_us = set(_re.findall(r'\b([A-Z]{2,5})\b', web_text))

    # Load universe files to validate candidates
    markets_to_check = hint_markets or [
        "CN", "US", "HK", "JP", "KR", "TW", "IN", "UK", "DE", "FR", "AU", "BR", "SA"
    ]
    uni_by_market = {}
    for m in markets_to_check:
        path = os.path.join(data_dir, "markets", m, "universe.parquet")
        if os.path.exists(path):
            try:
                uni = pd.read_parquet(path)
                if "ticker" in uni.columns:
                    uni_by_market[m] = set(uni["ticker"].astype(str))
            except Exception:
                pass

    # Validate candidates against local universe
    found = []  # (ticker, market)
    for m, known in uni_by_market.items():
        cands = cn_cands if m == "CN" else raw_us
        for ticker in cands & known:
            found.append((ticker, m))

    if not found:
        return ""

    # Load local price + info for each discovered ticker (cap at 15)
    lines = ["**📊 Local pipeline data for stocks mentioned in web results:**"]
    for ticker, market in sorted(found)[:15]:
        price = _load_local_price(ticker, market, data_dir)
        info  = _load_local_universe(ticker, market, data_dir)
        if not price:
            continue
        name = info.get("name", ticker)
        subsector = info.get("subsector") or info.get("industry") or info.get("sector", "")
        close  = f"{price['close']:.2f}" if price.get("close") else "?"
        r1d    = f"{price['return_1d']:+.1%}" if price.get("return_1d") is not None else "?"
        r5d    = f"{price['return_5d']:+.1%}" if price.get("return_5d") is not None else "?"
        vr     = f"{price['volume_ratio']:.1f}x" if price.get("volume_ratio") is not None else "?"
        sub_str = f" [{subsector}]" if subsector else ""
        lines.append(
            f"- **{ticker}** {name}{sub_str} ({market}): "
            f"close={close}, 1d={r1d}, 5d={r5d}, vol={vr}"
        )

    return "\n".join(lines) if len(lines) > 1 else ""


def _web_sector_analyze(sector_query: str, markets: list, date: str,
                        provider: str, api_key: str, web_search_fn,
                        data_dir: str = None) -> str:
    """
    Sector analysis when no direct local data matches the query.
    Flow:
      1. Web search → discover relevant stocks and context
      2. LLM planner: given web results + sector query, decide what local data to fetch
      3. LLM synthesizes web context + local pipeline data (hybrid)
    """
    market_hint = ", ".join(markets) if markets else "global"
    queries = [
        f"{sector_query} sector stocks {market_hint} {date}",
        f"{sector_query} 板块 行情 {date}" if any('\u4e00' <= c <= '\u9fff' for c in sector_query)
        else f"{sector_query} industry outlook top companies",
    ]
    raw_results = []
    all_urls = []
    for q in queries:
        r = web_search_fn(q, max_results=5)
        if "(no web results)" not in r and "(web search failed" not in r:
            raw_results.append(r)
            all_urls += [l.strip() for l in r.split("\n") if l.strip().startswith("http")]

    if not raw_results:
        return f"No data found for **'{sector_query}'** sector — neither local pipeline nor web search returned results."

    combined_web = "\n\n".join(raw_results)[:2500]

    # ── Step 2: LLM planner decides what local data to fetch ─────────────────
    local_enrichment = ""
    if data_dir:
        try:
            from web.agent_llm import _web_to_local_enrichment
        except ImportError:
            from agent_llm import _web_to_local_enrichment
        local_enrichment = _web_to_local_enrichment(
            combined_web, f"sector analysis: {sector_query}",
            data_dir, markets[0] if len(markets) == 1 else "ALL",
            provider, api_key
        )

    # ── Step 3: hybrid LLM analysis ──────────────────────────────────────────
    data_section = f"WEB SEARCH RESULTS:\n{combined_web}"
    if local_enrichment:
        data_section += f"\n\n{local_enrichment}"
        source_label = "live web search + local pipeline"
    else:
        source_label = "live web search"

    system = (
        "You are a sector analyst. You have been given web search results and "
        "(where available) real-time local pipeline data for stocks in this sector. "
        "Prioritize the local pipeline data (prices, volume) for performance analysis; "
        "use web results for context, themes, and catalysts. "
        "Cover: recent performance, key stocks, main themes/catalysts, bull case, bear case, outlook. "
        "Be concise and beginner-friendly. Use markdown headers."
    )
    user = (
        f"Sector: {sector_query}  Markets: {market_hint}  Date: {date}\n\n"
        f"{data_section}\n\n"
        "Write a structured sector analysis (max 450 words). "
        "If local pipeline data is present, cite specific price moves."
    )
    analysis = _call_llm(system, user, provider, api_key, max_tokens=700)

    sources_footer = ""
    if all_urls:
        sources_footer = "\n\n---\n**🔍 Sources:**\n" + "\n".join(f"- {u}" for u in all_urls[:6])

    return (
        f"# Sector Analysis: {sector_query}\n"
        f"*{market_hint} · {date} · source: {source_label}*\n\n"
        f"{analysis}"
        f"{sources_footer}"
    )


def local_sector_analyze(sector_query: str, market: str, data_dir: str) -> str:
    """
    Multi-agent sector analysis using local pipeline data.

    sector_query: user-provided sector name (e.g. "tech", "半导体", "finance")
    market:       market code (CN, US, HK, …) or "ALL" for cross-market
    """
    try:
        from web.agent_llm import get_llm_provider, _web_search
    except ImportError:
        from agent_llm import get_llm_provider, _web_search

    provider, api_key = get_llm_provider()
    if not provider:
        return "⚙ LLM API key not configured."

    date = __import__("datetime").datetime.now().strftime("%Y-%m-%d")

    # Gather data for requested market(s)
    markets_to_run = []
    if market in (None, "ALL", "global"):
        # Try all markets that have universe data
        for m in ["CN", "US", "HK", "JP", "KR", "TW", "IN", "UK", "DE", "FR", "AU", "BR", "SA"]:
            uni_path = os.path.join(data_dir, "markets", m, "universe.parquet")
            if os.path.exists(uni_path):
                markets_to_run.append(m)
    else:
        markets_to_run = [market]

    all_data = {}
    for m in markets_to_run:
        d = _gather_sector_data(sector_query, m, data_dir)
        if "error" not in d and d.get("stock_count", 0) > 0:
            all_data[m] = d

    if not all_data:
        # No direct local sector match — web search + cross-reference local pipeline
        return _web_sector_analyze(sector_query, markets_to_run, date, provider, api_key, _web_search, data_dir=data_dir)

    # Build compact data summary for LLM (keep under token limit)
    data_sections = []
    for m, d in all_data.items():
        s = d["summary"]
        # Show whether match was on sector or subsector level
        match_info = d.get("matched_sectors") or d.get("matched_subsectors", [])
        match_src  = d.get("match_source", "sector")
        lines = [f"**{m}** — {d['stock_count']} stocks matched via {match_src}: {match_info[:5]}"]
        if s:
            avg1 = s.get("avg_return_1d")
            avg5 = s.get("avg_return_5d")
            if avg1 is not None: lines.append(f"  Avg 1d return: {avg1:+.2%}  ({s['up_count']} up / {s['down_count']} down)")
            if avg5 is not None: lines.append(f"  Avg 5d return: {avg5:+.2%}")
            if s.get("high_volume_count"): lines.append(f"  High volume stocks: {s['high_volume_count']}")
        if d["top_gainers"]:
            g = d["top_gainers"]
            lines.append(f"  Top gainers: " + ", ".join(
                f"{r.get('name', r['ticker'])} {r['return_1d']:+.1%}" for r in g[:3] if r.get('return_1d') is not None
            ))
        if d["top_losers"]:
            l = d["top_losers"]
            lines.append(f"  Top losers:  " + ", ".join(
                f"{r.get('name', r['ticker'])} {r['return_1d']:+.1%}" for r in l[:3] if r.get('return_1d') is not None
            ))
        data_sections.append("\n".join(lines))

    # News across all markets
    all_news = []
    for d in all_data.values():
        all_news.extend(d.get("news", []))
    all_alerts = []
    for d in all_data.values():
        all_alerts.extend(d.get("alerts", []))

    news_text = _fmt(all_news[:10])[:800] if all_news else ""

    # Web search for sector news if no local news — collect sources
    web_sources_footer = ""
    if not all_news:
        search_q = f"{sector_query} sector stocks news {date}"
        web_news = _web_search(search_q, max_results=5)
        news_text = web_news[:1200]
        # Extract source URLs for footer
        news_urls = [l.strip() for l in web_news.split("\n") if l.strip().startswith("http")]
        if news_urls:
            web_sources_footer = "\n\n---\n**🔍 Sources:**\n" + "\n".join(f"- {u}" for u in news_urls[:5])

    data_summary = "\n\n".join(data_sections)

    def _step(label, fn, *args):
        try:
            r = fn(*args)
            return r or f"({label}: no output)"
        except Exception as e:
            return f"({label} unavailable: {e})"

    # Agent 1: Performance analyst
    perf = _step("Performance", _call_llm,
        "You are a sector performance analyst. Analyze the sector's price performance, "
        "identify leaders and laggards, and assess overall sector momentum. Be concise and beginner-friendly.",
        f"Sector: {sector_query}  Date: {date}\n\n{data_summary}\n\nAnalyze sector performance (max 200 words).",
        provider, api_key, 400)

    # Agent 2: News/sentiment analyst
    news_sys = (
        "You are a sector news and sentiment analyst. Identify key themes, "
        "catalysts, and overall market sentiment for this sector."
    )
    news_user = (
        f"Sector: {sector_query}\n\nNEWS:\n{news_text or '(none)'}\n\n"
        f"ALERTS: {json.dumps(all_alerts[:8], ensure_ascii=False, default=str)[:400]}\n\n"
        "Analyze sector sentiment (max 150 words)."
    )
    sentiment = _step("Sentiment", _call_llm, news_sys, news_user, provider, api_key, 350)

    # Agent 3: Bull case
    bull = _step("Bull", _call_llm,
        "You are a bull researcher. Make the bull case for investing in this sector now. "
        "3 bullet points, grounded in the data.",
        f"Sector: {sector_query}\n\nPerformance:\n{perf[:400]}\n\nSentiment:\n{sentiment[:300]}\n\n"
        "Bull case (max 120 words).",
        provider, api_key, 250)

    # Agent 4: Bear case
    bear = _step("Bear", _call_llm,
        "You are a bear researcher. Make the bear case against this sector. "
        "3 bullet points, grounded in the data.",
        f"Sector: {sector_query}\n\nPerformance:\n{perf[:400]}\n\nSentiment:\n{sentiment[:300]}\n\n"
        "Bear case (max 120 words).",
        provider, api_key, 250)

    # Agent 5: Sector strategist (final verdict)
    verdict = _step("Strategist", _call_llm,
        "You are a sector strategist. Give a concise sector outlook. Format:\n"
        "**Outlook: Bullish / Neutral / Bearish**\n"
        "**Key driver:** 1 sentence.\n"
        "**Stocks to watch:** 2-3 names from the data.\n"
        "**Beginner takeaway:** 1 plain-language sentence.",
        f"Sector: {sector_query}  Markets: {list(all_data.keys())}\n\n"
        f"Performance:\n{perf[:350]}\nBull: {bull[:200]}\nBear: {bear[:200]}\n\n"
        "Final sector outlook.",
        provider, api_key, 300)

    market_label = list(all_data.keys())
    lines = [
        f"# Sector Analysis: {sector_query.title()}",
        f"*Markets: {', '.join(market_label)} · {date}*",
        "",
        "---",
        "## 📊 Performance Overview",
        data_summary,
        "",
        "## 📈 Trend Analysis",
        perf,
        "",
        "## 📰 News & Sentiment",
        sentiment,
        "",
        "## 🐂 Bull Case",
        bull,
        "",
        "## 🐻 Bear Case",
        bear,
        "",
        "---",
        "## 🏁 Sector Outlook",
        verdict,
    ]
    if web_sources_footer:
        lines.append(web_sources_footer)
    return "\n".join(lines)
