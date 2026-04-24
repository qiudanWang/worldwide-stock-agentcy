"""LLM-backed agent chat — each agent loads its data and uses an LLM to answer."""

import os
import re
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
    """Return one row per ticker (most recent date with valid close) with computed return signals.

    Reads the last 2 parquet files so a partial today-run (NaN close) doesn't
    shadow yesterday's complete data. Keeps only the latest row per ticker
    that has a valid close price.
    """
    import glob
    files = sorted(glob.glob(os.path.join(base, "market_daily_*.parquet")))
    if not files:
        return pd.DataFrame()
    try:
        frames = []
        for f in files[-2:]:
            try:
                frames.append(pd.read_parquet(f))
            except Exception:
                pass
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
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

        # Prefer rows with valid close so today's NaN rows don't shadow yesterday's data
        if "close" in df.columns and "date" in df.columns:
            df_valid = df[df["close"].notna()].sort_values("date")
            return df_valid.drop_duplicates("ticker", keep="last") if not df_valid.empty else _latest_per_ticker(df)
        return _latest_per_ticker(df)
    except Exception:
        return pd.DataFrame()


def _load_ticker_context(ticker: str, market: str, data_dir: str) -> str:
    """Focused context for a single ticker — used when question is about one company.
    Much smaller than _load_market_context so the whole token budget goes to this stock.
    """
    base = os.path.join(data_dir, "markets", market)
    sections = []
    names = _name_lookup(base)

    # Price/volume for this ticker only
    snap = _latest_daily_parquet(base)
    if not snap.empty and "ticker" in snap.columns:
        row = snap[snap["ticker"] == ticker]
        if not row.empty:
            row = _merge_names(row, names)
            # Format return columns as % strings so LLM doesn't misread decimals
            row = row.copy()
            for ret_col in ["return_1d", "return_5d", "return_20d"]:
                if ret_col in row.columns:
                    row[ret_col] = row[ret_col].apply(
                        lambda v: f"{v*100:+.2f}%" if pd.notna(v) else None)
            cols = [c for c in ["ticker", "name", "close", "return_1d", "return_5d",
                                "return_20d", "volume", "volume_ratio"] if c in row.columns]
            sections.append(f"PRICE DATA ({ticker}):\n" + _df_to_text(row[cols]))

    # Fundamentals from cache (if available)
    fund_path = os.path.join(base, "fundamentals.parquet")
    if os.path.exists(fund_path):
        try:
            fund = pd.read_parquet(fund_path)
            frow = fund[fund["ticker"].astype(str) == ticker]
            if not frow.empty:
                frow = frow.drop(columns=["ticker", "market"], errors="ignore")
                sections.append(f"FUNDAMENTALS ({ticker}):\n" + _df_to_text(frow))
        except Exception:
            pass

    # Indices for context
    idx = _load_parquet(os.path.join(base, "indices.parquet"))
    if not idx.empty:
        sections.append(f"MARKET INDICES:\n" + _df_to_text(idx, 5))

    # News for this ticker
    news = _load_parquet(os.path.join(base, "news.parquet"), max_rows=50)
    if not news.empty and "ticker" in news.columns:
        tnews = news[news["ticker"] == ticker]
        if not tnews.empty:
            cols = [c for c in ["title", "publisher", "hit_count"] if c in tnews.columns]
            sections.append(f"NEWS ({ticker}):\n" + _df_to_text(tnews[cols], 8))

    # Sector peers — find same-sector companies from universe and show their recent returns
    uni_path = os.path.join(base, "universe.parquet")
    if os.path.exists(uni_path):
        try:
            uni = pd.read_parquet(uni_path)
            row_uni = uni[uni["ticker"].astype(str) == ticker]
            if not row_uni.empty and "sector" in uni.columns:
                sector = row_uni.iloc[0]["sector"]
                peers = uni[(uni["sector"] == sector) & (uni["ticker"].astype(str) != ticker)]
                if not peers.empty and not snap.empty and "ticker" in snap.columns:
                    peer_tickers = peers["ticker"].astype(str).tolist()
                    peer_snap = snap[snap["ticker"].astype(str).isin(peer_tickers)].copy()
                    if not peer_snap.empty:
                        peer_snap = _merge_names(peer_snap, names)
                        # Format returns as % strings (consistent with PRICE DATA block)
                        for ret_col in ["return_1d", "return_20d"]:
                            if ret_col in peer_snap.columns:
                                peer_snap[ret_col] = peer_snap[ret_col].apply(
                                    lambda v: f"{v*100:+.2f}%" if pd.notna(v) else None)
                        cols = [c for c in ["ticker", "name", "close", "return_1d",
                                            "return_20d", "market_cap"] if c in peer_snap.columns]
                        # Sort by market_cap desc, show top 10
                        if "market_cap" in peer_snap.columns:
                            peer_snap = peer_snap.sort_values("market_cap", ascending=False)
                        sections.append(
                            f"SECTOR PEERS ({sector}, top {min(10, len(peer_snap))}):\n"
                            + _df_to_text(peer_snap[cols], 10)
                        )
        except Exception:
            pass

    return "\n\n".join(sections) if sections else _load_data_context(market, data_dir)


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
        cols = [c for c in ["ticker", "name", "close", "return_1d", "return_5d", "return_20d",
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
    """Load all data relevant to the Global Strategist.

    Ordering is intentional: highest-signal sections first so they survive the
    character-cap truncation. Bulky low-density sections (alerts, peers, geo) go last.
    """
    gdir = os.path.join(data_dir, "global")
    sections = []

    # ── 1. Capital flows (always first — most commonly queried, small footprint) ──
    import glob as _glob
    flow_summaries = []
    for flow_path in sorted(_glob.glob(os.path.join(data_dir, "markets", "*", "capital_flow.parquet"))):
        try:
            df = pd.read_parquet(flow_path)
            if df.empty:
                continue
            market_code = flow_path.split(os.sep)[-2]
            flow_col = "net_flow" if "net_flow" in df.columns else "net_flow_proxy"
            if flow_col not in df.columns:
                continue
            if "date" in df.columns:
                df = df.sort_values("date")
                # Drop weekend rows — ETF proxies on Sat/Sun are non-trading noise
                df = df[pd.to_datetime(df["date"]).dt.dayofweek < 5]
            recent = df.tail(10)
            values = recent[flow_col].dropna()
            if values.empty:
                continue
            last_date = str(recent["date"].iloc[-1])[:10] if "date" in recent.columns else "n/a"
            last_val  = round(float(values.iloc[-1]), 2)
            avg_5d    = round(float(values.tail(5).mean()), 2)
            total_10d = round(float(values.sum()), 2)
            direction = "INFLOW ▲" if avg_5d > 0 else "OUTFLOW ▼"
            unit = "亿CNY northbound" if market_code == "CN" else "USD ETF-proxy"
            # Warn if data is stale (>30 days old)
            try:
                days_old = (pd.Timestamp.today() - pd.Timestamp(last_date)).days
                stale_note = f"  ⚠ DATA STALE ({days_old}d old — CN exchanges discontinued northbound flow reporting after Aug 2024)" if days_old > 30 else ""
            except Exception:
                stale_note = ""
            flow_summaries.append(
                f"  {market_code}: {direction}  last={last_val} {unit}  "
                f"5d-avg={avg_5d}  10d-total={total_10d}  (as of {last_date}){stale_note}"
            )
        except Exception:
            pass
    if flow_summaries:
        note = ("+ = inflow (foreign money entering), - = outflow (foreign money leaving). "
                "CN = Stock Connect northbound. Others = country ETF proxy. "
                "IMPORTANT: Always state the 'as of' date when citing these numbers — "
                "the user's UI may show a different trading day.")
        sections.append(
            "CAPITAL FLOWS BY MARKET:\n" + note + "\n" + "\n".join(flow_summaries)
        )

    # ── 2. Macro snapshot (compact: top 15 indicators only) ──
    macro = _load_json(os.path.join(gdir, "macro_latest.json"))
    if macro:
        top = dict(list(macro.items())[:15])
        sections.append("MACRO INDICATORS (top 15):\n" + json.dumps(top, indent=1, default=str))

    # ── 3. Sector performance (compact) ──
    sec = _load_parquet(os.path.join(gdir, "sector_performance.parquet"))
    if not sec.empty:
        sections.append(f"SECTOR PERFORMANCE:\n" + _df_to_text(sec, 20))

    # ── 4. Correlations ──
    corr = _load_json(os.path.join(gdir, "correlations.json"))
    if corr:
        sections.append("CORRELATIONS:\n" + json.dumps(corr, indent=1, default=str)[:1000])

    # ── 5. Geopolitical news (most recent 8 items) ──
    geo = _load_json(os.path.join(gdir, "geopolitical_context.json"))
    if geo:
        items = geo if isinstance(geo, list) else geo.get("items", [])
        if items:
            sections.append(f"GEOPOLITICAL NEWS:\n" + json.dumps(items[:8], indent=1, default=str)[:1500])

    # ── 6. Alerts (small sample) — lowest priority, most verbose ──
    alerts = _load_json(os.path.join(gdir, "alerts.json"))
    if alerts and isinstance(alerts, list):
        sections.append(f"GLOBAL ALERTS (sample):\n" + json.dumps(alerts[:10], indent=1, default=str)[:1000])

    return "\n\n".join(sections) if sections else "No global data yet. Run the pipeline first."


# ── Tool registry ────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "local_data": {
        "description": "Local OHLCV prices, returns (1d/5d/20d), volume ratios, market cap, sector/industry classification from most recent pipeline run",
        "when_to_use": "Always include as a step when the question involves stocks in our local universe — even when web_search covers revenue/profit. Provides price performance and sector context that complements web financial data. Use for: market overview, ranking by return/volume, single-stock price history, or as a complement alongside web_search.",
        "input": {"ticker": "optional — omit for market-wide data, provide for single stock"},
        "output": "Price, returns, volume ratio, market cap table; sector/industry info; or single-stock OHLCV",
        "constraints": "Data is from last pipeline run — not real-time. No PE/EPS — use web_search or openbb_fundamentals for those.",
    },
    "local_news": {
        "description": "Local scraped news articles with keyword relevance scores",
        "when_to_use": "User asks about news, headlines, or recent articles for a stock or market",
        "input": {"ticker": "optional"},
        "output": "News titles, publisher, hit count, keywords matched",
        "constraints": "Only covers news scraped in the last pipeline run. Not real-time.",
    },
    "local_peers": {
        "description": "Find peer companies in the same industry/sub-sector from local universe — includes price performance (1d/20d returns), volume ratio, and latest annual + quarterly revenue/net profit (亿元, YoY%) from local financials cache",
        "when_to_use": "User asks about peers, competitors, or similar companies for a stock. Always include for peer_comparison — provides peer list with price AND financials from local data. Add web_search only when forward estimates or analyst targets are also needed.",
        "input": {"ticker": "required"},
        "output": "Peer list with ticker, name, sector, industry, close price, 1d/20d returns, vol_ratio, latest annual revenue/NP (YoY%), latest quarterly revenue/NP (YoY%)",
        "constraints": "Only covers stocks in local universe. Financials data available after quarterly pipeline refresh — may be stale for very recent reports.",
    },
    "local_signals": {
        "description": "Local alerts: volume spikes, capital flow anomalies, unusual activity signals",
        "when_to_use": "User asks about unusual activity, alerts, or anomalies",
        "input": {},
        "output": "Alert list with ticker, signal type, magnitude",
        "constraints": "CN market only. Based on last pipeline run.",
    },
    "openbb_gainers": {
        "description": "Live top gaining stocks today",
        "when_to_use": "User asks for top gainers or best performing stocks today",
        "input": {},
        "output": "List of top gaining tickers with % change",
        "constraints": "US STOCKS ONLY. Never use for CN/HK/JP/KR/TW/IN/UK/DE/FR/AU/BR/SA markets.",
    },
    "openbb_losers": {
        "description": "Live top losing stocks today",
        "when_to_use": "User asks for top losers or worst performing stocks today",
        "input": {},
        "output": "List of top losing tickers with % change",
        "constraints": "US STOCKS ONLY. Never use for non-US markets.",
    },
    "openbb_active": {
        "description": "Live most actively traded stocks by volume today",
        "when_to_use": "User asks for most active or highest volume stocks today",
        "input": {},
        "output": "List of most active tickers with volume",
        "constraints": "US STOCKS ONLY. Never use for non-US markets.",
    },
    "openbb_indices": {
        "description": "Live major global indices: S&P 500, Nasdaq, Nikkei, FTSE, Hang Seng, CSI 300, etc.",
        "when_to_use": "User asks about market indices, global markets, or macro overview. Use for any market.",
        "input": {},
        "output": "Index name, current level, % change today",
        "constraints": "Index-level only — no individual stock data.",
    },
    "openbb_quote": {
        "description": "Live price quote for a specific stock — works for CN (A-share), HK, US, JP and other markets via yfinance",
        "when_to_use": "User asks for current price, today's change, or live quote of a specific stock. Use for CN A-share tickers too — provide ticker with exchange suffix.",
        "input": {"ticker": "required — exchange-suffixed: 603881.SS (Shanghai 6xxxxx), 300442.SZ (Shenzhen 0/3xxxxx), 0700.HK, AAPL"},
        "output": "Current price, % change today, volume, market cap",
        "constraints": "One ticker per call. CN coverage via yfinance — may have slight delay.",
    },
    "openbb_news": {
        "description": "Latest news articles for a specific company via financial data APIs",
        "when_to_use": "User asks for recent news about a specific named company or ticker",
        "input": {"ticker": "required"},
        "output": "News headline, source, date, summary",
        "constraints": "Requires a ticker. Coverage varies by market — US coverage best.",
    },
    "openbb_history": {
        "description": "30-day price history for a specific stock",
        "when_to_use": "User asks about price trend, chart, or historical performance over weeks",
        "input": {"ticker": "required — exchange-suffixed for non-US"},
        "output": "Daily OHLCV for last 30 days",
        "constraints": "Requires a ticker. 30-day window only.",
    },
    "openbb_profile": {
        "description": "Company profile: sector, industry, business description, key stats",
        "when_to_use": "User asks what a company does, its sector/industry, or wants a company overview",
        "input": {"ticker": "required"},
        "output": "Sector, industry, description, employee count, HQ location",
        "constraints": "Requires a ticker. Coverage varies — US best.",
    },
    "openbb_fundamentals": {
        "description": "Financial ratios and TTM financials: P/E, EPS, revenue, profit margins, ROE, debt/equity",
        "when_to_use": "User asks about fundamentals, valuation, earnings, revenue, or financial health of a specific company. Use for peer comparison when ticker is known.",
        "input": {"ticker": "required — exchange-suffixed: 600519.SS (Shanghai), 000001.SZ (Shenzhen), 0700.HK, AAPL"},
        "output": "Revenue TTM, net income, P/E trailing/forward, EPS, profit margin, gross margin, ROE, debt/equity, free cash flow",
        "constraints": "Requires a ticker. For CN stocks use .SS (6xxxxx) or .SZ (0xxxxx/3xxxxx) suffix. Coverage: US best, CN/HK partial via yfinance.",
    },
    "peer_agent": {
        "description": "Fixed-format peer comparison report: ranks peers by 20d momentum, flags volume activity, summarizes financials (revenue/NP annual + quarterly, YoY%), gives relative strength verdict for the anchor stock vs its industry peers",
        "when_to_use": "User asks to compare a stock with peers, competitors, or similar companies. Always use for a full analysis report — peer_agent calls local_peers internally and adds LLM ranking, financials summary, and verdict.",
        "input": {"ticker": "required — the anchor stock to compare against its peers"},
        "output": "Peer table with price + financials, relative strength ranking (strongest→weakest), volume activity, financials comparison (revenue/NP with YoY%), verdict (Leader/Middle/Laggard)",
        "constraints": "Only covers stocks in local universe. Financials shown when available from quarterly pipeline refresh.",
    },
    "trading_agent": {
        "description": "Deep multi-agent analysis: fundamentals + sentiment + technicals + bull/bear debate",
        "when_to_use": "User explicitly asks to analyze, deep-dive, or get a full report on a specific stock. Slow (30-60s).",
        "input": {"ticker": "required — US ticker (e.g. NVDA) or CN 6-digit code (e.g. 688031)"},
        "output": "Full analysis: price, fundamentals, sentiment, technical signals, bull/bear arguments, outlook",
        "constraints": "One stock only. Slow. Do not use for peer comparison or market overview.",
    },
    "sector_agent": {
        "description": "Multi-agent sector analysis: performance, sentiment, bull/bear debate for a whole sector",
        "when_to_use": "User asks to analyze a sector, industry, or market segment (e.g. semiconductors, tech sector, healthcare)",
        "input": {"ticker": "sector name as ticker field, e.g. 'semiconductors' or 'technology'"},
        "output": "Sector performance, top stocks, sentiment, outlook",
        "constraints": "Sector-level only — not for individual stocks.",
    },
    "web_search": {
        "description": "Search the internet for real-time news, recent financials, analyst estimates, or any information not in local/OpenBB data",
        "when_to_use": "Use when: (1) data needed is more recent than last pipeline run, (2) user asks about 2025/2026 revenue/profit results or analyst estimates, (3) local/OpenBB tools lack coverage. Do NOT use for peer discovery — use local_peers instead.",
        "input": {"query": "search query string"},
        "output": "Web snippets: title, body text, URL for each result",
        "constraints": "Results quality varies. CN queries: use 'cn-zh' region. Non-CN: use 'wt-wt'.",
    },
    "fetch_url": {
        "description": "Fetch and read the full text content of a specific URL (news article, announcement, financial report)",
        "when_to_use": "User pastes a URL in their message — always fetch it as the first step",
        "input": {"url": "the URL from the user's message"},
        "output": "Full article text, cleaned of HTML",
        "constraints": "One URL per call. Some sites block crawlers.",
    },
}

# Backward compat — existing code references TOOLS for name validation
TOOLS = {k: v["description"] for k, v in TOOL_REGISTRY.items()}


def _registry_for_llm() -> str:
    """Format TOOL_REGISTRY as a compact string for LLM prompts."""
    lines = []
    for name, spec in TOOL_REGISTRY.items():
        inp = ", ".join(f"{k}: {v}" for k, v in spec["input"].items()) or "none"
        lines.append(
            f"[{name}]\n"
            f"  desc: {spec['description']}\n"
            f"  use when: {spec['when_to_use']}\n"
            f"  input: {inp}\n"
            f"  output: {spec['output']}\n"
            f"  constraints: {spec['constraints']}"
        )
    return "\n\n".join(lines)



# ── Two-phase planner ─────────────────────────────────────────────────

def _call_llm(system, history, message, max_tokens=None):
    """Unified LLM call. Reads provider/api_key from config — no need to pass them around."""
    provider, api_key = get_llm_provider()
    if provider == "anthropic":
        return _call_anthropic(api_key, system, history, message, max_tokens or 8096)
    return _call_openai(api_key, system, history, message, max_tokens)


def _plan_message(message: str, market, agent_type: str,
                  chat_history: list = None) -> dict:
    """Phase 1: LLM understands the question and produces a structured execution plan.

    Returns dict:
      intent: single_stock | peer_comparison | sector_analysis | market_overview | macro | news | general
      tickers: list of explicit ticker codes from the question
      company_name: str
      steps: list[{id, tool, ticker?, url?, query?, query_template?, depends_on?, purpose}]
      response_focus: precise instruction for the final synthesis LLM call
    """
    tools_list = _registry_for_llm()

    # Include last 2 turns so planner can resolve references like "it", "its peers", "that company"
    history_msgs = []
    if chat_history:
        for msg in chat_history[-2:]:
            role = "assistant" if msg.get("role") == "agent" else "user"
            history_msgs.append({"role": role, "content": msg["content"][:400]})

    planner_system = (
        "You are a financial research planner. "
        "Output ONLY valid JSON — no markdown fences, no explanation. "
        "Use the conversation history to resolve references like 'it', 'its peers', or 'that company'."
    )
    planner_user = f"""Market: {market or "global"}  Agent: {agent_type}
Question: {message}

Available tools:
{tools_list}

Output a JSON execution plan:
{{
  "intent": "<single_stock | peer_comparison | sector_analysis | market_overview | macro | news | general>",
  "tickers": ["<ticker codes the user explicitly typed>"],
  "company_name": "<company name if user mentioned one, else empty>",
  "steps": [
    {{
      "id": 1,
      "tool": "<tool name>",
      "ticker": "<for openbb/local tools>",
      "url": "<for fetch_url>",
      "query": "<for web_search — fixed query>",
      "query_template": "<for web_search steps that depend on a prior step — write query with {{peer_companies}} where the prior step's discovered company names will be inserted>",
      "depends_on": [],
      "purpose": "<one sentence>"
    }}
  ],
  "response_focus": "<what the synthesizer should focus on — specific metric, comparison, conclusion>"
}}

Rules:
- Tool "use when" and "constraints" fields are the single source of truth — read them carefully to choose tools.
- Only include fields relevant to the tool (e.g. skip url/query/ticker if not needed).
- tickers: ONLY codes the user explicitly typed. Never infer from company names.
- URL in message: fetch_url is always step 1. Other steps follow after.
- Peer comparison — two cases:
  (a) User asks "find peers / competitors of stock X" (unknown peer list): use peer_agent with X as anchor — it discovers and ranks peers automatically.
  (b) User names N specific stocks for direct comparison: use local_data (list each ticker in the ticker field) + web_search. Do NOT use peer_agent here — it is designed to discover unknown peers, not compare an explicit named list.
- Always include a tool even if coverage is partial — partial data is better than nothing.
- Max 4 steps."""

    try:
        raw = _call_llm(planner_system, history_msgs, planner_user)

        import re as _re
        match = _re.search(r'\{.*\}', raw, _re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            steps = [s for s in parsed.get("steps", []) if isinstance(s, dict) and s.get("tool") in TOOLS]
            if steps or parsed.get("intent"):
                parsed["steps"] = steps
                print(
                    f"[planner] intent={parsed.get('intent')} "
                    f"steps={[s['tool'] for s in steps]} "
                    f"focus={parsed.get('response_focus', '')[:80]}",
                    flush=True,
                )
                return parsed
    except Exception as e:
        print(f"[planner] failed: {e}", flush=True)

    return {}


def _extract_peer_names_llm(text: str) -> list[str]:
    """Use LLM to extract company names from a step output for use in chained queries."""
    prompt = (
        "Extract all company names from the text below. "
        "Return only the names, one per line, no explanations, no numbering. "
        "Include both English and Chinese names as they appear.\n\n"
        f"{text[:2000]}"
    )
    try:
        raw = _call_llm("You extract company names from text.", [], prompt)
        names = [n.strip() for n in raw.strip().splitlines() if n.strip()]
        return names[:8]
    except Exception:
        return []


def _execute_plan(plan: dict, tickers: list, market, agent_type: str,
                  data_dir: str, message: str) -> dict:
    """Phase 2: Execute plan steps. Independent steps (no depends_on) run in parallel;
    dependent steps run sequentially after their deps complete.

    Handles all tools including trading_agent and sector_agent.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import re as _re

    tool_results: dict = {}
    step_outputs: dict = {}          # step_id → raw result text
    is_global = agent_type == "global" or market in (None, "ALL")

    def _resolve_query(step, dep_ids):
        query = step.get("query") or ""
        tmpl  = step.get("query_template") or ""
        if tmpl:
            placeholder = "{peer_companies}" if "{peer_companies}" in tmpl else "{peer_names}" if "{peer_names}" in tmpl else ""
            if dep_ids and placeholder:
                dep_text = " ".join(step_outputs.get(d, "") for d in dep_ids)
                names = _extract_peer_names_llm(dep_text)
                sub = " ".join(names) if names else message[:60]
                return tmpl.replace(placeholder, sub)
            return tmpl
        return query

    def _run_step(step):
        step_id = step.get("id", 0)
        tool    = step.get("tool", "")
        dep_ids = step.get("depends_on") or []
        query   = _resolve_query(step, dep_ids)
        result  = ""

        if tool == "fetch_url":
            url = step.get("url") or ""
            if not url:
                urls = _re.findall(r'https?://\S+', message)
                url = urls[0].rstrip(".,)") if urls else ""
            result = _fetch_url(url) if url else "(no URL found in message)"

        elif tool == "local_data":
            ticker = step.get("ticker") or (tickers[0] if tickers else None)
            if is_global:
                result = _load_global_context(data_dir)
            elif tickers and len(tickers) > 1:
                # Multiple tickers: load each and concatenate
                parts = []
                for t in tickers:
                    ctx = _load_ticker_context(t, market, data_dir)
                    if ctx and not ctx.startswith("("):
                        parts.append(ctx)
                result = "\n\n---\n\n".join(parts) if parts else _load_data_context(market, data_dir)
            elif ticker:
                result = _load_ticker_context(ticker, market, data_dir)
            else:
                result = _load_data_context(market, data_dir)

        elif tool == "web_search":
            intent  = plan.get("intent", "")
            company = plan.get("company_name", "")
            is_cn   = market == "CN" or any(_re.match(r'^\d{6}$', t) for t in tickers)
            search_region = "cn-zh" if is_cn else "wt-wt"
            if not query:
                if intent == "peer_comparison" and company:
                    query = f"{company} A-share competitors China server computing AI 算力 2025" if not dep_ids \
                            else f"{company} 竞争对手 浪潮信息 算力 A股 2025 2026 营收"
                elif tickers:
                    query = f"{tickers[0]} {company} 2025 2026 revenue 营收 净利润 annual results"
                else:
                    query = _build_macro_search_query(message, market)
            res = _web_search(query, max_results=5, region=search_region)
            result = res

        elif tool == "trading_agent":
            ticker = step.get("ticker") or (tickers[0] if tickers else None)
            if ticker:
                try:
                    from web.local_trading_agent import detect_market, local_trading_analyze
                except ImportError:
                    from local_trading_agent import detect_market, local_trading_analyze
                result = trading_agents_analyze(ticker) if detect_market(ticker) == "US" \
                         else local_trading_analyze(ticker, data_dir)

        elif tool == "peer_agent":
            ticker = step.get("ticker") or (tickers[0] if tickers else None)
            if ticker:
                try:
                    from web.local_trading_agent import local_peers_analyze, detect_market
                except ImportError:
                    from local_trading_agent import local_peers_analyze, detect_market
                peer_market = market or detect_market(ticker)
                result = local_peers_analyze(ticker, peer_market, data_dir) if peer_market else "(peer_agent: could not detect market)"
            else:
                result = "(peer_agent: no ticker)"

        elif tool == "sector_agent":
            sector_query = step.get("ticker") or (tickers[0] if tickers else "")
            if sector_query:
                try:
                    from web.local_trading_agent import local_sector_analyze
                except ImportError:
                    from local_trading_agent import local_sector_analyze
                result = local_sector_analyze(sector_query, market, data_dir)

        elif tool == "local_news":
            ticker = step.get("ticker") or (tickers[0] if tickers else None)
            if ticker and market:
                try:
                    try:
                        from web.local_trading_agent import _load_local_news
                    except ImportError:
                        from local_trading_agent import _load_local_news
                    rows = _load_local_news(ticker, market, data_dir)
                    result = "\n".join(
                        f"- {r.get('title','')} [{r.get('publisher','')}] hits={r.get('hit_count','')}"
                        for r in rows
                    ) if rows else "(no local news found)"
                except Exception as e:
                    result = f"(local_news error: {e})"
            else:
                result = "(local_news: ticker or market missing)"

        elif tool == "local_signals":
            ticker = step.get("ticker") or (tickers[0] if tickers else None)
            if ticker and market:
                try:
                    try:
                        from web.local_trading_agent import _load_local_alerts
                    except ImportError:
                        from local_trading_agent import _load_local_alerts
                    alerts = _load_local_alerts(ticker, market, data_dir)
                    result = "\n".join(str(a) for a in alerts) if alerts else "(no signals found)"
                except Exception as e:
                    result = f"(local_signals error: {e})"
            else:
                result = "(local_signals: ticker or market missing)"

        elif tool == "local_peers":
            ticker = step.get("ticker") or (tickers[0] if tickers else None)
            if ticker:
                try:
                    try:
                        from web.local_trading_agent import _load_local_peers, detect_market
                    except ImportError:
                        from local_trading_agent import _load_local_peers, detect_market
                    peer_market = market or detect_market(ticker)
                    result = _load_local_peers(ticker, peer_market, data_dir) if peer_market else "(local_peers: could not detect market)"
                except Exception as e:
                    result = f"(local_peers error: {e})"
            else:
                result = "(local_peers: no ticker)"

        else:
            # OpenBB tools
            ticker_list = [step["ticker"]] if step.get("ticker") else tickers
            sub = _run_openbb_tools([tool], ticker_list, market, data_dir, message=query or message)
            result = sub.get(tool, "")

        return step_id, tool, result

    def _merge(tool, result):
        """Merge a step result into tool_results (web_search accumulates; others overwrite)."""
        if not result:
            return
        if tool == "web_search":
            existing = tool_results.get("web_search", "")
            tool_results["web_search"] = (existing + "\n\n" + result) if existing else result
        else:
            tool_results[tool] = result

    steps = plan.get("steps", [])
    independent = [s for s in steps if not s.get("depends_on")]
    dependent   = [s for s in steps if s.get("depends_on")]

    # Run independent steps in parallel
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_run_step, s): s for s in independent}
        for fut in as_completed(futures):
            step_id, tool, result = fut.result()
            step_outputs[step_id] = result
            _merge(tool, result)

    # Run dependent steps sequentially (query chaining requires prior step_outputs)
    for step in dependent:
        step_id, tool, result = _run_step(step)
        step_outputs[step_id] = result
        _merge(tool, result)

    return tool_results


def _cn_peer_financials(peer_tickers: list, anchor_ticker: str, data_dir: str) -> str:
    """Fetch revenue/profit for CN A-share peers using akshare + local universe names.

    Uses stock_financial_abstract_ths (same source as trading_agent) to get
    annual revenue and net profit — bypasses unreliable web search for structured data.
    """
    import akshare as ak
    import re as _re

    # Build ticker → name map from universe
    names = {}
    try:
        import pandas as _pd
        uni = _pd.read_parquet(f"{data_dir}/markets/CN/universe.parquet")
        for _, row in uni.iterrows():
            names[str(row["ticker"])] = row.get("name", row["ticker"])
    except Exception:
        pass

    all_tickers = list(dict.fromkeys([anchor_ticker] + peer_tickers))[:8]
    rows = []

    for ticker in all_tickers:
        name = names.get(ticker, ticker)
        try:
            fin = ak.stock_financial_abstract_ths(symbol=ticker, indicator="按年度")
            if fin.empty:
                rows.append(f"{name} ({ticker}): ⚠️ no data")
                continue
            # Columns vary — look for revenue and net profit
            col_map = {c: c for c in fin.columns}
            rev_col    = next((c for c in fin.columns if "营业总收入" in c or "营收" in c), None)
            rev_yoy    = next((c for c in fin.columns if "营业总收入同比" in c), None)
            profit_col = next((c for c in fin.columns if c == "净利润"), None)
            profit_yoy = next((c for c in fin.columns if "净利润同比" in c), None)
            period_col = next((c for c in fin.columns if "报告期" in c), None)

            # tail(3) → most recent 3 years (data sorted oldest-first)
            recent = fin.tail(3)
            lines = [f"\n**{name} ({ticker})**"]
            for _, r in recent.iterrows():
                period = str(r[period_col]) if period_col else "?"
                rev    = r[rev_col]    if rev_col    else "⚠️"
                ry     = r[rev_yoy]    if rev_yoy    else ""
                profit = r[profit_col] if profit_col else "⚠️"
                py     = r[profit_yoy] if profit_yoy else ""
                lines.append(f"  {period}年: 营收={rev} ({ry})  净利润={profit} ({py})")
            rows.append("\n".join(lines))
        except Exception as e:
            rows.append(f"{name} ({ticker}): ⚠️ fetch failed ({e})")

    if not rows:
        return ""
    return "=== CN PEER FINANCIALS (akshare, annual) ===\n" + "\n".join(rows)


# ── Extractor ─────────────────────────────────────────────────────────

_EXTRACT_TRIGGERS = {
    "营收", "净利润", "收益", "业绩", "revenue", "profit", "earnings",
    "financial", "增长", "下滑", "财报", "年报",
}


def _clean_web_text(text: str) -> str:
    """Step 1 of extraction: remove noise from raw web content.

    Strips HTML artifacts, navigation fragments, ads, and repetitive boilerplate
    that survive after fetch/search. Leaves only readable content.
    """
    import re as _re
    import html as _html
    # Unescape HTML entities (&amp; &nbsp; etc.)
    text = _html.unescape(text)
    # Remove residual HTML tags (e.g. from partially-fetched pages)
    text = _re.sub(r'<[^>]{1,200}>', ' ', text)
    # Remove URLs embedded mid-text (keep line-start URLs for source tracking)
    text = _re.sub(r'(?<!\n)\s+https?://\S+', ' ', text)
    # Collapse runs of whitespace / blank lines
    text = _re.sub(r'\n{3,}', '\n\n', text)
    text = _re.sub(r'[ \t]{2,}', ' ', text)
    # Remove common navigation/boilerplate fragments
    _boilerplate = [
        r'版权所有.*?保留', r'copyright.*?reserved', r'cookie.*?policy',
        r'隐私政策', r'服务条款', r'免责声明\s*$', r'广告', r'点击查看',
        r'登录\s*/\s*注册', r'下载APP',
    ]
    for pat in _boilerplate:
        text = _re.sub(pat, '', text, flags=_re.IGNORECASE)
    return text.strip()


def _extract_financials(tool_results: dict, companies_hint: list,
                        message: str) -> str:
    """Relevance-filtered extractor: clean noise → LLM extracts only content relevant to the question.

    Step 1: _clean_web_text strips HTML artifacts and boilerplate.
    Step 2: LLM extracts only content relevant to the user's question — discards off-topic results
            (geopolitics, general news, unrelated company coverage) by LLM judgment, not hardcoded rules.

    Returns "" if the web content contains nothing relevant to the question.
    """
    raw_parts = []
    if tool_results.get("fetch_url"):
        raw_parts.append(_clean_web_text(tool_results["fetch_url"])[:1500])
    if tool_results.get("web_search"):
        raw_parts.append(_clean_web_text(tool_results["web_search"])[:2500])
    if tool_results.get("web_search_missing"):
        raw_parts.append(_clean_web_text(tool_results["web_search_missing"])[:2000])
    if not raw_parts:
        return ""

    raw_text = "\n\n".join(raw_parts)

    extractor_system = "You are a relevance filter and information extractor. Your job is to extract only the content that is directly relevant to the user's question, and discard everything else."
    extractor_prompt = f"""User's question: {message}

Web content to filter:
{raw_text}

Instructions:
- Extract ONLY content that directly answers or informs the user's question (companies, revenue, profit, dates, events related to the question)
- DISCARD entire snippets that are off-topic: geopolitics, unrelated company news, general AI policy, anything not about the specific companies or financials asked about
- Keep all relevant facts, numbers, dates, names exactly as-is — do not paraphrase or add analysis
- If the entire content is off-topic and contains nothing relevant to the question, respond with exactly: NONE
- Use the same language as the source"""

    try:
        raw = _call_llm(extractor_system, [], extractor_prompt)
        raw = raw.strip()
        if not raw or raw == "NONE":
            print(f"[extractor] off-topic — no relevant content found", flush=True)
            return ""

        print(f"[extractor] extracted ({len(raw)} chars)", flush=True)
        return raw

    except Exception as e:
        print(f"[extractor] failed: {e}", flush=True)
        return ""


def _fetch_url(url: str, max_chars: int = 1500) -> str:
    """Fetch and extract readable text from a specific URL (e.g. a news article)."""
    try:
        import urllib.request as _ur
        import html as _html
        import re as _re
        req = _ur.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; StockAgent/1.0)",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        })
        with _ur.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        # Strip HTML tags
        text = _re.sub(r'<script[^>]*>.*?</script>', '', raw, flags=_re.DOTALL | _re.IGNORECASE)
        text = _re.sub(r'<style[^>]*>.*?</style>', '', text, flags=_re.DOTALL | _re.IGNORECASE)
        text = _re.sub(r'<[^>]+>', ' ', text)
        text = _html.unescape(text)
        text = _re.sub(r'\s+', ' ', text).strip()
        return f"[Article from {url}]\n\n{text[:max_chars]}"
    except Exception as e:
        return f"(fetch_url failed for {url}: {e})"


# Domains that indicate DDG returned garbage (off-topic generic sites)
_DDG_GARBAGE_DOMAINS = {
    "wordpress.com", "wordpress.org", "nhs.uk", "imdb.com", "healthcenter.com",
    "wikipedia.org", "amazon.com", "apple.com", "microsoft.com", "google.com",
    "suoxinkj.com", "chinairn.com", "aiqicha.baidu.com",
}

def _web_search(query: str, max_results: int = 6, body_chars: int = 0,
                region: str = "wt-wt") -> str:
    """Search the web via DuckDuckGo. Returns compact text for LLM context.

    body_chars: max chars per result body. 0 = no truncation (take full DDG snippet).
    Note: DDG snippets are already short (~200-500 chars) — truncating them further
    just loses information. Use fetch_url on a specific result to get full article text.
    """
    import re as _re
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, region=region, max_results=max_results):
                title = r.get("title", "")
                body  = r.get("body", "")
                if body_chars > 0:
                    body = body[:body_chars]
                href  = r.get("href", "")
                # Skip garbage domains
                m = _re.match(r'https?://(?:www\.)?([^/]+)', href)
                domain = m.group(1) if m else ""
                if any(domain == g or domain.endswith("." + g) for g in _DDG_GARBAGE_DOMAINS):
                    continue
                results.append(f"• {title}\n  {body}\n  {href}")
        if not results:
            return "(no web results)"
        return f"WEB SEARCH: {query!r}\n" + "\n\n".join(results)
    except Exception as e:
        return f"(web search failed: {e})"


_SOURCE_BLOCKLIST = {
    "google.com", "play.google.com", "accounts.google.com", "photos.google.com",
    "facebook.com", "twitter.com", "x.com", "instagram.com", "tiktok.com",
    "youtube.com", "reddit.com", "pinterest.com", "linkedin.com",
    "apple.com", "microsoft.com", "amazon.com",
    # Spam / off-topic domains seen in DDG results
    "suoxinkj.com", "wordpress.com", "wordpress.org", "nhs.uk", "imdb.com",
    "baijiahao.baidu.com", "aiqicha.baidu.com", "chinairn.com",
}


def _extract_sources(web_result: str) -> str:
    """Extract source URLs and titles from _web_search output. Returns a markdown footer.
    Filters out obviously irrelevant sources (social media, tech company homepages, etc.)."""
    import re as _re
    sources = []
    current_title = ""
    for line in web_result.split("\n"):
        stripped = line.strip()
        if stripped.startswith("•"):
            current_title = stripped[1:].strip()
        elif stripped.startswith("http"):
            url = stripped
            # Extract domain and skip blocklisted ones
            m = _re.match(r'https?://(?:www\.)?([^/]+)', url)
            domain = m.group(1) if m else ""
            if any(domain == b or domain.endswith("." + b) for b in _SOURCE_BLOCKLIST):
                current_title = ""
                continue
            # Skip titles that are clearly off-topic
            _skip_title_kw = ["ebook", "pdf 免费", "gemini for google", "提示词指南", "wordpress"]
            if any(kw in current_title.lower() for kw in _skip_title_kw):
                current_title = ""
                continue
            sources.append((current_title[:70] if current_title else domain, url))
            current_title = ""
    if not sources:
        return ""
    parts = [f"- [{title}]({url})" if title else f"- {url}" for title, url in sources[:6]]
    return "\n\n---\n**🔍 Sources:**\n" + "\n".join(parts)


_MARKET_YF_SUFFIX = {
    "HK": ".HK", "JP": ".T", "KR": ".KS", "TW": ".TW",
    "AU": ".AX", "IN": ".NS", "UK": ".L", "DE": ".DE",
    "FR": ".PA", "BR": ".SA",
}


def _apply_market_suffix(ticker: str, market: str | None) -> str:
    """Append yfinance exchange suffix if the ticker looks bare (no dot yet)."""
    if not market or "." in ticker:
        return ticker
    suffix = _MARKET_YF_SUFFIX.get(market, "")
    return ticker + suffix if suffix else ticker


def _build_macro_search_query(message: str, market: str | None) -> str:
    """Convert a potentially Chinese/mixed financial question into a clean English search query.

    Maps known topic keywords to good English query strings so DuckDuckGo returns
    relevant financial news rather than garbage.
    """
    msg_l = message.lower()

    # Capital flow / money flow
    if any(w in msg_l for w in ["capital flow", "foreign capital", "资本流", "外资", "资金流", "钱去哪"]):
        return "global foreign capital outflows 2025 where is money going safe haven"

    # Interest rates / monetary policy
    if any(w in msg_l for w in ["interest rate", "rate hike", "rate cut", "fed", "利率", "加息", "降息", "央行"]):
        return "central bank interest rate decision 2025 fed ecb boj impact"

    # Inflation / CPI
    if any(w in msg_l for w in ["inflation", "cpi", "通胀", "通货膨胀", "物价"]):
        return "global inflation CPI 2025 latest data impact markets"

    # Recession / economic slowdown
    if any(w in msg_l for w in ["recession", "slowdown", "gdp", "经济衰退", "经济放缓", "gdp增速"]):
        return "global recession risk GDP slowdown 2025 economic outlook"

    # Trade war / tariffs
    if any(w in msg_l for w in ["tariff", "trade war", "关税", "贸易战", "贸易摩擦"]):
        return "US China trade war tariffs 2025 impact markets"

    # USD / dollar strength
    if any(w in msg_l for w in ["dollar", "usd", "美元", "美元走强", "汇率"]):
        return "US dollar strength DXY 2025 emerging markets impact"

    # Sector / industry
    if any(w in msg_l for w in ["semiconductor", "半导体", "chip", "芯片"]):
        return "semiconductor industry outlook 2025 demand AI chips"
    if any(w in msg_l for w in ["ai", "artificial intelligence", "人工智能"]):
        return "AI artificial intelligence stocks market outlook 2025"
    if any(w in msg_l for w in ["energy", "oil", "crude", "能源", "石油", "原油"]):
        return "oil energy prices 2025 global demand supply outlook"

    # Geopolitical
    if any(w in msg_l for w in ["geopolit", "war", "conflict", "地缘", "战争", "冲突"]):
        return "geopolitical risk 2025 markets impact global"

    # Generic global market question — strip non-ASCII for a cleaner query
    import re as _re
    ascii_only = _re.sub(r'[^\x00-\x7F]+', ' ', message).strip()
    market_prefix = f"{market} market" if market and market not in (None, "ALL", "global") else "global markets"
    query = f"{market_prefix} {ascii_only[:100]}".strip()
    return query if len(query) > 10 else f"{market_prefix} outlook 2025"


def _run_openbb_tools(tools, tickers, market, data_dir, message=""):
    """Run OpenBB tools in parallel. Returns dict of tool_name → result text."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _run(name):
        try:
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
            # Apply market suffix (e.g. .HK, .T) for yfinance-backed tools
            suffixed = [_apply_market_suffix(t, market) for t in tickers]
            if name == "openbb_quote" and tickers:
                return name, get_stock_quote(",".join(suffixed[:3]))
            if name == "openbb_news" and tickers:
                return name, get_company_news(suffixed[0])
            if name == "openbb_history" and tickers:
                return name, get_stock_history(suffixed[0])
            if name == "openbb_profile" and tickers:
                return name, get_company_profile(suffixed[0])
            if name == "openbb_fundamentals" and tickers:
                # Run fundamentals for each ticker (up to 3), combine results
                parts = []
                for t in suffixed[:3]:
                    parts.append(f"[{t}]\n{get_fundamentals(t)}")
                return name, "\n\n".join(parts)

            return name, "(skipped — no tickers provided)"
        except Exception as e:
            return name, f"(tool error: {e})"

    results = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(_run, t): t for t in tools}
        for future in futures:
            name, result = future.result()
            results[name] = result
    return results


# ── System prompts per agent type ────────────────────────────────────

SYSTEM_PROMPTS = {
    "market": """You are a financial analyst for the {market} market.

IMPORTANT: All tools have already run. Results are in the DATA block below. Answer directly using that data — do NOT ask the user for more data, do NOT ask for permission, do NOT ask which companies to include. If data is missing for a specific stock, note it in one word ("unavailable") and move on.

Data available: stock prices (return_1d, return_20d), market cap, vol_ratio, news, signals, capital flow, peer comparison reports, live data from OpenBB.
Prefer live data (labeled "LIVE MARKET DATA") over local data when both are present — it's more current.

## How to respond
- Lead with data and specific numbers. Never pad with definitions or analogies.
- Answer the specific question asked — don't force a fixed structure onto every response.
- For peer/company comparisons: ONE table only. Merge price + financials into a single table. Do NOT repeat the same company list multiple times. Mark missing data as "—" (not "unavailable" repeated in paragraphs).
- For market overview: cover direction, notable movers, and key drivers.
- If a Task is specified below, follow it precisely — it defines exactly what to cover.
- If PEER AGENT data is in DATA: use its tables directly — do NOT restate or re-explain what is already in the table. Add only interpretation and the answer to the user's specific question.
- Calibrate depth to the question: quick fact lookup → short; analysis/explanation → go deep, cover all angles.
- Use local price data (return_1d, return_20d, market_cap, vol_ratio) even when revenue/profit is absent — price momentum is a valid proxy for relative strength.
- Use headers and bullet points where they help clarity, not by default.
- RESPOND IN ENGLISH ONLY. All text — section headers, bullet points, table headers, labels, commentary — must be in English. Do not switch to Chinese even for Chinese company names or A-share tickers.""",

    "global": """You are a financial analyst covering global markets.

IMPORTANT: All tools have already run. Results are in the DATA block below. Answer directly — do NOT ask the user for more data, do NOT ask for permission, do NOT ask which companies to analyze. If data is missing, note it in one word and continue.

Data available: capital flows by market, macro indicators (rates, commodities, VIX, currencies), sector performance, geopolitical news across 13 markets.

## How to respond
- Lead with data. Cite specific numbers from the DATA block — don't summarize vaguely.
- Answer the specific question asked. If asked where money is flowing, name the markets and the numbers.
- Web search results supplement local data — use both, cite the source when relevant.
- If a Task is specified below, treat it as a required checklist — address every point in it.
- Calibrate depth to the question: a simple lookup → concise; an analytical question → full depth, cover all mechanisms and evidence.
- Use headers where they help, not by default.
- RESPOND IN ENGLISH ONLY. All text — section headers, bullet points, table headers, labels, commentary — must be in English. Do not switch to Chinese even for Chinese company names or A-share tickers.""",
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
                              data_dir: str, market: str) -> str:
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
        raw = _call_llm(planner_system, [], planner_user)
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

    return "\n\n".join(parts) if parts else ""


def agent_chat(agent_type, market, message, data_dir, chat_history=None,
               language=None, context=None):
    """
    LLM-backed agent chat with LLM router.

    Args:
        context: dict with keys {ticker, market, company_name} — carried across turns
                 so follow-up messages like "what about its PE?" resolve correctly.
    Returns:
        (response_str, updated_context_dict)
    """
    import re as _re

    context = context or {}

    provider, api_key = get_llm_provider()
    if provider is None:
        return None, context

    _MARKET_NAMES = {
        "CN": "A股", "HK": "港股", "US": "美股", "JP": "日股",
        "KR": "韩股", "TW": "台股", "IN": "印度股", "UK": "英股",
        "DE": "德股", "FR": "法股", "AU": "澳股", "BR": "巴西股", "SA": "沙特股",
    }

    # ── Phase 1: PLAN ────────────────────────────────────────────────────
    plan = _plan_message(message, market, agent_type, chat_history)


    intent       = plan.get("intent", "general")
    tickers      = plan.get("tickers", [])
    company_name = plan.get("company_name", "")
    response_focus = plan.get("response_focus", "")
    tools        = list(dict.fromkeys(s["tool"] for s in plan.get("steps", []) if s.get("tool") in TOOLS))
    is_global    = agent_type == "global" or market in (None, "ALL")

    # Inherit ticker from previous turn when question uses pronouns
    if not tickers and not company_name and context.get("ticker"):
        tickers = [context["ticker"]]
        company_name = context.get("company_name", "")

    # ── Company-name → ticker resolution & market boundary check ─────────
    def _redirect_msg(name_or_ticker, ticker_market):
        current_label = _MARKET_NAMES.get(market, market)
        alt_label     = _MARKET_NAMES.get(ticker_market, ticker_market)
        return (f"**{name_or_ticker}** 不是{current_label}，"
                f"它在**{alt_label}**上市。\n\n"
                f"请切换到 **{ticker_market} Market Agent** 提问。")

    if intent == "single_stock" and market and market not in (None, "ALL", "global"):
        try:
            from web.local_trading_agent import detect_market as _detect
        except ImportError:
            from local_trading_agent import detect_market as _detect

        if tickers:
            foreign = [t for t in tickers if _detect(t) != market]
            if foreign:
                tk = foreign[0]
                return _redirect_msg(company_name or tk, _detect(tk)), context
        elif company_name:
            local_ticker = _resolve_cn_name_to_ticker(company_name, data_dir) \
                if market == "CN" else _find_company_in_market(company_name, market, data_dir)
            if local_ticker:
                tickers = [local_ticker]
            else:
                alt_ticker, alt_market = _find_company_in_markets(company_name, data_dir)
                if alt_ticker and alt_market:
                    return _redirect_msg(company_name, alt_market), context
                else:
                    company_name = ""

    # Global agent: resolve company_name → ticker across all markets
    if intent == "single_stock" and not tickers and company_name \
            and (not market or market in (None, "ALL", "global")):
        resolved_ticker, _ = _find_company_in_markets(company_name, data_dir)
        if resolved_ticker:
            tickers = [resolved_ticker]

    # ── Multi-company resolution: for peer_comparison with named stocks ─────
    # When the user lists N specific company names for direct comparison, the planner
    # may leave tickers=[] and put individual tickers only in step fields.
    # Collect all tickers referenced in plan steps and resolve any remaining names.
    _unresolved_names = []  # company names mentioned by user but not found in local universe
    if market and not is_global:
        step_tickers = [s["ticker"] for s in plan.get("steps", []) if s.get("ticker")]
        for t in step_tickers:
            if t not in tickers:
                tickers.append(t)
        # Resolve company names from the message; track ones that fail resolution
        import re as _re2
        _cn_names = _re2.findall(r'[\u4e00-\u9fff]{2,8}', message)
        _resolve_fn = _resolve_cn_name_to_ticker if market == "CN" else None
        if _resolve_fn:
            for name in _cn_names:
                if len(name) >= 3:
                    resolved = _resolve_fn(name, data_dir)
                    if resolved and resolved not in tickers:
                        tickers.append(resolved)
                    elif not resolved:
                        _unresolved_names.append(name)
        tickers = list(dict.fromkeys(tickers))  # deduplicate, preserve order

    # Save resolved ticker into context for future turns
    if tickers:
        context = {
            "ticker":       tickers[0],
            "market":       market or context.get("market", ""),
            "company_name": company_name or context.get("company_name", tickers[0]),
        }

    # ── Phase 2: EXECUTE PLAN ────────────────────────────────────────────
    # Independent steps run in parallel; dependent steps run sequentially.
    # trading_agent and sector_agent are handled inside _execute_plan.
    tool_results = _execute_plan(plan, tickers, market, agent_type, data_dir, message)

    if _unresolved_names and market == "CN":
        # Resolve names → tickers first, then search by ticker code
        _missing_parts = []
        for _mname in _unresolved_names[:3]:
            _tk = _resolve_cn_name_to_ticker(_mname, data_dir)
            _q = _tk if _tk else _mname
            _r = _web_search(_q, max_results=3, region="cn-zh")
            if _r and "(web search failed" not in _r:
                _missing_parts.append(_r)
        if _missing_parts:
            tool_results["web_search_missing"] = "\n\n".join(_missing_parts)
            print(f"[missing-stocks] web_search for: {_unresolved_names}", flush=True)

    # ── Phase 2b-inject: Global agent always needs local context ─────────
    # The planner may not always include a local_data step (e.g. macro questions
    # that only request web_search). For global agent, local context contains
    # capital flows, macro indicators, sector performance — always inject it.
    is_global_agent = agent_type == "global" or market in (None, "ALL")
    if is_global_agent and "local_data" not in tool_results:
        tool_results["local_data"] = _load_global_context(data_dir)

    # ── Phase 2b-ext: CN peer financials via akshare (bypasses unreliable web search) ──
    # For CN peer_comparison, pull revenue/profit directly from akshare instead of DDG.
    _is_cn_peer = (intent == "peer_comparison" and (market == "CN" or any(
        __import__("re").match(r"^\d{6}$", t) for t in tickers)))
    if _is_cn_peer and tickers:
        anchor = tickers[0]
        _peer_tks = []
        try:
            import pandas as _pd2, re as _re2
            _uni = _pd2.read_parquet(f"{data_dir}/markets/CN/universe.parquet")
            # Priority 1: tickers explicitly mentioned in web search results
            _ws = tool_results.get("web_search", "")
            # Only match valid A-share ticker prefixes (0xx=SZ, 3xx=SZ growth, 6xx=SH, 9xx=SH/BSE)
            # Exclude numbers embedded in URL paths (preceded by /)
            _ws_tks = list(dict.fromkeys(
                t for t in _re2.findall(r"(?<![/\d])([0369]\d{5})(?!\d)", _ws)
                if t != anchor
            ))[:5]
            # Priority 2: same-sector peers filtered by name relevance to 算力/服务器/计算
            _COMPUTE_KEYWORDS = ["信息", "服务器", "算力", "计算", "数创", "长城", "同方", "浪潮"]
            _anchor_row = _uni[_uni["ticker"].astype(str) == anchor]
            _sector_peers = []
            if not _anchor_row.empty and "sector" in _uni.columns:
                _sector = _anchor_row.iloc[0]["sector"]
                _sector_df = _uni[(_uni["sector"] == _sector) & (_uni["ticker"].astype(str) != anchor)]
                # Prefer companies whose names contain compute-related keywords
                _relevant = _sector_df[_sector_df["name"].apply(
                    lambda n: any(k in str(n) for k in _COMPUTE_KEYWORDS))]
                _sector_peers = _relevant["ticker"].astype(str).tolist()[:4]
            _peer_tks = list(dict.fromkeys(_ws_tks + _sector_peers))[:6]
        except Exception:
            pass
        ak_result = _cn_peer_financials(_peer_tks, anchor, data_dir)
        if ak_result:
            tool_results["cn_peer_financials"] = ak_result
            print(f"[akshare] fetched peer financials for {[anchor]+_peer_tks}", flush=True)

    # ── Phase 2c: EXTRACT (structure raw results into per-company financials) ─
    # Trigger when question involves financial comparisons or revenue/profit data.
    _needs_extraction = (
        intent in ("peer_comparison",)
        or any(w in message for w in _EXTRACT_TRIGGERS)
        or bool(tool_results.get("web_search_missing"))
    )
    if _needs_extraction and (tool_results.get("web_search") or tool_results.get("fetch_url") or tool_results.get("web_search_missing")):
        companies_hint = []
        if company_name:
            companies_hint.append(company_name)
        # Add any company names discovered during peer search (from web_search step 1)
        import re as _re_ext
        ws = tool_results.get("web_search", "")
        found_cn = list(dict.fromkeys(_re_ext.findall(r'[\u4e00-\u9fff]{2,6}(?:信息|科技|股份|系统|数据|网络|通信|电子|智能)?', ws)))
        companies_hint += [n for n in found_cn[:6] if n not in companies_hint]
        extracted = _extract_financials(tool_results, companies_hint, message)
        if extracted:
            # Replace raw web_search in context with the clean extracted version
            # Keep raw web_search available for source URLs but put extracted first
            tool_results["web_summary"] = extracted

    # 4a. Web → local enrichment via LLM planner
    # Skip for peer_comparison — peer discovery is web-based, local enrichment adds noise.
    if "web_search" in tool_results and intent not in ("peer_comparison",):
        web_raw = tool_results["web_search"]
        if "(web search failed" not in web_raw and "(no web results)" not in web_raw:
            local_enrichment = _web_to_local_enrichment(
                web_raw, message, data_dir, market
            )
            if local_enrichment:
                tool_results["local_data_from_web"] = local_enrichment

    # 4b. Build combined context from tool outputs.
    is_global_agent = agent_type == "global" or market in (None, "ALL")
    # Order: primary analysis first, then structured financial data, then raw web
    _priority = ["peer_agent", "trading_agent", "sector_agent", "cn_peer_financials", "web_summary"]
    ordered_results = {k: tool_results[k] for k in _priority if k in tool_results}
    for k, v in tool_results.items():
        if k not in _priority:
            ordered_results[k] = v

    context_parts = []
    for tool_name, result in ordered_results.items():
        label = tool_name.replace("_", " ").upper()
        context_parts.append(f"=== {label} ===\n{result}")

    combined_context = "\n\n".join(context_parts)

    # 5. Pick system prompt
    is_global = agent_type == "global" or market in (None, "ALL")
    if is_global:
        base_system = SYSTEM_PROMPTS["global"]
    else:
        base_system = SYSTEM_PROMPTS["market"].format(market=market)

    system = base_system

    focus_block = f"\n\nTask: {response_focus}" if response_focus else ""

    # Tell the LLM that all fetching has already been done — data is in the DATA block.
    _data_notes = []
    if tool_results.get("peer_agent"):
        _data_notes.append("PEER AGENT contains a complete peer comparison report with formatted tables (price + financials). Use it as the PRIMARY output — preserve its tables as-is, do NOT rewrite or re-summarize them. Only add web search findings as brief supplementary notes below.")
    if tool_results.get("trading_agent"):
        _data_notes.append("TRADING AGENT contains a complete stock analysis — use it as the primary source. Supplement with other DATA sections to answer the specific question.")
    if tool_results.get("sector_agent"):
        _data_notes.append("SECTOR AGENT contains a complete sector analysis — use it as the primary source.")
    if tool_results.get("fetch_url"):
        _data_notes.append("The URL in the user's message has already been fetched — its content is in the DATA section below. Do not say you cannot access it.")
    if tool_results.get("web_search") or tool_results.get("web_search_missing") or tool_results.get("web_summary"):
        _data_notes.append("Web search has already been performed — results are in the DATA section below. At the end of your response, add a brief '## News highlights' section (3–5 bullet points) summarizing the most relevant recent news or announcements from the web search results. Skip this section only if the web content is entirely off-topic.")
    if "cn_peer_financials" in tool_results and "web_summary" in tool_results:
        _data_notes.append("CN PEER FINANCIALS has structured annual data. EXTRACTED FINANCIALS may contain more recent figures — use both.")
    if "cn_peer_financials" in tool_results:
        _data_notes.append("IMPORTANT: Use the exact years shown in the data (e.g. 2024). Never relabel or shift years.")
    _data_notes_block = "\n".join(_data_notes) + "\n\n" if _data_notes else ""

    full_system = (
        f"{_data_notes_block}"
        f"{system}{focus_block}\n\n"
        f"--- DATA ---\n{combined_context}"
    )

    # 6. Build conversation history (last 4 turns)
    history_msgs = []
    if chat_history:
        for msg in chat_history[-4:]:
            role = "assistant" if msg.get("role") == "agent" else "user"
            history_msgs.append({"role": role, "content": msg["content"]})

    web_result = tool_results.get("web_search", "")

    # For peer_comparison where peer_agent ran: constrain synthesizer so it doesn't
    # re-enumerate every company in prose after the tables already show the data.
    wrapped_message = message
    if intent == "peer_comparison" and tool_results.get("peer_agent"):
        wrapped_message = (
            f"{message}\n\n"
            "[INSTRUCTION] The PEER AGENT section already contains complete tables. "
            "Output those tables verbatim. Then add ONLY a brief conclusion (≤5 bullet points): "
            "which companies show the strongest growth, any key divergence vs the anchor, "
            "and what the web search adds that is NOT already in the tables. "
            "Do NOT re-list each company in prose. Do NOT repeat numbers already in the tables."
        )

    print(f"[synthesis] system_chars={len(full_system)} history_turns={len(history_msgs)}", flush=True)

    try:
        result = _call_llm(full_system, history_msgs, wrapped_message)

        # Append a compact sources footer as a fallback reference
        if web_result and "(web search failed" not in web_result and "(no web results)" not in web_result:
            sources_footer = _extract_sources(web_result)
            if sources_footer and result and "http" not in result:
                result = (result or "") + sources_footer

        return result or "", context
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"(error: {e})", context


def _call_anthropic(api_key, system, history, message, max_tokens=8096):
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


def _call_openai(api_key, system, history, message, max_tokens=None):
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

    total_chars = sum(len(str(m.get("content",""))) for m in messages)
    print(f"[openai] model={model} input_chars={total_chars} (~{total_chars//4}t)", flush=True)

    # Only pass a token limit if explicitly requested; otherwise let the model decide.
    token_kwargs = {}
    if max_tokens is not None:
        _new_token_param_models = ("o1", "o3", "o4", "gpt-5")
        _use_completion_tokens = any(model.startswith(p) for p in _new_token_param_models)
        token_kwargs = {"max_completion_tokens": max_tokens} if _use_completion_tokens else {"max_tokens": max_tokens}

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        **token_kwargs,
    )
    content = resp.choices[0].message.content or ""
    finish = resp.choices[0].finish_reason
    print(f"[openai] finish_reason={finish} output_chars={len(content)}", flush=True)
    return content


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

            # Estimate message tokens. Chinese chars cost ~2t each, ASCII ~0.25t.
            # Use a conservative 2 chars/token to avoid underestimating CJK content.
            def _tokens(m):
                return len(str(m.get("content") or "")) // 2

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
