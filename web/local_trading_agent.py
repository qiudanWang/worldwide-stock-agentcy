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
        suffix_map = {"T": "JP", "KS": "KR", "KQ": "KR", "TW": "TW",
                      "NS": "IN", "BO": "IN", "L": "UK", "DE": "DE",
                      "PA": "FR", "AX": "AU", "SA": "BR",
                      "SS": "CN", "SZ": "CN", "HK": "HK"}
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
        # Read last 2 files so we can fall back to yesterday's close when today's
        # snapshot was written before market close (many NaN close values).
        dfs = []
        for f in files[-2:]:
            try:
                dfs.append(pd.read_parquet(f))
            except Exception:
                pass
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        if "ticker" in df.columns:
            tdf = df[df["ticker"] == ticker].copy()
            if tdf.empty:
                return result
            tdf = tdf.sort_values("date") if "date" in tdf.columns else tdf
            # Prefer rows with valid close; fall back to any row
            tdf_valid = tdf[tdf["close"].notna()] if "close" in tdf.columns else tdf
            if tdf_valid.empty:
                tdf_valid = tdf

            latest = tdf_valid.iloc[-1]
            result["close"]       = latest.get("close")
            result["return_1d"]   = latest.get("return_1d")
            result["return_5d"]   = latest.get("return_5d")
            result["return_20d"]  = latest.get("return_20d")
            result["volume_ratio"]= latest.get("volume_ratio")
            result["volume"]      = latest.get("volume")
            # Include the date of the latest data point so LLM can cite it
            if latest.get("date") is not None:
                d = latest["date"]
                result["date"] = str(d.date()) if hasattr(d, "date") else str(d)[:10]

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


def _load_local_peers(ticker: str, market: str, data_dir: str) -> str:
    """Find peer companies in the same industry/sector from local universe data.

    Returns formatted text with peer tickers, names, price performance, and descriptions.
    """
    import glob as _glob

    base = os.path.join(data_dir, "markets", market)

    # Load universe
    try:
        uni = pd.read_parquet(os.path.join(base, "universe.parquet"))
    except Exception as e:
        return f"(local_peers: could not load universe — {e})"

    # Normalize: strip exchange suffix (.SS, .SZ, .HK, .T, etc.) for local lookup
    ticker_clean = ticker.split(".")[0] if "." in ticker else ticker
    anchor_row = uni[uni["ticker"].astype(str) == ticker_clean]
    if anchor_row.empty:
        return f"(local_peers: ticker {ticker_clean} not found in {market} universe)"
    ticker = ticker_clean  # use clean ticker for rest of function

    anchor = anchor_row.iloc[0]
    anchor_name = anchor.get("name", ticker)

    # Determine industry/subsector and sector columns available
    industry_col = None
    for col in ("industry", "subsector"):
        if col in uni.columns:
            industry_col = col
            break
    sector_col = "sector" if "sector" in uni.columns else None

    anchor_industry = anchor.get(industry_col) if industry_col else None
    anchor_sector = anchor.get(sector_col) if sector_col else None

    # Try industry-level peers first, fall back to sector
    peers = pd.DataFrame()
    match_level = "industry"
    if anchor_industry and industry_col:
        peers = uni[
            (uni[industry_col] == anchor_industry) &
            (uni["ticker"].astype(str) != ticker)
        ]
    if len(peers) < 3 and anchor_sector and sector_col:
        peers = uni[
            (uni[sector_col] == anchor_sector) &
            (uni["ticker"].astype(str) != ticker)
        ]
        match_level = "sector"

    if peers.empty:
        return f"(local_peers: no peers found for {ticker} in {market})"

    # Keyword-based relevance filter: if the anchor company name contains
    # compute/server/storage keywords, prefer peers with matching keywords.
    # This avoids lumping unrelated companies (e.g. 万集科技, 兆日科技) with
    # server/AI-compute players (e.g. 浪潮信息, 海光信息).
    _COMPUTE_KW = ["服务器", "算力", "计算", "存储", "数据", "信息", "智能", "芯片",
                   "半导体", "处理器", "网络", "安全", "云", "IDC", "数创", "长城",
                   "浪潮", "曙光", "超聚变", "同方", "麒麟", "鲲鹏"]
    anchor_name_str = str(anchor_name)
    if "name" in peers.columns and any(kw in anchor_name_str for kw in _COMPUTE_KW):
        filtered = peers[peers["name"].apply(
            lambda n: any(kw in str(n) for kw in _COMPUTE_KW)
        )]
        if len(filtered) >= 3:
            peers = filtered

    # Sort by market_cap if available, else by name
    if "market_cap" in peers.columns:
        peers = peers.sort_values("market_cap", ascending=False)
    elif "name" in peers.columns:
        peers = peers.sort_values("name")
    peers = peers.head(15)

    # Load latest price snapshot
    price_snap = {}
    files = sorted(_glob.glob(os.path.join(base, "market_daily_*.parquet")))
    if files:
        try:
            # Read last 2 files so we can fall back to yesterday's close when
            # today's snapshot was written before market close (NaN close values).
            dfs = []
            for f in files[-2:]:
                try:
                    dfs.append(pd.read_parquet(f))
                except Exception:
                    pass
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                if "ticker" in df.columns and "date" in df.columns:
                    # Keep only rows with valid close, then pick the most recent per ticker
                    df_valid = df[df["close"].notna()] if "close" in df.columns else df
                    if df_valid.empty:
                        df_valid = df  # fallback: use any data if nothing has valid close
                    df_valid = df_valid.sort_values("date").drop_duplicates("ticker", keep="last")
                    for _, row in df_valid.iterrows():
                        price_snap[str(row["ticker"])] = row
        except Exception:
            pass

    # Load financials snapshot (latest annual + latest quarterly per ticker)
    fin_snap = {}
    fin_path = os.path.join(base, "financials.parquet")
    if os.path.exists(fin_path):
        try:
            fin_df = pd.read_parquet(fin_path)
            if not fin_df.empty and "ticker" in fin_df.columns and "period_end" in fin_df.columns:
                fin_df["period_end"] = pd.to_datetime(fin_df["period_end"], errors="coerce")
                fin_df = fin_df.dropna(subset=["period_end"])
                for tkr, grp in fin_df.groupby("ticker"):
                    annual = grp[grp["period_type"] == "annual"].sort_values("period_end")
                    qtrly  = grp[grp["period_type"] == "quarterly"].sort_values("period_end")
                    fin_snap[str(tkr)] = {
                        "annual":  annual.iloc[-1].to_dict() if not annual.empty else None,
                        "quarter": qtrly.iloc[-1].to_dict()  if not qtrly.empty  else None,
                    }
        except Exception:
            pass

    # Determine whether we have any financials data for the peers
    any_fin = any(fin_snap.get(str(peer["ticker"])) for _, peer in peers.iterrows())
    has_fin_cols = any_fin and any(
        (fin_snap.get(str(peer["ticker"]), {}) or {}).get("annual")
        for _, peer in peers.iterrows()
    )

    # ── Build markdown table ────────────────────────────────────────
    industry_label = anchor_industry or anchor_sector or "N/A"
    sector_label = anchor_sector or "N/A"

    # Price table
    price_header = "| Ticker | Name | Close | 1d | 20d | Vol Ratio |"
    price_sep    = "|--------|------|-------|----|-----|-----------|"
    price_rows = []

    # Financials table (only built if we have fin data)
    fin_header = "| Ticker | Name | Ann Period | Ann Rev (亿) | Ann NP (亿) | Latest Qtr | Qtr Rev (亿) | Qtr NP (亿) |"
    fin_sep    = "|--------|------|------------|-------------|------------|------------|-------------|------------|"
    fin_rows = []

    def _fmt_pct(v):
        if v is None: return "—"
        try: return f"{float(v):+.1f}%"
        except: return "—"

    def _fmt_val(v, decimals=1):
        if v is None: return "—"
        try: return f"{float(v):.{decimals}f}"
        except: return "—"

    # Include anchor as first row
    all_peers = [(ticker, anchor_name, price_snap.get(ticker), fin_snap.get(ticker, {}))]
    for _, peer in peers.iterrows():
        pt = str(peer["ticker"])
        all_peers.append((pt, peer.get("name", pt), price_snap.get(pt), fin_snap.get(pt, {})))

    for pticker, pname, snap, fin in all_peers:
        # Price row (snap is a pandas Series or None)
        close = _fmt_val(snap.get("close"), 2) if snap is not None else "—"
        r1d   = _fmt_pct(snap.get("return_1d") * 100  if snap is not None and snap.get("return_1d")  is not None else None)
        r20d  = _fmt_pct(snap.get("return_20d") * 100 if snap is not None and snap.get("return_20d") is not None else None)
        vr    = f"{snap.get('volume_ratio'):.1f}x" if snap is not None and snap.get("volume_ratio") is not None else "—"
        price_rows.append(f"| {pticker} | {pname} | {close} | {r1d} | {r20d} | {vr} |")

        # Collect financials row with sort key (fiscal_year) for grouping later
        if has_fin_cols:
            ann = (fin or {}).get("annual") or {}
            qtr = (fin or {}).get("quarter") or {}
            fy       = ann.get("fiscal_year")
            fy_label = str(fy) if fy else "—"
            ann_rev  = _fmt_val(ann.get("revenue"))
            ann_np   = _fmt_val(ann.get("net_profit"), 2)
            ann_note = ""
            if ann.get("revenue_yoy") is not None:
                ann_note = f"({_fmt_pct(ann.get('revenue_yoy'))})"
            qend     = str(qtr.get("period_end", ""))[:7] if qtr else "—"
            qtr_rev  = _fmt_val(qtr.get("revenue"))
            qtr_np   = _fmt_val(qtr.get("net_profit"), 2)
            fin_rows.append((
                fy or 0,  # sort key
                f"| {pticker} | {pname} | FY{fy_label} | {ann_rev} {ann_note} | {ann_np} | {qend} | {qtr_rev} | {qtr_np} |"
            ))

    price_table = "\n".join([price_header, price_sep] + price_rows)

    # Build financials table sorted by fiscal year descending (most recent first)
    fin_table = ""
    if has_fin_cols and fin_rows:
        fin_rows_sorted = sorted(fin_rows, key=lambda x: x[0], reverse=True)
        data_rows = [row for _, row in fin_rows_sorted]
        fin_table = "\n".join([fin_header, fin_sep] + data_rows)

    header = (
        f"**Peers: {ticker} ({anchor_name})**  \n"
        f"Industry: {industry_label} / Sector: {sector_label} ({market}, matched by {match_level})\n\n"
    )
    parts = [header + price_table]
    if fin_table:
        parts.append("\n**Financials (亿 local currency, YoY%)**\n\n" + fin_table)
    return "\n".join(parts)


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


def _load_yf_fundamentals(ticker: str, market: str = None, data_dir: str = None) -> dict:
    """Load fundamentals: pipeline cache first, then live yfinance fallback."""
    # ── 1. Try pipeline cache (fundamentals.parquet written by DataAgent weekly) ──
    if market and data_dir:
        cache_path = os.path.join(data_dir, "markets", market, "fundamentals.parquet")
        bare = ticker.split(".")[0]  # strip .HK etc.
        try:
            cache = pd.read_parquet(cache_path)
            row = cache[cache["ticker"].astype(str) == bare]
            if not row.empty:
                d = row.iloc[0].dropna().to_dict()
                d.pop("ticker", None)
                d.pop("market", None)
                if d:
                    return d
        except Exception:
            pass

    # ── 2. Live yfinance fetch (fallback when cache missing or stale) ──
    result = {}
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info or {}
        wanted = ["shortName", "sector", "industry", "trailingPE", "priceToBook",
                  "returnOnEquity", "revenueGrowth", "profitMargins", "debtToEquity",
                  "currentPrice", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
                  "marketCap", "dividendYield", "earningsGrowth", "forwardPE",
                  "totalRevenue", "grossProfits", "operatingMargins",
                  "trailingEps", "forwardEps", "freeCashflow", "ebitda"]
        for k in wanted:
            if info.get(k) is not None:
                result[k] = info[k]
    except Exception:
        pass

    # Quarterly financials: last 3 quarters of Revenue, Gross Profit, Net Income, Operating Income
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        qf = t.quarterly_financials
        if qf is not None and not qf.empty:
            key_rows = ["Total Revenue", "Gross Profit", "Net Income", "Operating Income", "EBITDA"]
            quarterly = {}
            for row in key_rows:
                if row in qf.index:
                    series = qf.loc[row].dropna()
                    quarterly[row] = {
                        str(ts.date()): round(v / 1e8, 2)  # convert to 亿
                        for ts, v in series.head(3).items()
                    }
            if quarterly:
                result["quarterly_financials_100M_CNY"] = quarterly
    except Exception:
        pass

    # Balance sheet: key items (cash, total debt, current ratio, total assets)
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        bs = t.quarterly_balance_sheet
        if bs is not None and not bs.empty:
            bs_rows = ["Cash And Cash Equivalents", "Total Debt", "Current Assets",
                       "Current Liabilities", "Total Assets", "Stockholders Equity"]
            balance = {}
            for row in bs_rows:
                if row in bs.index:
                    series = bs.loc[row].dropna()
                    if not series.empty:
                        latest_ts = series.index[0]
                        balance[row] = {
                            "latest_quarter": str(latest_ts.date()),
                            "value_100M_CNY": round(series.iloc[0] / 1e8, 2)
                        }
            if balance:
                result["balance_sheet_latest"] = balance
    except Exception:
        pass

    # Cash flow: operating and free cash flow (last 3 quarters)
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        cf = t.quarterly_cashflow
        if cf is not None and not cf.empty:
            cf_rows = ["Operating Cash Flow", "Free Cash Flow", "Capital Expenditure"]
            cashflow = {}
            for row in cf_rows:
                if row in cf.index:
                    series = cf.loc[row].dropna()
                    cashflow[row] = {
                        str(ts.date()): round(v / 1e8, 2)
                        for ts, v in series.head(3).items()
                    }
            if cashflow:
                result["quarterly_cashflow_100M_CNY"] = cashflow
    except Exception:
        pass

    # Analyst ratings: consensus target price, buy/hold/sell counts, recent upgrades
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        # Price targets
        targets = t.analyst_price_targets
        if targets and isinstance(targets, dict):
            result["analyst_price_targets"] = {
                k: round(v, 2) for k, v in targets.items()
                if v is not None and k in ("current", "low", "high", "mean", "median", "numberOfAnalysts")
            }
        # Recommendations summary (buy/hold/sell counts)
        rec = t.recommendations_summary
        if rec is not None and not rec.empty:
            latest = rec.iloc[0].to_dict()
            result["analyst_recommendations"] = {
                k: int(v) for k, v in latest.items()
                if isinstance(v, (int, float)) and not __import__("math").isnan(float(v))
            }
        # Recent upgrades/downgrades (last 5)
        upgrades = t.upgrades_downgrades
        if upgrades is not None and not upgrades.empty:
            recent = upgrades.head(5).reset_index()
            result["recent_rating_changes"] = recent[
                [c for c in ["GradeDate", "Firm", "ToGrade", "FromGrade", "Action"]
                 if c in recent.columns]
            ].to_dict("records")
    except Exception:
        pass

    return result


def _load_index_benchmark(market: str, data_dir: str) -> list:
    """Load recent index performance for benchmark context (last 5 trading days)."""
    try:
        import os as _os
        import pandas as _pd
        path = _os.path.join(data_dir, "markets", market, "indices.parquet")
        df = _pd.read_parquet(path)
        if df.empty:
            return []
        df["date"] = _pd.to_datetime(df["date"])
        result = []
        for symbol, grp in df.groupby("symbol"):
            grp = grp.sort_values("date").tail(5)
            result.append({
                "symbol": symbol,
                "name": grp.iloc[-1].get("name", symbol),
                "latest_close": round(float(grp.iloc[-1]["close"]), 2),
                "change_1d_pct": round(float(grp.iloc[-1]["change_pct"]) * 100, 2)
                    if grp.iloc[-1].get("change_pct") is not None else None,
                "5d_history": [
                    {"date": str(r["date"].date()), "close": round(float(r["close"]), 2)}
                    for _, r in grp.iterrows()
                ],
            })
        return result
    except Exception:
        return []


def gather_data(ticker: str, market: str, data_dir: str) -> dict:
    """Collect all available data for the ticker."""
    info    = _load_local_universe(ticker, market, data_dir)
    price   = _load_local_price(ticker, market, data_dir)
    news    = _load_local_news(ticker, market, data_dir)
    alerts  = _load_local_alerts(ticker, market, data_dir)
    indices = _load_index_benchmark(market, data_dir)

    if market == "CN":
        fundamentals = _load_cn_fundamentals(ticker)
    else:
        # Add exchange suffix for yfinance if not already present
        _YF_SUFFIX = {"HK": ".HK", "JP": ".T", "AU": ".AX", "IN": ".NS",
                      "KR": ".KS", "TW": ".TW", "DE": ".DE", "FR": ".PA",
                      "UK": ".L", "BR": ".SA", "SA": ".SR"}
        suffix = _YF_SUFFIX.get(market, "")
        yf_ticker = ticker if (not suffix or ticker.endswith(suffix)) else ticker + suffix
        fundamentals = _load_yf_fundamentals(yf_ticker, market=market, data_dir=data_dir)

    return {
        "ticker":       ticker,
        "market":       market,
        "info":         info,
        "price":        price,
        "news":         news,
        "alerts":       alerts,
        "fundamentals": fundamentals,
        "indices":      indices,
    }


# ── Prompt builders ───────────────────────────────────────────────

def _fmt(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False, default=str, indent=2)


def _price_summary(price: dict) -> str:
    if not price:
        return "No price data available from local pipeline."
    lines = []
    date_str = price.get("date", "")
    if date_str:
        lines.append(f"Data date: {date_str}")
    if price.get("close"):      lines.append(f"Close: {price['close']}")
    if price.get("return_1d") is not None: lines.append(f"1d return ({date_str}): {price['return_1d']:+.2%}")
    if price.get("return_5d") is not None: lines.append(f"5d return: {price['return_5d']:+.2%}")
    if price.get("return_20d") is not None: lines.append(f"20d return: {price['return_20d']:+.2%}")
    if price.get("volume_ratio") is not None:
        vr = price["volume_ratio"]
        flag = " ⚠️ abnormal" if vr >= 2.0 else ""
        lines.append(f"Volume ratio: {vr:.2f}x{flag}")
    hist = price.get("history", [])
    if hist:
        # Show last 10 days with dates
        recent = [h for h in hist if h.get("close")][-10:]
        entries = []
        for h in recent:
            d = str(h["date"])[:10] if h.get("date") else ""
            entries.append(f"{d}:{round(h['close'], 2)}")
        lines.append(f"Recent close history: [{', '.join(entries)}]")
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
    indices = data.get("indices", [])
    index_text = ""
    if indices:
        lines = []
        for idx in indices:
            chg = f"{idx['change_1d_pct']:+.2f}%" if idx.get("change_1d_pct") is not None else "N/A"
            latest_date = idx["5d_history"][-1]["date"] if idx.get("5d_history") else ""
            date_note = f" on {latest_date}" if latest_date else ""
            lines.append(f"  {idx['name']} ({idx['symbol']}): {idx['latest_close']} ({chg}{date_note})")
        index_text = "MARKET INDICES (benchmark context):\n" + "\n".join(lines) + "\n\n"
    system = (
        "You are a technical market analyst. Analyze the provided price and volume data. "
        "Compare the stock's performance to the market index (relative strength). "
        "Be concise. Output 3-5 bullet points covering: trend direction, volume signal, "
        "performance vs index, and momentum. Plain language, beginner-friendly. "
        "ALWAYS use the exact date from the data (e.g. '2026-04-02') when referencing any price or index move. "
        "NEVER use vague terms like 'today', 'yesterday', 'last trading day', or 'recently' — always state the specific date."
    )
    user = (
        f"Stock: {name} ({ticker})\n\n"
        f"PRICE & VOLUME DATA:\n{_price_summary(data['price'])}\n\n"
        f"{index_text}"
        f"ALERTS: {json.dumps(data['alerts'], ensure_ascii=False, default=str)[:500]}\n\n"
        "Write a concise technical analysis (max 200 words). Include comparison to the index."
    )
    return _call_llm(system, user, provider, api_key, max_tokens=400)


def run_fundamentals_analyst(data: dict, provider: str, api_key: str) -> str:
    ticker = data["ticker"]
    name   = data["info"].get("name", ticker)
    fund   = data["fundamentals"]
    info   = data["info"]
    market = data.get("market", "")

    # If yfinance returned nothing (SSL errors, rate-limit, etc.), fall back to web search
    web_fund_text = ""
    if not fund:
        try:
            try:
                from web.agent_llm import _web_search, _apply_market_suffix
            except ImportError:
                from agent_llm import _web_search, _apply_market_suffix
            _YF_SUFFIX = {"HK": ".HK", "JP": ".T", "AU": ".AX", "IN": ".NS",
                          "KR": ".KS", "TW": ".TW", "DE": ".DE", "FR": ".PA",
                          "UK": ".L", "BR": ".SA", "SA": ".SR"}
            suffix = _YF_SUFFIX.get(market, "")
            yf_ticker = ticker if (not suffix or ticker.endswith(suffix)) else ticker + suffix
            r1 = _web_search(
                f'"{name}" {yf_ticker} annual revenue profit earnings 2024 2025 results',
                max_results=4, body_chars=700)
            r2 = _web_search(
                f'{yf_ticker} PE ratio valuation analyst target 2025',
                max_results=3, body_chars=600)
            web_fund_text = f"=== Web search fallback (yfinance unavailable) ===\n{r1}\n\n{r2}"
        except Exception:
            pass

    system = (
        "You are a fundamental analyst. Assess the company's financial health using the exact numbers provided. "
        "ALWAYS cite specific figures (e.g. 'Q3 FY2026 revenue was ¥2,848亿', 'net profit margin 8.9%', 'P/E 21.5x'). "
        "NEVER make vague statements like 'revenue grew' without giving the actual number. "
        "Cover: (1) latest quarterly revenue & profit with YoY comparison, "
        "(2) valuation multiples (P/E, P/B, forward P/E), "
        "(3) profitability margins, (4) debt situation. "
        "Be concise and beginner-friendly. If a number is missing, say so explicitly. "
        "NEVER start with 'I'm sorry' or 'Unfortunately' — lead with what data you DO have."
    )
    # Separate out analyst ratings to highlight them
    analyst_text = ""
    targets = fund.get("analyst_price_targets", {})
    recs    = fund.get("analyst_recommendations", {})
    changes = fund.get("recent_rating_changes", [])
    if targets or recs or changes:
        parts = []
        if targets:
            parts.append(f"Analyst price targets: {json.dumps(targets, ensure_ascii=False)}")
        if recs:
            parts.append(f"Analyst recommendations: {json.dumps(recs, ensure_ascii=False)}")
        if changes:
            parts.append(f"Recent rating changes (last 5): {json.dumps(changes, ensure_ascii=False, default=str)}")
        analyst_text = "\n\nANALYST RATINGS:\n" + "\n".join(parts)

    if fund:
        fund_text = f"FUNDAMENTAL DATA (yfinance):\n{_fmt(fund)[:1800]}"
    elif web_fund_text:
        fund_text = f"FUNDAMENTAL DATA (web search — yfinance unavailable):\n{web_fund_text[:2000]}"
    else:
        fund_text = "Fundamental data unavailable from both yfinance and web search."

    user = (
        f"Stock: {name} ({ticker})\n"
        f"Sector: {info.get('sector', 'N/A')}  Industry: {info.get('industry', 'N/A')}\n\n"
        f"{fund_text}"
        f"{analyst_text}\n\n"
        "Write a concise fundamentals analysis (max 250 words). "
        "You MUST include specific numbers — do not generalize. "
        "If analyst targets are available, state the consensus target vs current price."
    )
    return _call_llm(system, user, provider, api_key, max_tokens=450)


def run_news_analyst(data: dict, provider: str, api_key: str) -> str:
    ticker = data["ticker"]
    name   = data["info"].get("name", ticker)
    market = data.get("market", "")
    news   = data["news"]

    try:
        from web.agent_llm import _web_search, _apply_market_suffix
    except ImportError:
        from agent_llm import _web_search, _apply_market_suffix

    # Build a short, searchable name (drop legal suffixes that hurt search quality)
    short_name = name.split("(")[0].strip() if "(" in name else name
    short_name = short_name.replace(" Company Limited", "").replace(" Co., Ltd", "").strip()
    yf_ticker = _apply_market_suffix(ticker, market)

    # Run two targeted searches: recent news + earnings/results
    web_parts = []
    web_parts.append(_web_search(
        f"{short_name} {yf_ticker} news 2025",
        max_results=4, body_chars=600))
    web_parts.append(_web_search(
        f"{short_name} earnings results analyst 2025",
        max_results=3, body_chars=600))
    web_raw = "\n\n".join(web_parts)

    if news:
        news_text = (
            "=== Local pipeline (cached headlines) ===\n" + _fmt(news)[:600] +
            "\n\n=== Live web search ===\n" + web_raw[:2000]
        )
        source_label = "local pipeline + live web search"
    else:
        news_text = web_raw[:2500]
        source_label = "live web search"

    system = (
        "You are a news and sentiment analyst. Summarise ALL available information about this stock. "
        "NEVER say there is 'no news' or dismiss results as 'general information' — "
        "use everything in the search results, even company profile snippets or stock page descriptions. "
        "Extract: (1) any recent events, earnings, product launches, or partnerships, "
        "(2) overall investor sentiment from the language used, "
        "(3) any specific numbers (revenue, profit, growth%) mentioned. "
        "CITATION FORMAT: after each claim, include a markdown link — example: "
        "'Sales grew 12% *(The Edge Singapore: https://theedgesingapore.com/...)* ' "
        "The URL is in the search results on the line after the body text. Use it. "
        "Be concise and beginner-friendly. NEVER start with 'Unfortunately' or 'I'm sorry'."
    )
    user = (
        f"Stock: {name} ({ticker})\n"
        f"News source: {source_label}\n\n"
        f"SEARCH RESULTS (format: • Title / body / URL on next line):\n{news_text}\n\n"
        "Write a concise news/sentiment analysis (max 200 words). "
        "For every claim, add an inline markdown link using the URL from the search result. "
        "Example: 'Revenue ¥43B *([Yahoo Finance](https://finance.yahoo.com/...))*' "
        "Include any numbers you find."
    )
    return _call_llm(system, user, provider, api_key, max_tokens=400)


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

def local_trading_analyze(ticker: str, data_dir: str, _allow_us: bool = False) -> str:
    """
    Run full multi-agent analysis for a non-US ticker.
    Pass _allow_us=True to bypass the US guard (used as TradingAgents fallback for ADRs).

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
    if market == "US" and not _allow_us:
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
            dfs = []
            for f in files[-2:]:
                try:
                    dfs.append(pd.read_parquet(f))
                except Exception:
                    pass
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                if "ticker" in df.columns:
                    df = df.sort_values("date") if "date" in df.columns else df
                    # Keep most recent valid close per ticker
                    df_valid = df[df["close"].notna()] if "close" in df.columns else df
                    if df_valid.empty:
                        df_valid = df
                    df = df_valid.groupby("ticker").last().reset_index()
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


def local_peers_analyze(ticker: str, market: str, data_dir: str) -> str:
    """Fixed-format peer comparison report using local price/universe data.

    Produces a structured report with peer performance table, relative strength
    ranking, volume activity, and a summary — similar to local_sector_analyze.
    """
    try:
        from web.agent_llm import get_llm_provider
    except ImportError:
        from agent_llm import get_llm_provider

    provider, api_key = get_llm_provider()
    if not provider:
        return "⚙ LLM API key not configured."

    date = __import__("datetime").datetime.now().strftime("%Y-%m-%d")

    # ── 1. Load peer data ────────────────────────────────────────────
    # Normalize ticker — strip exchange suffix (.SS, .SZ, .HK, etc.)
    ticker = ticker.split(".")[0] if "." in ticker else ticker

    raw = _load_local_peers(ticker, market, data_dir)
    if raw.startswith("(local_peers:"):
        return f"# Peer Comparison: {ticker}\n\n{raw}"

    # ── 2. Also load anchor stock's own price data ───────────────────
    anchor_info = _load_local_universe(ticker, market, data_dir)
    anchor_price = _load_local_price(ticker, market, data_dir)
    anchor_name = anchor_info.get("name", ticker)
    anchor_sector = anchor_info.get("sector", "N/A")
    anchor_industry = anchor_info.get("industry") or anchor_info.get("subsector", "N/A")

    anchor_line = f"{ticker} ({anchor_name})"
    if anchor_price:
        c = anchor_price.get("close")
        r1 = anchor_price.get("return_1d")
        r20 = anchor_price.get("return_20d")
        vr = anchor_price.get("volume_ratio")
        parts = []
        if c is not None:   parts.append(f"close={c:.2f}" if isinstance(c, float) else f"close={c}")
        if r1 is not None:  parts.append(f"1d={r1:+.1%}")
        if r20 is not None: parts.append(f"20d={r20:+.1%}")
        if vr is not None:  parts.append(f"vol_ratio={vr:.1f}x")
        if parts:
            anchor_line += " | " + " | ".join(parts)

    # ── 3. Load financials snapshot ──────────────────────────────────
    fin_snap = {}
    fin_path = os.path.join(data_dir, "markets", market, "financials.parquet")
    if os.path.exists(fin_path):
        try:
            fin_df = pd.read_parquet(fin_path)
            if not fin_df.empty and "ticker" in fin_df.columns and "period_end" in fin_df.columns:
                fin_df["period_end"] = pd.to_datetime(fin_df["period_end"], errors="coerce")
                fin_df = fin_df.dropna(subset=["period_end"])
                for tkr, grp in fin_df.groupby("ticker"):
                    annual = grp[grp["period_type"] == "annual"].sort_values("period_end")
                    qtrly  = grp[grp["period_type"] == "quarterly"].sort_values("period_end")
                    fin_snap[str(tkr)] = {
                        "annual":  annual.iloc[-1].to_dict() if not annual.empty else None,
                        "quarter": qtrly.iloc[-1].to_dict()  if not qtrly.empty  else None,
                    }
        except Exception:
            pass

    # Build financials table text
    def _fin_row(tkr, name, fin):
        ann = fin.get("annual")  if fin else None
        qtr = fin.get("quarter") if fin else None
        row = f"{tkr} ({name})"
        if ann:
            fy  = ann.get("fiscal_year") or str(ann.get("period_end", ""))[:4]
            rev = ann.get("revenue"); ry = ann.get("revenue_yoy")
            np_ = ann.get("net_profit"); ny = ann.get("net_profit_yoy")
            parts = []
            if rev is not None: parts.append(f"rev={rev:.1f}亿" + (f"({ry:+.1f}%)" if ry is not None else ""))
            if np_  is not None: parts.append(f"NP={np_:.2f}亿" + (f"({ny:+.1f}%)" if ny is not None else ""))
            if parts: row += f" | FY{fy}: {', '.join(parts)}"
        if qtr:
            qend = str(qtr.get("period_end", ""))[:10]
            rev  = qtr.get("revenue"); ry = qtr.get("revenue_yoy")
            np_  = qtr.get("net_profit"); ny = qtr.get("net_profit_yoy")
            parts = []
            if rev is not None: parts.append(f"rev={rev:.1f}亿" + (f"({ry:+.1f}%)" if ry is not None else ""))
            if np_  is not None: parts.append(f"NP={np_:.2f}亿" + (f"({ny:+.1f}%)" if ny is not None else ""))
            if parts: row += f" | {qend}: {', '.join(parts)}"
        return row

    # Parse ticker+name from the markdown table rows in raw
    # Table rows look like: | Ticker | Name | Close | ... (skip header/sep rows)
    fin_lines = [_fin_row(ticker, anchor_name, fin_snap.get(ticker, {}))]
    for line in raw.split("\n"):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) < 2:
            continue
        tk = cells[0]
        nm = cells[1]
        # Skip header rows (Ticker, 代码, ---, etc.) and separator rows
        if tk in ("Ticker", "代码", "---", "------") or tk.startswith("---"):
            continue
        if tk and tk != ticker:
            fin_lines.append(_fin_row(tk, nm, fin_snap.get(tk, {})))

    fin_block = "\n".join(fin_lines) if fin_lines else "(no financials data available)"
    has_financials = any("|" in l and ("FY" in l or "Q" in l or "rev=" in l) for l in fin_lines)

    # ── 4. Build compact data block for LLM ─────────────────────────
    data_block = (
        f"ANCHOR: {anchor_line}\n"
        f"Industry: {anchor_industry} / Sector: {anchor_sector}\n\n"
        f"PEERS:\n{raw}"
    )

    def _step(label, system, user, max_tok=400):
        try:
            return _call_llm(system, user, provider, api_key, max_tokens=max_tok) or f"({label}: no output)"
        except Exception as e:
            return f"({label} unavailable: {e})"

    # ── 5. Agent 1: Relative strength ranking ───────────────────────
    ranking = _step("Ranking",
        "You are a relative strength analyst. The data table is already shown to the user — "
        "do NOT repeat numbers from it. Provide only the ranking order and one-line interpretation per stock.",
        f"Date: {date}\n\n{data_block}\n\n"
        "Rank all peers strongest→weakest by 20d return. "
        "Format: `TICKER Name — one-line reason why it ranks here.` "
        "No numbers (they are in the table). Max 150 words.",
        300)

    # ── 6. Agent 2: Volume & activity analysis ───────────────────────
    volume = _step("Volume",
        "You are a volume and market activity analyst. The data table is already shown — "
        "do NOT repeat vol_ratio or price numbers. Only name the stocks with notable volume and interpret the signal.",
        f"Date: {date}\n\n{data_block}\n\n"
        "List only stocks with vol_ratio > 1.5x. For each: ticker, name, "
        "and whether it signals accumulation or distribution — one line each. Max 80 words.",
        200)

    # ── 7. Agent 3: Financials summary (only if data available) ──────
    fin_section = ""
    if has_financials:
        fin_section = _step("Financials",
            "You are a fundamental analyst. The financials table is already shown to the user — "
            "do NOT repeat revenue or profit figures. Provide interpretation only.",
            f"Date: {date}\n\nFinancials data:\n{fin_block}\n\n"
            "Answer in 3 bullets (no numbers, refer to tickers by name): "
            "1. Who leads on revenue growth and why it matters. "
            "2. Who is most profitable vs who is burning cash. "
            "3. Any notable divergence worth watching. Max 120 words.",
            250)

    # ── 8. Agent 4: Summary verdict ──────────────────────────────────
    verdict = _step("Verdict",
        "You are a peer comparison strategist. Give a concise verdict on the anchor stock's "
        "relative position vs its peers. Format exactly:\n"
        "**Relative Position: Leader / Middle / Laggard**\n"
        "**Key divergence:** 1 sentence comparing anchor to top peer.\n"
        "**Watch:** 1-2 peer tickers worth monitoring and why.\n"
        "**Beginner takeaway:** 1 plain-language sentence.",
        f"Anchor: {ticker} ({anchor_name})  Date: {date}\n\n"
        f"Ranking:\n{ranking[:400]}\n\nVolume:\n{volume[:200]}\n\n"
        + (f"Financials:\n{fin_section[:300]}\n\n" if fin_section else "")
        + "Final peer comparison verdict.",
        300)

    # ── 9. Assemble fixed-format report ─────────────────────────────
    sections = [
        f"# Peer Comparison: {ticker} ({anchor_name})",
        f"*Industry: {anchor_industry} · Sector: {anchor_sector} · {market} · {date}*",
        "",
        "---",
        "## 📊 Peer Data",
        f"**Anchor:** {anchor_line}",
        "",
        raw,
        "",
        "## 🏆 Relative Strength Ranking",
        ranking,
        "",
        "## 📦 Volume Activity",
        volume,
    ]
    if fin_section:
        sections += ["", "## 💰 Financials", fin_section]
    sections += ["", "## 💡 Verdict", verdict]
    return "\n".join(sections)


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
