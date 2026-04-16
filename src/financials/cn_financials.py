"""CN A-share financials fetcher via akshare.

Fetches multi-period (annual + quarterly) revenue, net profit, and YoY growth
for CN tech universe stocks. Saves to data/markets/CN/financials.parquet.

Schema (long format — one row per ticker × period):
  ticker, market, period_end, period_type, fiscal_year,
  revenue, revenue_yoy, net_profit, net_profit_yoy,
  net_margin, roe, eps
  (revenue and net_profit in 亿元)
"""

import pandas as pd
from src.common.logger import get_logger

log = get_logger("financials.cn")

_PERIOD_SUFFIX = {
    "03-31": "Q1", "06-30": "Q2", "09-30": "Q3", "12-31": "annual"
}


def _safe_float(val, unit_suffix=""):
    """Parse a value that may carry a unit suffix like 亿, 万, or % sign."""
    try:
        if val is None or str(val).strip() in ("", "-", "None", "nan", "--", "不适用"):
            return None
        s = str(val).strip()
        # Strip trailing % (YoY growth columns)
        s = s.rstrip("%")
        # Handle unit suffixes: 亿 (×10^8), 万 (×10^4)
        if s.endswith("亿"):
            return float(s[:-1])          # already in 亿, keep as-is
        if s.endswith("万"):
            return round(float(s[:-1]) / 10000, 6)  # convert 万 → 亿
        return float(s)
    except (ValueError, TypeError):
        return None


def _period_type(period_str: str) -> str:
    for suffix, ptype in _PERIOD_SUFFIX.items():
        if str(period_str).endswith(suffix):
            return ptype
    return "unknown"


def fetch_cn_financials_multiperiod(ticker: str) -> list:
    """Fetch multi-period financials for a single CN A-share ticker.

    Returns a list of dicts, one per reporting period.
    """
    try:
        import akshare as ak
        df = ak.stock_financial_abstract_ths(symbol=ticker, indicator="按报告期")
        if df is None or df.empty:
            return []

        # Discover columns dynamically
        period_col  = next((c for c in df.columns if "报告期" in c), None)
        rev_col     = next((c for c in df.columns if "营业总收入" in c and "同比" not in c), None)
        rev_yoy_col = next((c for c in df.columns if ("营业总收入同比" in c) or ("营收" in c and "同比" in c)), None)
        np_col      = next((c for c in df.columns if c in ("净利润", "归母净利润")), None)
        np_yoy_col  = next((c for c in df.columns if "净利润同比" in c or "归母净利润同比" in c), None)
        margin_col  = next((c for c in df.columns if "销售净利率" in c or ("净利率" in c and "同比" not in c)), None)
        roe_col     = next((c for c in df.columns if "净资产收益率" in c), None)
        eps_col     = next((c for c in df.columns if "基本每股收益" in c or "每股收益" in c), None)

        if not period_col:
            return []

        rows = []
        for _, r in df.iterrows():
            period = str(r[period_col]).strip()
            if not period or period in ("None", "nan", ""):
                continue

            # akshare returns values already suffixed with 亿/万 — _safe_float handles the conversion
            rev = _safe_float(r[rev_col]) if rev_col else None
            np_ = _safe_float(r[np_col]) if np_col else None

            ptype = _period_type(period)
            fy = int(period[:4]) if len(period) >= 4 else None

            rows.append({
                "ticker": ticker,
                "market": "CN",
                "period_end": period,
                "period_type": ptype,
                "fiscal_year": fy,
                "revenue": rev,
                "revenue_yoy": _safe_float(r[rev_yoy_col]) if rev_yoy_col else None,
                "net_profit": np_,
                "net_profit_yoy": _safe_float(r[np_yoy_col]) if np_yoy_col else None,
                "net_margin": _safe_float(r[margin_col]) if margin_col else None,
                "roe": _safe_float(r[roe_col]) if roe_col else None,
                "eps": _safe_float(r[eps_col]) if eps_col else None,
            })

        return rows

    except Exception as e:
        log.debug(f"[CN] {ticker}: fetch failed — {e}")
        return []


def fetch_cn_financials_batch(tickers: list, max_workers: int = 8) -> pd.DataFrame:
    """Fetch multi-period financials for a list of A-share tickers in parallel.

    Uses a thread pool to run concurrent akshare HTTP requests, reducing
    runtime from ~60 minutes (sequential) to ~8 minutes (8 workers).

    Returns a long-format DataFrame (one row per ticker × period).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_rows = []
    failed = 0
    completed = 0
    total = len(tickers)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_cn_financials_multiperiod, t): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                rows = future.result()
                if rows:
                    all_rows.extend(rows)
                else:
                    failed += 1
            except Exception as e:
                log.debug(f"[CN] {ticker}: exception — {e}")
                failed += 1
            completed += 1
            if completed % 100 == 0:
                log.info(f"  [CN] Progress: {completed}/{total} ({len(all_rows)} periods, {failed} failed)")

    if not all_rows:
        log.warning("[CN] No financials fetched")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df = df.sort_values(["ticker", "period_end"]).reset_index(drop=True)
    log.info(f"[CN] Fetched {df['ticker'].nunique()} stocks, {len(df)} periods total ({failed} failed)")
    return df


# ── Backward-compat shim (used by old normalize_financials.py) ─────────────
def fetch_cn_financials(ticker):
    rows = fetch_cn_financials_multiperiod(ticker)
    if not rows:
        return {}
    latest = rows[-1]
    return {
        "ticker": latest["ticker"],
        "market": "CN",
        "report_date": str(latest.get("period_end", "")),
        "eps": latest.get("eps"),
        "revenue": latest.get("revenue"),
        "revenue_growth": latest.get("revenue_yoy"),
        "net_profit": latest.get("net_profit"),
        "net_margin": latest.get("net_margin"),
        "roe": latest.get("roe"),
    }
