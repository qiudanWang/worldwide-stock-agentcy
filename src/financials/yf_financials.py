"""Multi-period financials fetcher for non-CN markets via yfinance.

Fetches annual (last 4 FY) + quarterly (last 4 quarters) income statement data.
Calculates YoY growth from the fetched periods. Saves per-market to:
  data/markets/{MARKET}/financials.parquet

Schema (long format — one row per ticker × period):
  ticker, market, period_end, period_type (annual/quarterly),
  fiscal_year, revenue, net_profit, gross_profit, operating_income,
  revenue_yoy, net_profit_yoy, net_margin, gross_margin
  (all monetary values in 亿 units of local currency)
"""

import time
import pandas as pd
from src.common.logger import get_logger

log = get_logger("financials.yf")

_YF_SUFFIX = {
    "HK": ".HK", "JP": ".T",  "AU": ".AX", "IN": ".NS",
    "KR": ".KS", "TW": ".TW", "DE": ".DE", "FR": ".PA",
    "UK": ".L",  "BR": ".SA", "SA": ".SR", "US": "",
}

_REVENUE_ROWS = ["Total Revenue", "Revenue"]
_NET_INC_ROWS = ["Net Income", "Net Income Common Stockholders", "Net Income Applicable To Common Shares"]
_GROSS_ROWS   = ["Gross Profit"]
_OPER_ROWS    = ["Operating Income", "EBIT"]


def _pick(df: pd.DataFrame, candidates: list):
    for name in candidates:
        if name in df.index:
            return df.loc[name]
    return None


def _yoy(current, prior):
    try:
        if prior is not None and abs(prior) > 0:
            return round((current - prior) / abs(prior) * 100, 2)
    except Exception:
        pass
    return None


def _to_yi(val) -> float | None:
    """Convert raw yfinance value (in base currency units) to 亿 (×10^8)."""
    try:
        if val is None or (hasattr(val, '__float__') and pd.isna(val)):
            return None
        return round(float(val) / 1e8, 4)
    except Exception:
        return None


def _parse_income(fin: pd.DataFrame, period_type: str, ticker: str, market: str) -> list:
    """Parse a yfinance income statement DataFrame into period dicts."""
    if fin is None or fin.empty:
        return []

    # Columns are dates (most recent first after sort)
    cols = sorted(fin.columns, reverse=True)
    rows = []

    for i, col in enumerate(cols):
        def _v(candidates):
            s = _pick(fin, candidates)
            return _to_yi(s.get(col) if s is not None else None)

        rev = _v(_REVENUE_ROWS)
        np_ = _v(_NET_INC_ROWS)
        gp  = _v(_GROSS_ROWS)
        oi  = _v(_OPER_ROWS)

        # YoY: quarterly → compare to same quarter last year (i+4), annual → prior year (i+1)
        prior_col = None
        if period_type == "quarterly" and i + 4 < len(cols):
            prior_col = cols[i + 4]
        elif period_type == "annual" and i + 1 < len(cols):
            prior_col = cols[i + 1]

        rev_yoy = np_yoy = None
        if prior_col is not None:
            def _pv(candidates):
                s = _pick(fin, candidates)
                return _to_yi(s.get(prior_col) if s is not None else None)
            rev_yoy = _yoy(rev, _pv(_REVENUE_ROWS)) if rev is not None else None
            np_yoy  = _yoy(np_, _pv(_NET_INC_ROWS))  if np_ is not None else None

        net_margin = gross_margin = None
        if rev and rev != 0:
            if np_ is not None: net_margin   = round(np_ / rev * 100, 2)
            if gp  is not None: gross_margin = round(gp  / rev * 100, 2)

        try:
            period_end = col.date().isoformat() if hasattr(col, "date") else str(col)[:10]
        except Exception:
            period_end = str(col)[:10]

        fy = int(period_end[:4]) if len(period_end) >= 4 else None

        rows.append({
            "ticker": ticker,
            "market": market,
            "period_end": period_end,
            "period_type": period_type,
            "fiscal_year": fy,
            "revenue": rev,
            "net_profit": np_,
            "gross_profit": gp,
            "operating_income": oi,
            "revenue_yoy": rev_yoy,
            "net_profit_yoy": np_yoy,
            "net_margin": net_margin,
            "gross_margin": gross_margin,
        })

    return rows


def fetch_yf_financials_one(ticker: str, market: str) -> list:
    """Fetch annual + quarterly financials for a single ticker."""
    import yfinance as yf

    suffix = _YF_SUFFIX.get(market, "")
    yf_sym = ticker if (not suffix or str(ticker).endswith(suffix)) else ticker + suffix

    try:
        t = yf.Ticker(yf_sym)
        rows = []

        try:
            rows.extend(_parse_income(t.financials, "annual", ticker, market))
        except Exception:
            pass

        try:
            rows.extend(_parse_income(t.quarterly_financials, "quarterly", ticker, market))
        except Exception:
            pass

        return rows
    except Exception as e:
        log.debug(f"[{market}] {ticker}: {e}")
        return []


def fetch_yf_financials_batch(tickers: list, market: str, delay: float = 0.5) -> pd.DataFrame:
    """Fetch multi-period financials for a list of tickers in a market.

    Returns a long-format DataFrame. Returns empty DataFrame on total failure.
    """
    all_rows = []
    failed = 0

    for i, ticker in enumerate(tickers):
        rows = fetch_yf_financials_one(ticker, market)
        if rows:
            all_rows.extend(rows)
        else:
            failed += 1

        if (i + 1) % 20 == 0:
            log.info(f"  [{market}] Progress: {i+1}/{len(tickers)} ({len(all_rows)} periods, {failed} failed)")
            time.sleep(delay)

    if not all_rows:
        log.warning(f"[{market}] No financials fetched for any of {len(tickers)} tickers")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df = df.sort_values(["ticker", "period_end"]).reset_index(drop=True)
    log.info(f"[{market}] Financials: {df['ticker'].nunique()}/{len(tickers)} stocks, {len(df)} periods")
    return df
