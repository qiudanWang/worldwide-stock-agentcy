"""Fetch market capitalization for CN and US stocks."""

import akshare as ak
import yfinance as yf
import pandas as pd
from src.common.logger import get_logger

log = get_logger("market.cap")


def fetch_cn_market_cap_batch(tickers):
    """Fetch market cap for A-share stocks via individual stock info.

    Uses stock_individual_info_em per ticker (reliable but slower).
    Returns DataFrame with columns: ticker, market_cap (RMB yuan).
    """
    rows = []
    for i, ticker in enumerate(tickers):
        if i > 0 and i % 50 == 0:
            log.info(f"  Market cap progress: {i}/{len(tickers)}")
        try:
            df = ak.stock_individual_info_em(symbol=ticker)
            cap_row = df[df["item"] == "总市值"]
            if not cap_row.empty:
                cap = float(cap_row["value"].iloc[0])
            else:
                cap = None
            rows.append({"ticker": ticker, "market_cap": cap})
        except Exception:
            rows.append({"ticker": ticker, "market_cap": None})

    result = pd.DataFrame(rows)
    got = result["market_cap"].notna().sum()
    log.info(f"Got market cap for {got}/{len(tickers)} CN stocks")
    return result


def fetch_cn_market_cap_spot():
    """Fetch market cap for all A-share stocks via spot data (fast but unreliable).

    Falls back to yfinance if akshare spot API fails (e.g. outside China).
    Returns DataFrame with columns: ticker, market_cap (RMB yuan).
    """
    try:
        log.info("Fetching CN spot data for market cap...")
        df = ak.stock_zh_a_spot_em()
        df = df[["代码", "总市值"]].copy()
        df.columns = ["ticker", "market_cap"]
        df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
        log.info(f"Got market cap for {len(df)} CN stocks via spot")
        return df
    except Exception as e:
        log.warning(f"Spot API failed ({e}), falling back to yfinance...")
        return pd.DataFrame(columns=["ticker", "market_cap"])


def fetch_cn_market_cap_yf(tickers):
    """Fetch CN A-share market cap via yfinance as fallback.

    Uses .SS suffix for Shanghai (6xx, 9xx) and .SZ for Shenzhen (0xx, 3xx).
    Returns DataFrame with columns: ticker, market_cap (CNY yuan).
    """
    from src.common.rate_limiter import yf_limiter
    rows = []
    for ticker in tickers:
        suffix = ".SS" if ticker.startswith(("6", "9")) else ".SZ"
        symbol = ticker + suffix
        cap = None
        try:
            with yf_limiter:
                fi = yf.Ticker(symbol).fast_info
                cap = getattr(fi, "market_cap", None)
        except Exception:
            pass
        rows.append({"ticker": ticker, "market_cap": cap})

    result = pd.DataFrame(rows)
    got = result["market_cap"].notna().sum()
    log.info(f"Got market cap for {got}/{len(tickers)} CN stocks via yfinance")
    return result


def fetch_us_market_cap(tickers):
    """Fetch market cap for US stocks via yfinance.

    Returns DataFrame with columns: ticker, market_cap
    市值 unit: USD
    """
    rows = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            cap = info.get("marketCap", None)
            rows.append({"ticker": ticker, "market_cap": cap})
        except Exception as e:
            log.warning(f"Failed to fetch market cap for {ticker}: {e}")
            rows.append({"ticker": ticker, "market_cap": None})

    df = pd.DataFrame(rows)
    log.info(f"Got market cap for {len(df)} US stocks")
    return df


def fetch_all_market_caps(us_tickers, cn_tickers=None):
    """Fetch market caps for both CN and US stocks.

    For CN: tries fast spot API first, falls back to individual lookup.
    Returns DataFrame with columns: ticker, market_cap
    """
    # CN market cap
    cn = fetch_cn_market_cap_spot()
    if cn.empty and cn_tickers:
        cn = fetch_cn_market_cap_batch(cn_tickers)

    # US market cap
    us = fetch_us_market_cap(us_tickers)
    return pd.concat([cn, us], ignore_index=True)


def save_market_caps(cap_df):
    """Save market cap data to a cached parquet file."""
    from src.common.config import get_data_path
    if cap_df.empty:
        return
    path = get_data_path("processed", "market_cap.parquet")
    cap_df.to_parquet(path, index=False)
    log.info(f"Saved market cap for {len(cap_df)} stocks to {path}")


def load_market_caps():
    """Load cached market cap data from all markets. Returns dict of ticker -> market_cap."""
    import glob
    from src.common.config import get_data_path
    frames = []
    for p in glob.glob(get_data_path("markets", "*", "market_cap.parquet")):
        try:
            frames.append(pd.read_parquet(p))
        except Exception:
            pass
    try:
        frames.append(pd.read_parquet(get_data_path("processed", "market_cap.parquet")))
    except Exception:
        pass
    if not frames:
        return {}
    combined = pd.concat(frames, ignore_index=True).dropna(subset=["market_cap"])
    combined = combined.sort_values("market_cap", ascending=False).drop_duplicates("ticker")
    return dict(zip(combined["ticker"], combined["market_cap"]))


CURRENCY_SYMBOLS = {
    "CN": "¥", "US": "$", "HK": "HK$", "JP": "¥", "IN": "₹",
    "UK": "£", "DE": "€", "FR": "€", "KR": "₩", "TW": "NT$",
    "AU": "A$", "BR": "R$", "SA": "SAR",
}


def format_market_cap(value, market="CN"):
    """Format market cap for display.

    CN: in 亿 (100M RMB)
    Others: in B/T with currency symbol
    """
    if pd.isna(value) or value is None:
        return "-"
    if market == "CN":
        yi = value / 1e8
        if yi >= 1000:
            return f"{yi / 1000:.1f}千亿"
        elif yi >= 1:
            return f"{yi:.1f}亿"
        else:
            return f"{value / 1e4:.0f}万"
    else:
        sym = CURRENCY_SYMBOLS.get(market, "$")
        if value >= 1e12:
            return f"{sym}{value / 1e12:.2f}T"
        elif value >= 1e9:
            return f"{sym}{value / 1e9:.1f}B"
        elif value >= 1e6:
            return f"{sym}{value / 1e6:.0f}M"
        else:
            return f"{sym}{value:,.0f}"
