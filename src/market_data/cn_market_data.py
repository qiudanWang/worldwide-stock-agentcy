import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from src.common.config import get_settings
from src.common.logger import get_logger

log = get_logger("market.cn")

_BATCH_SIZE = 50  # yfinance handles batches well


def _ticker_to_yf(ticker: str) -> str:
    """Convert CN ticker to Yahoo Finance symbol.

    Shanghai (SSE):  6xxxxx → .SS,  900xxx (B-share) → .SS
    Shenzhen (SZSE): 0xxxxx, 2xxxxx, 3xxxxx → .SZ
    Beijing (BSE):   8xxxxx, 9xxxxx (except 900xxx) → not on Yahoo, return None
    """
    if ticker.startswith("6") or ticker.startswith("900"):
        return ticker + ".SS"
    if ticker.startswith("8") or (ticker.startswith("9") and not ticker.startswith("900")):
        return None  # Beijing Stock Exchange — not available on Yahoo Finance
    return ticker + ".SZ"  # 0xxxxx, 2xxxxx, 3xxxxx → Shenzhen


def fetch_cn_stock_history(ticker, days=None):
    """Fetch daily OHLCV for a single A-share stock via yfinance."""
    if days is None:
        days = get_settings()["market_data"]["cn_history_days"]
    yf_sym = _ticker_to_yf(ticker)
    if yf_sym is None:
        return pd.DataFrame()  # Not available on Yahoo Finance
    try:
        raw = yf.download(yf_sym, period=f"{days}d", progress=False,
                          auto_adjust=True, actions=False)
        if raw.empty:
            return pd.DataFrame()
        raw = raw.reset_index()
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        df = raw.rename(columns={"Date": "date", "Open": "open", "High": "high",
                                  "Low": "low", "Close": "close", "Volume": "volume"})
        df["ticker"] = ticker
        df["market"] = "CN"
        df["turnover"] = df["close"] * df["volume"]
        return df[["date", "ticker", "market", "open", "high", "low",
                   "close", "volume", "turnover"]]
    except Exception as e:
        log.warning(f"Failed to fetch {ticker} ({yf_sym}): {e}")
        return pd.DataFrame()


def fetch_cn_batch(tickers):
    """Fetch history for all CN tickers using yfinance batch downloads."""
    if not tickers:
        return pd.DataFrame()

    days = get_settings()["market_data"]["cn_history_days"]
    all_data = []
    total = len(tickers)

    # Process in batches of _BATCH_SIZE
    for batch_start in range(0, total, _BATCH_SIZE):
        batch = tickers[batch_start:batch_start + _BATCH_SIZE]
        # Skip tickers not available on Yahoo Finance (e.g. Beijing Stock Exchange)
        valid = [(t, _ticker_to_yf(t)) for t in batch if _ticker_to_yf(t) is not None]
        if not valid:
            continue
        yf_syms = [yf_sym for _, yf_sym in valid]
        ticker_map = {yf_sym: t for t, yf_sym in valid}  # yf_sym → original ticker

        log.info(f"  Fetching batch {batch_start + 1}-{min(batch_start + _BATCH_SIZE, total)}/{total}")
        try:
            raw = yf.download(
                " ".join(yf_syms),
                period=f"{days}d",
                progress=False,
                auto_adjust=True,
                actions=False,
                group_by="ticker",
            )
            if raw.empty:
                continue

            # Normalize MultiIndex columns: yfinance returns (Price, Ticker) levels
            if isinstance(raw.columns, pd.MultiIndex):
                ticker_level = "Ticker" if "Ticker" in raw.columns.names else 1
                available = raw.columns.get_level_values(ticker_level).unique().tolist()
                for yf_sym in yf_syms:
                    if yf_sym not in available:
                        continue
                    try:
                        df = raw.xs(yf_sym, axis=1, level=ticker_level).copy().reset_index()
                        df.columns = [c.lower() for c in df.columns]
                        df = df.rename(columns={"price": "date"})  # Date index becomes 'date'
                        if "date" not in df.columns:
                            df = df.rename(columns={df.columns[0]: "date"})
                        if df.empty or "close" not in df.columns or df["close"].isna().all():
                            continue
                        orig_ticker = ticker_map[yf_sym]
                        df["ticker"] = orig_ticker
                        df["market"] = "CN"
                        df["volume"] = df.get("volume", 0)
                        df["turnover"] = df["close"] * df["volume"]
                        cols = ["date", "ticker", "market", "open", "high",
                                "low", "close", "volume", "turnover"]
                        all_data.append(df[[c for c in cols if c in df.columns]])
                    except Exception as e:
                        log.debug(f"  Skip {yf_sym}: {e}")
            else:
                # Single ticker — flat DataFrame
                df = raw.reset_index()
                df.columns = [c.lower() for c in df.columns]
                if not df.empty and "close" in df.columns:
                    if "date" not in df.columns:
                        df = df.rename(columns={df.columns[0]: "date"})
                    orig_ticker = ticker_map.get(yf_syms[0], valid[0][0])
                    df["ticker"] = orig_ticker
                    df["market"] = "CN"
                    df["volume"] = df.get("volume", 0)
                    df["turnover"] = df["close"] * df["volume"]
                    cols = ["date", "ticker", "market", "open", "high",
                            "low", "close", "volume", "turnover"]
                    all_data.append(df[[c for c in cols if c in df.columns]])

        except Exception as e:
            log.warning(f"  Batch {batch_start}-{batch_start + _BATCH_SIZE} failed: {e}")
            # Fall back to individual fetches for this batch
            for ticker in batch:
                df = fetch_cn_stock_history(ticker, days)
                if not df.empty:
                    all_data.append(df)

    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)
