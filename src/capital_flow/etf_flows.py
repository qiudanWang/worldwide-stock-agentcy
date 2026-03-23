"""ETF proxy capital flow estimation for markets without direct flow data."""

import yfinance as yf
import pandas as pd
from src.common.logger import get_logger

log = get_logger("capital.etf")


def fetch_etf_flow_proxy(etf_symbol, market, days=60):
    """Estimate capital flows using country ETF volume as a proxy.

    Higher-than-average volume on a country ETF suggests capital movement.
    Returns DataFrame with: date, flow_type, volume, avg_volume_20d, volume_ratio, net_flow_proxy

    Args:
        etf_symbol: The ETF ticker (e.g., "EWJ" for Japan, "INDA" for India).
        market: Market code for tagging.
        days: Number of days of history.
    """
    try:
        t = yf.Ticker(etf_symbol)
        df = t.history(period=f"{days}d")
        if df.empty:
            log.warning(f"No data for ETF {etf_symbol}")
            return pd.DataFrame()

        df = df.reset_index()
        df = df.rename(columns={
            "Date": "date",
            "Close": "close",
            "Volume": "volume",
        })
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        # Compute volume signals
        df["avg_volume_20d"] = df["volume"].rolling(window=20, min_periods=5).mean()
        df["volume_ratio"] = df["volume"] / df["avg_volume_20d"]

        # Estimate net flow direction: price change * volume as proxy
        df["return_1d"] = df["close"].pct_change(1)
        df["net_flow_proxy"] = df["return_1d"] * df["volume"]

        df["flow_type"] = f"etf_proxy_{etf_symbol}"
        df["market"] = market

        result = df[["date", "market", "flow_type", "volume",
                      "avg_volume_20d", "volume_ratio",
                      "close", "return_1d", "net_flow_proxy"]].copy()

        log.info(f"[{market}] ETF proxy ({etf_symbol}): {len(result)} days")
        return result

    except Exception as e:
        log.warning(f"Failed to fetch ETF {etf_symbol}: {e}")
        return pd.DataFrame()
