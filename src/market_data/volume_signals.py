import pandas as pd
from src.common.logger import get_logger

log = get_logger("market.volume")


def compute_volume_signals(market_data):
    """Compute volume ratio and return signals for each ticker.

    Adds columns:
        avg_volume_20d: 20-day rolling average volume
        volume_ratio: today's volume / avg_volume_20d
        return_1d: 1-day price return
        return_5d: 5-day price return
        return_20d: 20-day price return
    """
    if market_data.empty:
        return market_data

    result = []
    for ticker, group in market_data.groupby("ticker"):
        df = group.sort_values("date").copy()

        df["avg_volume_20d"] = df["volume"].rolling(window=20, min_periods=5).mean()
        df["volume_ratio"] = df["volume"] / df["avg_volume_20d"]

        df["return_1d"] = df["close"].pct_change(1)
        df["return_5d"] = df["close"].pct_change(5)
        df["return_20d"] = df["close"].pct_change(20)

        result.append(df)

    combined = pd.concat(result, ignore_index=True)
    log.info(f"Computed volume signals for {combined['ticker'].nunique()} tickers")
    return combined


def get_latest_signals(market_data_with_signals):
    """Get the most recent day's signals for each ticker."""
    if market_data_with_signals.empty:
        return market_data_with_signals

    latest = (
        market_data_with_signals
        .sort_values("date")
        .groupby("ticker")
        .tail(1)
        .reset_index(drop=True)
    )
    return latest
