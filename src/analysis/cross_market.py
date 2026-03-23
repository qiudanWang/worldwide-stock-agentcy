"""Cross-market divergence detection."""

import pandas as pd
from src.common.logger import get_logger

log = get_logger("analysis.crossmarket")


def detect_cross_market_divergence(sector_perf, threshold=0.03):
    """Detect same-sector divergence across markets.

    When the same sector moves in opposite directions in different markets,
    it may signal a trading opportunity or market-specific risk.

    Args:
        sector_perf: DataFrame with: market, sector, avg_return_1d
        threshold: Minimum divergence to flag.

    Returns:
        List of divergence events.
    """
    if sector_perf.empty:
        return []

    divergences = []
    sectors = sector_perf["sector"].unique()

    for sector in sectors:
        sector_data = sector_perf[sector_perf["sector"] == sector]
        if len(sector_data) < 2:
            continue

        markets = sector_data.sort_values("avg_return_1d")
        worst = markets.iloc[0]
        best = markets.iloc[-1]

        spread = best["avg_return_1d"] - worst["avg_return_1d"]
        if abs(spread) >= threshold:
            divergences.append({
                "sector": sector,
                "best_market": best["market"],
                "best_return": round(best["avg_return_1d"], 4),
                "worst_market": worst["market"],
                "worst_return": round(worst["avg_return_1d"], 4),
                "spread": round(spread, 4),
            })

    divergences.sort(key=lambda x: abs(x["spread"]), reverse=True)
    log.info(f"Cross-market divergence: {len(divergences)} events")
    return divergences


def detect_index_breakout(index_data, lookback_days=20):
    """Detect indices that break their N-day high or low.

    Args:
        index_data: DataFrame with: date, symbol, name, close, market
        lookback_days: Number of days for high/low range.

    Returns:
        List of breakout events.
    """
    if index_data.empty:
        return []

    breakouts = []

    for symbol in index_data["symbol"].unique():
        idx = index_data[index_data["symbol"] == symbol].sort_values("date")
        if len(idx) < lookback_days + 1:
            continue

        current = idx.iloc[-1]
        lookback = idx.iloc[-(lookback_days + 1):-1]

        high = lookback["close"].max()
        low = lookback["close"].min()
        price = current["close"]

        if price > high:
            breakouts.append({
                "symbol": symbol,
                "name": current.get("name", symbol),
                "market": current.get("market", ""),
                "type": "breakout_high",
                "price": round(price, 2),
                "threshold": round(high, 2),
                "lookback_days": lookback_days,
            })
        elif price < low:
            breakouts.append({
                "symbol": symbol,
                "name": current.get("name", symbol),
                "market": current.get("market", ""),
                "type": "breakout_low",
                "price": round(price, 2),
                "threshold": round(low, 2),
                "lookback_days": lookback_days,
            })

    log.info(f"Index breakouts: {len(breakouts)} events")
    return breakouts
