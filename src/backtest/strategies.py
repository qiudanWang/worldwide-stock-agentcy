"""
Backtest Strategies
===================
Each strategy implements a `select(universe_df, history, date)` method that
returns a list of tickers to hold at rebalance.

universe_df  — the market universe (columns: ticker, sector, subsector, ...)
history      — dict[ticker → ohlcv_df]  (all available history up to `date`)
date         — the signal date (pd.Timestamp); execution at next open
"""

from __future__ import annotations

import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    name: str
    description: str
    timeframe: str   # "daily" | "weekly" | "monthly" | "yearly"

    @abstractmethod
    def select(
        self,
        universe_df: pd.DataFrame,
        history: dict[str, pd.DataFrame],
        date: pd.Timestamp,
    ) -> list[str]:
        """Return list of tickers to hold. Max `top_n` items."""
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_n_closes(history: dict[str, pd.DataFrame], ticker: str, date: pd.Timestamp, n: int) -> Optional[pd.Series]:
    """Return the last n closing prices up to and including `date`."""
    df = history.get(ticker)
    if df is None or df.empty:
        return None
    df = df[df["date"] <= date].tail(n)
    if len(df) < n:
        return None
    return df["close"].reset_index(drop=True)


def _momentum(closes: pd.Series) -> float:
    """Simple price momentum: last / first - 1."""
    if closes is None or len(closes) < 2:
        return float("-inf")
    return float(closes.iloc[-1] / closes.iloc[0] - 1)


# ---------------------------------------------------------------------------
# Daily: Volume Breakout
# ---------------------------------------------------------------------------

class DailyVolumeBreakout(BaseStrategy):
    """
    Signal: volume_ratio > 2x AND 1-day return > 0.
    Rank by volume_ratio desc. Hold top_n stocks, rebalance daily.
    """
    name        = "daily_volume_breakout"
    description = "Volume ratio > 2× & positive day — top 10 by volume ratio"
    timeframe   = "daily"

    def __init__(self, top_n: int = 10, min_vol_ratio: float = 2.0, vol_lookback: int = 20):
        self.top_n          = top_n
        self.min_vol_ratio  = min_vol_ratio
        self.vol_lookback   = vol_lookback

    def select(self, universe_df, history, date):
        tickers = universe_df["ticker"].tolist()
        scores  = []
        for tk in tickers:
            df = history.get(tk)
            if df is None or df.empty:
                continue
            df = df[df["date"] <= date].tail(self.vol_lookback + 1)
            if len(df) < 2:
                continue
            today = df.iloc[-1]
            prev  = df.iloc[-2]
            if prev["close"] <= 0:
                continue
            ret_1d = float(today["close"] / prev["close"] - 1)
            avg_vol = df.iloc[:-1]["volume"].mean()
            if avg_vol <= 0:
                continue
            vol_ratio = float(today["volume"] / avg_vol)
            if vol_ratio >= self.min_vol_ratio and ret_1d > 0:
                scores.append((tk, vol_ratio))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [tk for tk, _ in scores[: self.top_n]]


# ---------------------------------------------------------------------------
# Weekly: 5-Day Momentum
# ---------------------------------------------------------------------------

class WeeklyMomentum5d(BaseStrategy):
    """
    Rank by 5-day return desc. Hold top_n stocks, rebalance weekly.
    """
    name        = "weekly_momentum_5d"
    description = "Top 10 by 5-day return — rebalanced weekly"
    timeframe   = "weekly"

    def __init__(self, top_n: int = 10):
        self.top_n = top_n

    def select(self, universe_df, history, date):
        tickers = universe_df["ticker"].tolist()
        scores  = []
        for tk in tickers:
            closes = _last_n_closes(history, tk, date, 6)   # 5 days = 6 closes
            m = _momentum(closes)
            if m > float("-inf"):
                scores.append((tk, m))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [tk for tk, _ in scores[: self.top_n]]


# ---------------------------------------------------------------------------
# Monthly: 20-Day Sector-Aware Momentum
# ---------------------------------------------------------------------------

class MonthlySectorMomentum(BaseStrategy):
    """
    Rank by 20-day return, max `max_per_sector` stocks per sector.
    Hold top_n stocks, rebalance monthly.
    """
    name        = "monthly_sector_momentum_20d"
    description = "Top 10 by 20-day return, max 3 per sector — rebalanced monthly"
    timeframe   = "monthly"

    def __init__(self, top_n: int = 10, max_per_sector: int = 3):
        self.top_n          = top_n
        self.max_per_sector = max_per_sector

    def select(self, universe_df, history, date):
        sector_map = (
            universe_df.set_index("ticker")["sector"].to_dict()
            if "sector" in universe_df.columns else {}
        )
        tickers = universe_df["ticker"].tolist()
        scores  = []
        for tk in tickers:
            closes = _last_n_closes(history, tk, date, 21)   # ~20 trading days
            m = _momentum(closes)
            if m > float("-inf"):
                scores.append((tk, m))
        scores.sort(key=lambda x: x[1], reverse=True)

        result       = []
        sector_count: dict[str, int] = {}
        for tk, _ in scores:
            sector = sector_map.get(tk, "Unknown")
            if sector_count.get(sector, 0) >= self.max_per_sector:
                continue
            result.append(tk)
            sector_count[sector] = sector_count.get(sector, 0) + 1
            if len(result) >= self.top_n:
                break
        return result


# ---------------------------------------------------------------------------
# Yearly: 252-Day Momentum
# ---------------------------------------------------------------------------

class YearlyMomentum252d(BaseStrategy):
    """
    Rank by 252-day return desc. Hold top_n stocks, rebalance yearly.
    """
    name        = "yearly_momentum_252d"
    description = "Top 10 by 252-day return — rebalanced yearly"
    timeframe   = "yearly"

    def __init__(self, top_n: int = 10):
        self.top_n = top_n

    def select(self, universe_df, history, date):
        tickers = universe_df["ticker"].tolist()
        scores  = []
        for tk in tickers:
            closes = _last_n_closes(history, tk, date, 253)  # 252 + 1
            m = _momentum(closes)
            if m > float("-inf"):
                scores.append((tk, m))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [tk for tk, _ in scores[: self.top_n]]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, BaseStrategy] = {
    "daily_volume_breakout":      DailyVolumeBreakout(),
    "weekly_momentum_5d":         WeeklyMomentum5d(),
    "monthly_sector_momentum_20d": MonthlySectorMomentum(),
    "yearly_momentum_252d":       YearlyMomentum252d(),
}

# Default strategy per timeframe
DEFAULTS: dict[str, str] = {
    "daily":   "daily_volume_breakout",
    "weekly":  "weekly_momentum_5d",
    "monthly": "monthly_sector_momentum_20d",
    "yearly":  "yearly_momentum_252d",
}


def get_strategy(name: str) -> BaseStrategy:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy {name!r}. Available: {list(STRATEGIES)}")
    return STRATEGIES[name]


def strategies_for_timeframe(timeframe: str) -> list[dict]:
    """Return list of {name, description} for a given timeframe."""
    return [
        {"name": s.name, "description": s.description}
        for s in STRATEGIES.values()
        if s.timeframe == timeframe
    ]
