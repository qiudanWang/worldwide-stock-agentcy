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

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional
from src.common.tracing import observe


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    name: str
    description: str
    timeframe: str   # "daily" | "weekly" | "monthly" | "yearly"

    @abstractmethod
    @observe(name="BaseStrategy.select", type="tool")
    def select(
        self,
        universe_df: pd.DataFrame,
        history: dict[str, pd.DataFrame],
        date: pd.Timestamp,
    ) -> list[str]:
        """Return list of tickers to hold. Max `top_n` items."""
        ...

    @observe(name="BaseStrategy.__repr__", type="tool")
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _until(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """O(log n) slice of df up to `date` — assumes df is sorted by date asc."""
    idx = int(df["date"].values.searchsorted(date.to_datetime64(), side="right"))
    return df.iloc[:idx]


@observe(name="_last_n_closes", type="tool")
def _last_n_closes(history: dict[str, pd.DataFrame], ticker: str, date: pd.Timestamp, n: int) -> Optional[pd.Series]:
    """Return the last n closing prices up to and including `date`."""
    df = history.get(ticker)
    if df is None or df.empty:
        return None
    df = _until(df, date).tail(n)
    if len(df) < n:
        return None
    return df["close"].reset_index(drop=True)


@observe(name="_momentum", type="tool")
def _momentum(closes: pd.Series) -> float:
    """Simple price momentum: last / first - 1."""
    if closes is None or len(closes) < 2:
        return float("-inf")
    first = closes.iloc[0]
    if not first or first <= 0:
        return float("-inf")
    return float(closes.iloc[-1] / first - 1)


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

    @observe(name="DailyVolumeBreakout.__init__", type="tool")
    def __init__(self, top_n: int = 10, min_vol_ratio: float = 2.0, vol_lookback: int = 20):
        self.top_n          = top_n
        self.min_vol_ratio  = min_vol_ratio
        self.vol_lookback   = vol_lookback

    @observe(name="DailyVolumeBreakout.select", type="tool")
    def select(self, universe_df, history, date):
        tickers = universe_df["ticker"].tolist()
        n = self.vol_lookback + 1
        tk_list, vol_ratios, ret_1ds = [], [], []

        for tk in tickers:
            df = history.get(tk)
            if df is None or df.empty:
                continue
            tail = _until(df, date).tail(n)
            if len(tail) < 2:
                continue
            close = tail["close"].values
            vol   = tail["volume"].values
            if close[-2] <= 0:
                continue
            avg_vol = vol[:-1].mean()
            if avg_vol <= 0:
                continue
            tk_list.append(tk)
            vol_ratios.append(vol[-1] / avg_vol)
            ret_1ds.append(close[-1] / close[-2] - 1)

        if not tk_list:
            return []

        vol_arr = np.array(vol_ratios)
        ret_arr = np.array(ret_1ds)
        mask    = (vol_arr >= self.min_vol_ratio) & (ret_arr > 0)
        idx     = np.where(mask)[0]
        if idx.size == 0:
            return []
        top = idx[np.argsort(vol_arr[idx])[::-1][: self.top_n]]
        return [tk_list[i] for i in top]


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

    @observe(name="WeeklyMomentum5d.__init__", type="tool")
    def __init__(self, top_n: int = 10):
        self.top_n = top_n

    @observe(name="WeeklyMomentum5d.select", type="tool")
    def select(self, universe_df, history, date):
        tickers = universe_df["ticker"].tolist()
        tk_list, moms = [], []
        for tk in tickers:
            closes = _last_n_closes(history, tk, date, 6)
            m = _momentum(closes)
            if m > float("-inf"):
                tk_list.append(tk)
                moms.append(m)
        if not tk_list:
            return []
        arr = np.array(moms)
        top = np.argsort(arr)[::-1][: self.top_n]
        return [tk_list[i] for i in top]


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

    @observe(name="MonthlySectorMomentum.__init__", type="tool")
    def __init__(self, top_n: int = 10, max_per_sector: int = 3):
        self.top_n          = top_n
        self.max_per_sector = max_per_sector

    @observe(name="MonthlySectorMomentum.select", type="tool")
    def select(self, universe_df, history, date):
        sector_map = (
            universe_df.set_index("ticker")["sector"].to_dict()
            if "sector" in universe_df.columns else {}
        )
        tickers = universe_df["ticker"].tolist()
        tk_list, moms = [], []
        for tk in tickers:
            closes = _last_n_closes(history, tk, date, 21)
            m = _momentum(closes)
            if m > float("-inf"):
                tk_list.append(tk)
                moms.append(m)

        if not tk_list:
            return []
        arr     = np.array(moms)
        ordered = [tk_list[i] for i in np.argsort(arr)[::-1]]

        result       = []
        sector_count: dict[str, int] = {}
        for tk in ordered:
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

    @observe(name="YearlyMomentum252d.__init__", type="tool")
    def __init__(self, top_n: int = 10):
        self.top_n = top_n

    @observe(name="YearlyMomentum252d.select", type="tool")
    def select(self, universe_df, history, date):
        tickers = universe_df["ticker"].tolist()
        tk_list, moms = [], []
        for tk in tickers:
            closes = _last_n_closes(history, tk, date, 253)
            m = _momentum(closes)
            if m > float("-inf"):
                tk_list.append(tk)
                moms.append(m)
        if not tk_list:
            return []
        arr = np.array(moms)
        top = np.argsort(arr)[::-1][: self.top_n]
        return [tk_list[i] for i in top]


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


@observe(name="get_strategy", type="tool")
def get_strategy(name: str) -> BaseStrategy:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy {name!r}. Available: {list(STRATEGIES)}")
    return STRATEGIES[name]


@observe(name="strategies_for_timeframe", type="tool")
def strategies_for_timeframe(timeframe: str) -> list[dict]:
    """Return list of {name, description} for a given timeframe."""
    return [
        {"name": s.name, "description": s.description}
        for s in STRATEGIES.values()
        if s.timeframe == timeframe
    ]
