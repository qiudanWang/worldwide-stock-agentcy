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

def _last_n_closes(history: dict[str, pd.DataFrame], ticker: str, date_np, n: int) -> Optional[np.ndarray]:
    """Return the last n closing prices up to `date` as a numpy array."""
    df = history.get(ticker)
    if df is None or df.empty:
        return None
    idx = int(df["date"].values.searchsorted(date_np, side="right"))
    if idx < n:
        return None
    return df["close"].values[idx - n : idx]


def _momentum(closes: np.ndarray) -> float:
    """Simple price momentum: last / first - 1."""
    if closes is None or len(closes) < 2:
        return float("-inf")
    first = closes[0]
    if not first or first <= 0:
        return float("-inf")
    return float(closes[-1] / first - 1)


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
        n       = self.vol_lookback + 1
        date_np = date.to_datetime64()
        tk_list, vol_ratios, ret_1ds = [], [], []

        for tk in tickers:
            df = history.get(tk)
            if df is None or df.empty:
                continue
            idx = int(df["date"].values.searchsorted(date_np, side="right"))
            if idx < 2:
                continue
            start = max(0, idx - n)
            close = df["close"].values[start:idx]
            vol   = df["volume"].values[start:idx]
            if len(close) < 2 or close[-2] <= 0:
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

    def __init__(self, top_n: int = 10):
        self.top_n = top_n

    def select(self, universe_df, history, date):
        tickers = universe_df["ticker"].tolist()
        date_np = date.to_datetime64()
        tk_list, moms = [], []
        for tk in tickers:
            closes = _last_n_closes(history, tk, date_np, 6)
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

    def __init__(self, top_n: int = 10, max_per_sector: int = 3):
        self.top_n          = top_n
        self.max_per_sector = max_per_sector

    def select(self, universe_df, history, date):
        sector_map = (
            universe_df.set_index("ticker")["sector"].to_dict()
            if "sector" in universe_df.columns else {}
        )
        tickers = universe_df["ticker"].tolist()
        date_np = date.to_datetime64()
        tk_list, moms = [], []
        for tk in tickers:
            closes = _last_n_closes(history, tk, date_np, 21)
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
# Daily: ATR Breakout from Low Base
# ---------------------------------------------------------------------------

class DailyATRBreakoutLowBase(BaseStrategy):
    """
    Signal: today's price change > atr_mult × ATR(atr_period) AND positive return
            AND today's close < 30th percentile of past price_pct_lookback closes.
    Rank by breakout magnitude (price_change / ATR) desc. Rebalance daily.
    """
    name        = "daily_atr_breakout_low_base"
    description = (
        "ATR breakout (>1.5× ATR20) from low-base (below 30th pct of 60-day range) "
        "with positive return — top 10 by breakout magnitude"
    )
    timeframe   = "daily"

    def __init__(
        self,
        top_n: int = 10,
        atr_mult: float = 1.5,
        atr_period: int = 20,
        price_pct_lookback: int = 60,
        price_pct_threshold: float = 30.0,
    ):
        self.top_n               = top_n
        self.atr_mult            = atr_mult
        self.atr_period          = atr_period
        self.price_pct_lookback  = price_pct_lookback
        self.price_pct_threshold = price_pct_threshold

    @observe(name="DailyATRBreakoutLowBase.select", type="tool")
    def select(self, universe_df, history, date):
        tickers = universe_df["ticker"].tolist()
        date_np = date.to_datetime64()
        needed  = max(self.atr_period + 2, self.price_pct_lookback + 1)

        candidates = []
        for tk in tickers:
            df = history.get(tk)
            if df is None or df.empty:
                continue
            idx = int(df["date"].values.searchsorted(date_np, side="right"))
            if idx < needed:
                continue

            close  = df["close"].values[:idx]
            high   = df["high"].values[:idx]
            low    = df["low"].values[:idx]

            # ATR(20): true range over last atr_period bars
            tr = np.maximum(
                high[-self.atr_period:] - low[-self.atr_period:],
                np.maximum(
                    np.abs(high[-self.atr_period:] - close[-(self.atr_period + 1):-1]),
                    np.abs(low[-self.atr_period:]  - close[-(self.atr_period + 1):-1]),
                ),
            )
            atr = tr.mean()
            if atr <= 0:
                continue

            today_close = close[-1]
            prev_close  = close[-2]
            price_change = today_close - prev_close

            # Condition 1: positive return
            if price_change <= 0:
                continue

            # Condition 2: breakout > atr_mult × ATR
            if price_change <= self.atr_mult * atr:
                continue

            # Condition 3: today's close below price_pct_threshold percentile
            hist_closes = close[-self.price_pct_lookback:]
            pct_val = np.percentile(hist_closes, self.price_pct_threshold)
            if today_close >= pct_val:
                continue

            candidates.append((tk, price_change / atr))

        if not candidates:
            return []

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [tk for tk, _ in candidates[: self.top_n]]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, BaseStrategy] = {
    "daily_volume_breakout":        DailyVolumeBreakout(),
    "daily_atr_breakout_low_base":  DailyATRBreakoutLowBase(),
    "weekly_momentum_5d":           WeeklyMomentum5d(),
    "monthly_sector_momentum_20d":  MonthlySectorMomentum(),
}

# Default strategy per timeframe
DEFAULTS: dict[str, str] = {
    "daily":   "daily_volume_breakout",
    "weekly":  "weekly_momentum_5d",
    "monthly": "monthly_sector_momentum_20d",
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
