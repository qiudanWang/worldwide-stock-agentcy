"""
Backtest Metrics
================
Pure functions — no I/O, no side effects.
All functions operate on a portfolio equity series (pd.Series indexed by date).
"""

import numpy as np
import pandas as pd
from typing import Optional


def total_return(equity: pd.Series) -> float:
    """Total return from first to last equity value."""
    if len(equity) < 2:
        return 0.0
    start = equity.iloc[0]
    if start <= 0 or np.isnan(start):
        return 0.0
    return float(equity.iloc[-1] / start - 1)


def cagr(equity: pd.Series) -> float:
    """Compound Annual Growth Rate."""
    if len(equity) < 2:
        return 0.0
    start_val = equity.iloc[0]
    if start_val <= 0 or np.isnan(start_val):
        return 0.0
    start = equity.index[0]
    end   = equity.index[-1]
    years = (end - start).days / 365.25
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / start_val) ** (1.0 / years) - 1)


def sharpe_ratio(equity: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Annualised Sharpe ratio using daily returns.
    Assumes 252 trading days per year.
    """
    if len(equity) < 2:
        return 0.0
    daily_ret = equity.pct_change().dropna()
    excess    = daily_ret - risk_free_rate / 252
    std       = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(252))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number, e.g. -0.25 = -25%)."""
    if len(equity) < 2:
        return 0.0
    if equity.iloc[0] <= 0 or np.isnan(equity.iloc[0]):
        return 0.0
    rolling_max = equity.cummax()
    # Avoid division by zero if equity ever reaches 0
    safe_max = rolling_max.replace(0, np.nan)
    drawdown = equity / safe_max - 1
    return float(drawdown.min())


def win_rate(trades: pd.DataFrame) -> float:
    """
    Fraction of trades that were profitable.
    trades must have a 'pnl_pct' column.
    """
    if trades is None or len(trades) == 0:
        return 0.0
    winners = (trades["pnl_pct"] > 0).sum()
    return float(winners / len(trades))


def avg_hold_days(trades: pd.DataFrame) -> float:
    """
    Average holding period in calendar days.
    trades must have 'entry_date' and 'exit_date' columns.
    """
    if trades is None or len(trades) == 0:
        return 0.0
    hold = (
        pd.to_datetime(trades["exit_date"]) - pd.to_datetime(trades["entry_date"])
    ).dt.days
    return float(hold.mean())


def compute_all(
    equity: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Compute all metrics in one call.

    Returns
    -------
    {
        "total_return":   float,   # e.g.  0.42 = +42%
        "cagr":           float,   # e.g.  0.18 = +18% p.a.
        "sharpe":         float,
        "max_drawdown":   float,   # e.g. -0.25 = -25%
        "win_rate":       float,   # fraction 0–1
        "avg_hold_days":  float,
        "num_trades":     int,
    }
    """
    return {
        "total_return":  total_return(equity),
        "cagr":          cagr(equity),
        "sharpe":        sharpe_ratio(equity, risk_free_rate),
        "max_drawdown":  max_drawdown(equity),
        "win_rate":      win_rate(trades) if trades is not None else 0.0,
        "avg_hold_days": avg_hold_days(trades) if trades is not None else 0.0,
        "num_trades":    len(trades) if trades is not None else 0,
    }
