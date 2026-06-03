"""
Backtest Engine
===============
Execution model
---------------
  Signal at T-day CLOSE -> execute at (T+1)-day OPEN
  Equal weight across held positions
  Commission: 0.1% per trade leg (buy + sell)
  Starting equity: 1 000 000 (normalised; metrics are ratio-based)

Rebalance frequency
-------------------
  daily   -- every trading day
  weekly  -- every Monday (or first available trading day of the week)
  monthly -- first trading day of each calendar month
  yearly  -- annual cohort mode (signal at year start, track full year return)

Cache
-----
  Results are stored at:
    data/backtest/results/{MARKET}/{TIMEFRAME}/{CACHE_KEY}.json
  Cache key: md5 hash of (market, timeframe, universe_source, signal, start_date, end_date)
  A result is considered fresh for CACHE_TTL_HOURS hours (default 24).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.backtest.data_loader import load_market_history_batch, load_universe
from src.backtest.metrics import compute_all
from src.backtest.strategies import BaseStrategy, DEFAULTS, get_strategy, STRATEGIES
from src.common.config import get_data_path
from src.common.logger import get_logger
from src.common.tracing import observe

log = get_logger("backtest.engine")

COMMISSION          = 0.001   # 0.1% per leg
INITIAL_EQUITY      = 1_000_000.0
CACHE_TTL_HOURS     = 24       # TTL for recent backtests (end_date near today)
CACHE_TTL_HIST_DAYS = 30       # TTL for historical backtests (end_date clearly in the past)

# How many years of history to use per timeframe
_HISTORY_YEARS = {
    "daily":   2,
    "weekly":  3,
    "monthly": 5,
    "yearly":  10,
}

# Minimum history needed (in trading days) to generate first signal
_MIN_BARS = {
    "daily":    21,
    "weekly":   6,
    "monthly":  21,
    "yearly":   253,
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

@observe(name="_make_cache_key", type="tool")
def _make_cache_key(market: str, timeframe: str, universe_source: dict,
                    signal: dict, start_date: str, end_date: str) -> str:
    payload = {
        "market": market,
        "timeframe": timeframe,
        "universe_source": universe_source,
        "signal": signal,
        "start_date": start_date,
        "end_date": end_date,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


@observe(name="_result_path", type="tool")
def _result_path(market: str, timeframe: str, cache_key: str) -> str:
    return get_data_path("backtest", "results", market, timeframe, f"{cache_key}.json")


@observe(name="_is_fresh", type="tool")
def _is_fresh(path: str, end_date: str = "") -> bool:
    if not os.path.exists(path):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    # Historical backtests (end_date clearly in the past) use a much longer TTL
    # so results don't shift between days due to adjusted-price updates.
    if end_date:
        try:
            end_ts = datetime.strptime(end_date, "%Y-%m-%d")
            if end_ts < datetime.now() - timedelta(days=7):
                return age < timedelta(days=CACHE_TTL_HIST_DAYS)
        except ValueError:
            pass
    return age < timedelta(hours=CACHE_TTL_HOURS)


@observe(name="load_cached_result", type="tool")
def load_cached_result(market: str, timeframe: str, cache_key: str,
                       end_date: str = "") -> Optional[dict]:
    path = _result_path(market, timeframe, cache_key)
    if _is_fresh(path, end_date):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Corrupted cache at {path}, re-running: {e}")
    return None


@observe(name="save_result", type="tool")
def save_result(market: str, timeframe: str, cache_key: str, result: dict):
    path = _result_path(market, timeframe, cache_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, default=str)


# ---------------------------------------------------------------------------
# Universe loading
# ---------------------------------------------------------------------------

@observe(name="_load_universe_df", type="tool")
def _load_universe_df(universe_source: dict, market: str) -> pd.DataFrame:
    """
    Load universe dataframe from universe_source spec.

    universe_source types:
      {"type": "market"}                        — all tickers in market universe.parquet
      {"type": "custom", "tickers": [...]}      — user-supplied ticker list
    """
    utype = universe_source.get("type", "market")

    if utype == "custom":
        tickers = universe_source.get("tickers", [])
        if not tickers:
            raise ValueError("Custom universe_source must include 'tickers' list")
        return pd.DataFrame({"ticker": tickers, "name": "", "sector": "", "subsector": ""})

    else:
        # Default: load market universe.parquet
        return load_universe(market)


# ---------------------------------------------------------------------------
# Signal dispatch
# ---------------------------------------------------------------------------

@observe(name="_build_signal_fn", type="tool")
def _build_signal_fn(signal: dict, timeframe: str) -> Callable:
    """Return a callable select(universe_df, history, date) -> list[str]."""
    stype = signal.get("type", "builtin")

    if stype == "builtin":
        names = signal.get("name") or DEFAULTS.get(timeframe)
        if isinstance(names, list):
            # Combination: union of all selected strategies (deduplicated, order preserved)
            strategies = [get_strategy(n) for n in names]
            def _combined(universe_df, history, date):
                seen = set()
                result = []
                for s in strategies:
                    for tk in s.select(universe_df, history, date):
                        if tk not in seen:
                            seen.add(tk)
                            result.append(tk)
                return result
            return _combined
        strategy = get_strategy(names)
        return strategy.select

    elif stype == "code":
        code = signal.get("code", "")
        if not code:
            raise ValueError("signal type 'code' requires 'code' field")
        from src.backtest.signal_builder import safe_compile
        fn, err = safe_compile(code)
        if fn is None:
            raise ValueError(f"Signal code compilation failed: {err}")
        return fn

    elif stype == "multi_code":
        # Union or intersection of multiple code signals (for multiple NL signals)
        codes = signal.get("codes", [])
        mode  = signal.get("mode", "union")   # "union" | "intersect"
        if not codes:
            raise ValueError("signal type 'multi_code' requires 'codes' list")
        from src.backtest.signal_builder import safe_compile
        fns = []
        for code in codes:
            fn, err = safe_compile(code)
            if fn is None:
                raise ValueError(f"Signal code compilation failed: {err}")
            fns.append(fn)
        if mode == "intersect":
            def _multi_intersect(universe_df, history, date, _fns=fns):
                sets = [set(fn(universe_df, history, date)) for fn in _fns]
                common = sets[0].intersection(*sets[1:]) if sets else set()
                # Preserve order from first signal
                first = [tk for tk in _fns[0](universe_df, history, date) if tk in common]
                return first
            return _multi_intersect
        else:
            def _multi_union(universe_df, history, date, _fns=fns):
                seen: set = set()
                result = []
                for fn in _fns:
                    for tk in fn(universe_df, history, date):
                        if tk not in seen:
                            seen.add(tk)
                            result.append(tk)
                return result
            return _multi_union

    else:
        raise ValueError(f"Unknown signal type {stype!r}")


# ---------------------------------------------------------------------------
# Rebalance date generation
# ---------------------------------------------------------------------------

@observe(name="_rebalance_dates", type="tool")
def _rebalance_dates(all_dates: pd.DatetimeIndex, timeframe: str) -> list[pd.Timestamp]:
    """Return the subset of `all_dates` that are rebalance dates."""
    dates = sorted(all_dates)
    if timeframe == "daily":
        return dates

    result = []
    prev   = None
    for d in dates:
        if prev is None:
            result.append(d)
            prev = d
            continue
        if timeframe == "weekly" and d.isocalendar()[1] != prev.isocalendar()[1]:
            result.append(d)
        elif timeframe == "monthly" and (d.month != prev.month or d.year != prev.year):
            result.append(d)
        elif timeframe == "yearly" and d.year != prev.year:
            result.append(d)
        prev = d
    return result


# ---------------------------------------------------------------------------
# Yearly cohort mode
# ---------------------------------------------------------------------------

def _compute_yearly_cohort(
    tickers: list[str],
    history: dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
    name_map: dict[str, str],
) -> tuple[float, list[dict]]:
    """
    Compute equal-weight portfolio return and trade records for one yearly cohort.
    Returns (total_return_fraction, trades_list).
    """
    if not tickers:
        return 0.0, []

    start_ts = pd.Timestamp(start_date)
    end_ts   = pd.Timestamp(end_date)
    returns  = []
    trades   = []

    for tk in tickers:
        df = history.get(tk)
        if df is None or df.empty:
            continue
        dates_arr = df["date"].values
        s_idx = int(dates_arr.searchsorted(start_ts.to_datetime64(), side="left"))
        e_idx = int(dates_arr.searchsorted(end_ts.to_datetime64(), side="right"))
        if s_idx >= len(dates_arr) or e_idx == 0:
            continue
        start_row = df.iloc[s_idx]
        end_row   = df.iloc[e_idx - 1]
        entry = float(start_row["open"]) if "open" in df.columns else float(start_row["close"])
        exit_ = float(end_row["close"])
        if entry > 0:
            pnl_pct = exit_ / entry - 1 - COMMISSION * 2
            returns.append(pnl_pct)
            trades.append({
                "ticker":      tk,
                "name":        name_map.get(tk, ""),
                "entry_date":  str(start_row["date"])[:10],
                "exit_date":   str(end_row["date"])[:10],
                "entry_price": round(entry, 4),
                "exit_price":  round(exit_, 4),
                "pnl_pct":     round(pnl_pct, 6),
            })

    total_return = float(np.mean(returns)) if returns else 0.0
    return total_return, trades


def _run_yearly_mode(
    market: str,
    universe_source: dict,
    signal_fn: Callable,
    start_date: str,
    end_date: str,
    history: dict[str, pd.DataFrame],
    progress_callback: Optional[Callable],
) -> dict:
    """Run yearly cohort backtest."""
    start_year = int(start_date[:4])
    end_year   = int(end_date[:4])

    cohorts    = []
    equity     = INITIAL_EQUITY
    equity_curve = []
    trades     = []

    # Load universe and name map once — reused for all years
    _uni = _load_universe_df(universe_source, market)
    _name_map: dict[str, str] = dict(zip(_uni["ticker"], _uni["name"])) if "name" in _uni.columns else {}

    all_years = list(range(start_year, end_year + 1))
    total = len(all_years)

    # Baseline entry so equity[0] == INITIAL_EQUITY, making total_return include year 1
    equity_curve.append((f"{start_year}-01-01", round(equity, 2)))

    for i, year in enumerate(all_years):
        as_of    = f"{year}-01-01"
        year_end = f"{year}-12-31"
        as_of_ts = pd.Timestamp(as_of)

        # Signal: select stocks at start of year
        # Pass full history — strategies filter by date internally
        try:
            selected = signal_fn(_uni, history, as_of_ts)
        except Exception as e:
            log.warning(f"[engine/yearly] Signal failed for {year}: {e}")
            selected = []

        # Compute return and record trades in a single pass
        year_return, year_trades = _compute_yearly_cohort(selected, history, as_of, year_end, _name_map)
        equity = equity * (1 + year_return)
        equity_curve.append((year_end, round(equity, 2)))
        trades.extend(year_trades)

        cohorts.append({
            "year":    year,
            "return":  round(year_return, 6),
            "tickers": selected,
        })
        log.info(f"[engine/yearly] Year {year}: {len(selected)} stocks, return={year_return:.2%}")

        if progress_callback:
            progress_callback(i + 1, total)

    return {
        "cohorts":      cohorts,
        "equity_curve": equity_curve,
        "trades":       trades,
    }


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

@observe(name="run_backtest", type="tool")
def run_backtest(
    market: str,
    timeframe: str,
    universe_source: Optional[dict] = None,
    signal: Optional[dict] = None,
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    force: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    # Legacy compatibility
    strategy_name: Optional[str] = None,
) -> dict:
    """
    Run a backtest or return cached result.

    Parameters
    ----------
    market           : e.g. "US", "CN"
    timeframe        : "daily" | "weekly" | "monthly" | "yearly"
    universe_source  : {"type": "index", "key": "CSI300"} or {"type": "custom", "tickers": [...]}
    signal           : {"type": "builtin", "name": "volume_breakout"} or {"type": "code", "code": "..."}
    start_date       : "2022-01-01"
    end_date         : "2024-12-31"
    force            : ignore cache and recompute
    progress_callback: (done, total) called periodically
    strategy_name    : legacy param; maps to signal={"type":"builtin","name":strategy_name}

    Returns
    -------
    {
        "market":    str,
        "timeframe": str,
        "strategy":  str,
        "metrics":   {...},
        "equity_curve": [[date_str, value], ...],
        "trades":    [{...}, ...],
        "cohorts":   [{...}, ...],  # yearly only
        "generated_at": str,
    }
    """
    # Handle legacy strategy_name param
    if strategy_name is not None and signal is None:
        signal = {"type": "builtin", "name": strategy_name}

    # Defaults
    if universe_source is None:
        universe_source = {"type": "market"}
    if signal is None:
        signal = {"type": "builtin", "name": DEFAULTS.get(timeframe, "daily_volume_breakout")}

    # Cache key
    cache_key = _make_cache_key(market, timeframe, universe_source, signal, start_date, end_date)

    if not force:
        cached = load_cached_result(market, timeframe, cache_key, end_date)
        if cached is not None:
            log.info(f"[{market}/{timeframe}] Returning cached result (key={cache_key})")
            return cached

    t0 = time.time()
    signal_label = signal.get("name") or signal.get("type") or "custom"
    log.info(f"[{market}/{timeframe}] Running {signal_label} (cache_key={cache_key}) ...")

    # ── 1. Load universe ─────────────────────────────────────────────────────
    universe_df = _load_universe_df(universe_source, market)
    if universe_df.empty:
        raise RuntimeError(f"No universe found for {universe_source}")

    # ── 2. Load history ───────────────────────────────────────────────────────
    history_years = _HISTORY_YEARS.get(timeframe, 3)
    # Use start_date with extra lookback for signals
    lookback_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=history_years * 365)).strftime("%Y-%m-%d")

    history = load_market_history_batch(
        market, lookback_start, end_date, progress_callback=progress_callback
    )
    if not history:
        raise RuntimeError(f"No history loaded for market {market!r}")

    # ── 3. Build signal function ──────────────────────────────────────────────
    signal_fn = _build_signal_fn(signal, timeframe)

    # ── 4. Yearly cohort mode ─────────────────────────────────────────────────
    if timeframe == "yearly":
        yearly_result = _run_yearly_mode(
            market=market,
            universe_source=universe_source,
            signal_fn=signal_fn,
            start_date=start_date,
            end_date=end_date,
            history=history,
            progress_callback=progress_callback,
        )
        equity_curve = yearly_result["equity_curve"]
        trades       = yearly_result["trades"]
        cohorts      = yearly_result["cohorts"]
    else:
        # ── 5. Build common trading calendar ──────────────────────────────────
        all_dates_set: set = set()
        for df in history.values():
            all_dates_set.update(df["date"].tolist())
        all_dates = pd.DatetimeIndex(sorted(all_dates_set))

        # Filter to [start_date, end_date]
        start_ts = pd.Timestamp(start_date)
        end_ts   = pd.Timestamp(end_date)
        all_dates = all_dates[(all_dates >= start_ts) & (all_dates <= end_ts)]

        min_bars     = _MIN_BARS.get(timeframe, 21)
        signal_dates = all_dates[min_bars:]

        rebal_dates = _rebalance_dates(signal_dates, timeframe)

        # Build price matrices (date-indexed, ticker columns) for O(1) .at[] lookups
        close_px = pd.DataFrame(
            {tk: df.set_index("date")["close"] for tk, df in history.items()}
        ).sort_index().ffill()
        open_px = pd.DataFrame(
            {tk: df.set_index("date")["open"] for tk, df in history.items() if "open" in df.columns}
        ).sort_index().ffill()

        # Name lookup for trades display
        name_map: dict[str, str] = {}
        if "name" in universe_df.columns:
            name_map = dict(zip(universe_df["ticker"], universe_df["name"]))

        # ── 6. Simulation loop ──────────────────────────────────────────────
        equity        = INITIAL_EQUITY
        held_tickers: list[str] = []
        # held_cost:   original entry cost basis — used ONLY for trade pnl calculation
        # held_prev_px: previous close — used for daily MTM (updated each iteration)
        held_cost:    dict[str, float] = {}
        held_prev_px: dict[str, float] = {}
        held_dates:   dict[str, pd.Timestamp] = {}

        equity_curve: list = []
        trades: list       = []
        cohorts            = []

        total_steps = len(rebal_dates)
        done_steps  = 0

        for signal_date in rebal_dates:
            _next_idx = all_dates.searchsorted(signal_date, side="right")
            exec_date = all_dates[_next_idx] if _next_idx < len(all_dates) else None

            # ── Mark equity FIRST using current (old) portfolio ──────────────
            # Vectorized MTM: use price matrix for all held tickers at once.
            if held_tickers:
                row_date = signal_date if signal_date in close_px.index else None
                if row_date is not None:
                    curr_prices = close_px.loc[row_date, [tk for tk in held_tickers if tk in close_px.columns]]
                else:
                    curr_prices = pd.Series(dtype=float)
                day_returns = []
                for tk in held_tickers:
                    curr_px = float(curr_prices.get(tk, np.nan)) if tk in curr_prices.index else np.nan
                    if np.isnan(curr_px) or curr_px <= 0:
                        curr_px = None
                    prev_px = held_prev_px.get(tk)
                    if curr_px and curr_px > 0 and prev_px and prev_px > 0:
                        day_returns.append(curr_px / prev_px)
                        held_prev_px[tk] = curr_px
                    elif curr_px and curr_px > 0:
                        held_prev_px[tk] = curr_px
                if day_returns:
                    equity = equity * sum(day_returns) / len(day_returns)
            equity_curve.append((str(signal_date)[:10], round(equity, 2)))

            if exec_date is None:
                done_steps += 1
                continue

            # ── Generate signal ──────────────────────────────────────────────
            # Pass history directly — strategies filter by date internally.
            # Boolean indexing (df[mask]) always returns a copy in pandas,
            # so there is no mutation risk on the master history dict.
            new_tickers = signal_fn(universe_df, history, signal_date)

            # ── Close positions not in new portfolio (at exec_date open) ─────
            for tk in list(held_tickers):
                if tk not in new_tickers:
                    exit_px = float(open_px.at[exec_date, tk]) if (exec_date in open_px.index and tk in open_px.columns) else None
                    if exit_px is None or np.isnan(exit_px) or exit_px <= 0:
                        exit_px = float(close_px.at[signal_date, tk]) if (signal_date in close_px.index and tk in close_px.columns) else None
                    if exit_px is not None and np.isnan(exit_px):
                        exit_px = None
                    if exit_px is None or exit_px <= 0:
                        held_tickers.remove(tk)
                        continue
                    entry_px = held_cost.get(tk)
                    if not entry_px or entry_px <= 0:
                        log.warning(f"No entry cost for {tk}, skipping trade record")
                        held_tickers.remove(tk)
                        held_cost.pop(tk, None)
                        held_prev_px.pop(tk, None)
                        held_dates.pop(tk, None)
                        continue
                    pnl_pct  = exit_px / entry_px - 1 - COMMISSION
                    trades.append({
                        "ticker":      tk,
                        "name":        name_map.get(tk, ""),
                        "entry_date":  str(held_dates.get(tk, signal_date))[:10],
                        "exit_date":   str(exec_date)[:10],
                        "entry_price": round(entry_px, 4),
                        "exit_price":  round(exit_px, 4),
                        "pnl_pct":     round(pnl_pct, 6),
                    })
                    held_tickers.remove(tk)
                    held_cost.pop(tk, None)
                    held_prev_px.pop(tk, None)
                    held_dates.pop(tk, None)

            # ── Open new positions (at exec_date open) ───────────────────────
            for tk in new_tickers:
                if tk not in held_tickers:
                    entry_px = float(open_px.at[exec_date, tk]) if (exec_date in open_px.index and tk in open_px.columns) else None
                    if entry_px is not None and np.isnan(entry_px):
                        entry_px = None
                    if entry_px is None or entry_px <= 0:
                        continue
                    held_tickers.append(tk)
                    held_cost[tk]    = entry_px * (1 + COMMISSION)   # for trade pnl
                    held_prev_px[tk] = entry_px                       # MTM reference: entry open
                    held_dates[tk]   = exec_date

            done_steps += 1
            if progress_callback and done_steps % 20 == 0:
                progress_callback(done_steps, total_steps)

        # Close remaining positions
        if held_tickers and all_dates.size > 0:
            last_date = all_dates[-1]
            for tk in held_tickers:
                _v = float(close_px.at[last_date, tk]) if (last_date in close_px.index and tk in close_px.columns) else np.nan
                exit_px = _v if not np.isnan(_v) else None
                entry_px = held_cost.get(tk)
                if not exit_px or exit_px <= 0:
                    log.debug(f"No exit price for {tk} at {last_date}, skipping close")
                    continue
                if not entry_px or entry_px <= 0:
                    log.warning(f"No entry cost recorded for {tk}, skipping close")
                    continue
                if exit_px and exit_px > 0 and entry_px > 0:
                    pnl_pct = exit_px / entry_px - 1 - COMMISSION
                    trades.append({
                        "ticker":      tk,
                        "name":        name_map.get(tk, ""),
                        "entry_date":  str(held_dates.get(tk, last_date))[:10],
                        "exit_date":   str(last_date)[:10],
                        "entry_price": round(entry_px, 4),
                        "exit_price":  round(exit_px, 4),
                        "pnl_pct":     round(pnl_pct, 6),
                    })

    # ── 7. Metrics ────────────────────────────────────────────────────────────
    if equity_curve:
        eq_series = pd.Series(
            [v for _, v in equity_curve],
            index=pd.to_datetime([d for d, _ in equity_curve]),
        )
        trades_df = pd.DataFrame(trades) if trades else None
        metrics   = compute_all(eq_series, trades_df)
    else:
        metrics = {k: 0.0 for k in ["total_return", "cagr", "sharpe", "max_drawdown", "win_rate", "avg_hold_days"]}
        metrics["num_trades"] = 0

    duration = round(time.time() - t0, 1)
    log.info(f"[{market}/{timeframe}] Done in {duration}s — {len(trades)} trades, CAGR={metrics.get('cagr', 0):.2%}")

    result = {
        "market":        market,
        "timeframe":     timeframe,
        "strategy":      signal_label,
        "universe":      universe_source,
        "metrics":       {k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
        "equity_curve":  equity_curve,
        "trades":        trades,
        "num_tickers":   len(universe_df),
        "history_years": history_years,
        "duration_s":    duration,
        "generated_at":  datetime.now().isoformat(),
    }
    if cohorts:
        result["cohorts"] = cohorts

    save_result(market, timeframe, cache_key, result)
    return result
