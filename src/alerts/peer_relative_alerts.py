"""Global Peer-Relative Alerts.

Compares ALL peers across ALL markets for divergences:
one peer moved significantly while another was flat.
"""
import pandas as pd
from src.common.config import load_yaml, get_data_path
from src.common.logger import get_logger

log = get_logger("alerts.peer")

# Market flag/emoji mapping for display
MARKET_FLAGS = {
    "CN": "CN", "US": "US", "HK": "HK", "JP": "JP", "KR": "KR",
    "TW": "TW", "IN": "IN", "UK": "UK", "DE": "DE", "FR": "FR",
    "AU": "AU", "BR": "BR", "SA": "SA",
}


def check_global_peer_alerts(all_market_signals, peer_mapping_df=None,
                             mover_threshold=0.03, flat_threshold=0.01):
    """Check for divergences across ALL global peer pairs.

    Alert trigger: one peer moved >= mover_threshold (default 3%) while
    another peer was flat (< flat_threshold, default 1%).

    Args:
        all_market_signals: DataFrame with columns: ticker, market, return_1d
        peer_mapping_df: global peer mapping (ticker, market, peer_ticker, peer_market, ...)
        mover_threshold: minimum absolute return to count as a "big move"
        flat_threshold: maximum absolute return to count as "flat"

    Returns:
        DataFrame of alerts with: mover_ticker, mover_market, mover_return,
        flat_ticker, flat_market, flat_return, peer_score, signal
    """
    if all_market_signals is None or all_market_signals.empty:
        return pd.DataFrame()

    if peer_mapping_df is None:
        try:
            path = get_data_path("global", "peer_mapping.parquet")
            peer_mapping_df = pd.read_parquet(path)
        except (FileNotFoundError, Exception):
            try:
                path = get_data_path("processed", "peer_mapping.parquet")
                peer_mapping_df = pd.read_parquet(path)
            except (FileNotFoundError, Exception):
                log.info("No peer mapping found, skipping global peer alerts")
                return pd.DataFrame()

    if peer_mapping_df.empty:
        return pd.DataFrame()

    # Build return lookup: (ticker) -> return_1d
    returns = dict(zip(all_market_signals["ticker"].astype(str),
                       all_market_signals["return_1d"]))

    # Use rank-1 peers only to avoid duplicate noise
    top_peers = peer_mapping_df[peer_mapping_df["rank"] == 1]

    # Determine columns based on format (global vs legacy)
    is_legacy = "cn_ticker" in top_peers.columns
    alerts = []

    if is_legacy:
        alerts = _check_legacy_alerts(top_peers, returns, mover_threshold, flat_threshold)
    else:
        alerts = _check_global_alerts(top_peers, returns, mover_threshold, flat_threshold)

    result = pd.DataFrame(alerts)
    if not result.empty:
        log.info(f"Global peer-relative alerts: {len(result)}")
        for _, a in result.iterrows():
            log.info(f"  {a['signal']}")
    else:
        log.info("No global peer-relative alerts")

    return result


def _check_global_alerts(peer_df, returns, mover_threshold, flat_threshold):
    """Check alerts using global peer mapping format."""
    alerts = []
    seen = set()

    for _, row in peer_df.iterrows():
        ticker_a = str(row["ticker"])
        ticker_b = str(row["peer_ticker"])
        market_a = row["market"]
        market_b = row["peer_market"]
        score = row.get("peer_score", 0)

        ret_a = returns.get(ticker_a)
        ret_b = returns.get(ticker_b)

        if ret_a is None or ret_b is None:
            continue

        # Check both directions: A moved + B flat, or B moved + A flat
        pair_key = tuple(sorted([ticker_a, ticker_b]))
        if pair_key in seen:
            continue

        if abs(ret_a) >= mover_threshold and abs(ret_b) < flat_threshold:
            seen.add(pair_key)
            direction = "up" if ret_a > 0 else "down"
            name_a = row.get("name", ticker_a)
            name_b = row.get("peer_name", ticker_b)
            alerts.append({
                "alert_type": "peer_divergence",
                "mover_ticker": ticker_a,
                "mover_market": market_a,
                "mover_name": name_a,
                "mover_return": round(ret_a, 4),
                "flat_ticker": ticker_b,
                "flat_market": market_b,
                "flat_name": name_b,
                "flat_return": round(ret_b, 4),
                "peer_score": score,
                "signal": (
                    f"[{market_a}] {name_a} ({ticker_a}) {direction} {abs(ret_a):.1%}, "
                    f"but peer [{market_b}] {name_b} ({ticker_b}) flat ({ret_b:+.1%})"
                ),
            })

        elif abs(ret_b) >= mover_threshold and abs(ret_a) < flat_threshold:
            seen.add(pair_key)
            direction = "up" if ret_b > 0 else "down"
            name_a = row.get("name", ticker_a)
            name_b = row.get("peer_name", ticker_b)
            alerts.append({
                "alert_type": "peer_divergence",
                "mover_ticker": ticker_b,
                "mover_market": market_b,
                "mover_name": name_b,
                "mover_return": round(ret_b, 4),
                "flat_ticker": ticker_a,
                "flat_market": market_a,
                "flat_name": name_a,
                "flat_return": round(ret_a, 4),
                "peer_score": score,
                "signal": (
                    f"[{market_b}] {name_b} ({ticker_b}) {direction} {abs(ret_b):.1%}, "
                    f"but peer [{market_a}] {name_a} ({ticker_a}) flat ({ret_a:+.1%})"
                ),
            })

    return alerts


def _check_legacy_alerts(peer_df, returns, mover_threshold, flat_threshold):
    """Check alerts using legacy CN->US peer mapping format."""
    alerts = []
    for _, row in peer_df.iterrows():
        us_ticker = str(row["us_ticker"])
        cn_ticker = str(row["cn_ticker"])

        us_ret = returns.get(us_ticker)
        cn_ret = returns.get(cn_ticker)

        if us_ret is None or cn_ret is None:
            continue

        if abs(us_ret) >= mover_threshold and abs(cn_ret) < flat_threshold:
            direction = "up" if us_ret > 0 else "down"
            alerts.append({
                "alert_type": "peer_divergence",
                "mover_ticker": us_ticker,
                "mover_market": "US",
                "mover_name": row.get("us_name", us_ticker),
                "mover_return": round(us_ret, 4),
                "flat_ticker": cn_ticker,
                "flat_market": "CN",
                "flat_name": row.get("cn_name", cn_ticker),
                "flat_return": round(cn_ret, 4),
                "peer_score": row.get("peer_score", 0),
                "signal": (
                    f"[US] {row.get('us_name', us_ticker)} ({us_ticker}) {direction} "
                    f"{abs(us_ret):.1%}, but peer [CN] {row.get('cn_name', cn_ticker)} "
                    f"({cn_ticker}) flat ({cn_ret:+.1%})"
                ),
                # Legacy fields
                "cn_name": row.get("cn_name", ""),
                "cn_ticker": cn_ticker,
                "cn_return_1d": round(cn_ret, 4),
                "us_name": row.get("us_name", ""),
                "us_ticker": us_ticker,
                "us_return_1d": round(us_ret, 4),
            })

        elif abs(cn_ret) >= mover_threshold and abs(us_ret) < flat_threshold:
            direction = "up" if cn_ret > 0 else "down"
            alerts.append({
                "alert_type": "peer_divergence",
                "mover_ticker": cn_ticker,
                "mover_market": "CN",
                "mover_name": row.get("cn_name", cn_ticker),
                "mover_return": round(cn_ret, 4),
                "flat_ticker": us_ticker,
                "flat_market": "US",
                "flat_name": row.get("us_name", us_ticker),
                "flat_return": round(us_ret, 4),
                "peer_score": row.get("peer_score", 0),
                "signal": (
                    f"[CN] {row.get('cn_name', cn_ticker)} ({cn_ticker}) {direction} "
                    f"{abs(cn_ret):.1%}, but peer [US] {row.get('us_name', us_ticker)} "
                    f"({us_ticker}) flat ({us_ret:+.1%})"
                ),
                "cn_name": row.get("cn_name", ""),
                "cn_ticker": cn_ticker,
                "cn_return_1d": round(cn_ret, 4),
                "us_name": row.get("us_name", ""),
                "us_ticker": us_ticker,
                "us_return_1d": round(us_ret, 4),
            })

    return alerts


def check_peer_relative_alerts(latest_signals_df, peer_mapping_df=None):
    """Backward-compatible wrapper: delegates to check_global_peer_alerts.

    Accepts the same interface as the old function.
    Ensures result DataFrame has legacy cn_ticker/us_ticker fields for
    callers that depend on them.
    """
    result = check_global_peer_alerts(
        latest_signals_df,
        peer_mapping_df,
        mover_threshold=0.03,
        flat_threshold=0.01,
    )
    if not result.empty and "cn_ticker" not in result.columns:
        # Add legacy-compat fields so pipeline code that reads cn_ticker still works
        result["cn_ticker"] = result["mover_ticker"]
        result["cn_name"] = result.get("mover_name", result["mover_ticker"])
        result["cn_return_1d"] = result["mover_return"]
        result["us_ticker"] = result["flat_ticker"]
        result["us_name"] = result.get("flat_name", result["flat_ticker"])
        result["us_return_1d"] = result["flat_return"]
    return result
