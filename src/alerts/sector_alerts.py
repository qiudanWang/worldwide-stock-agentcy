"""Sector rotation and cross-market divergence alerts."""

from src.common.config import load_yaml
from src.common.logger import get_logger

log = get_logger("alerts.sector")


def check_sector_rotation_alerts(rotation_events):
    """Generate alerts from sector rotation events.

    Args:
        rotation_events: List of dicts from detect_sector_rotation()

    Returns:
        List of alert dicts.
    """
    rules = load_yaml("alert_rules.yaml")
    threshold = rules.get("sector_rotation_alert", {}).get("rank_change_threshold", 5)
    alerts = []

    for event in rotation_events:
        if abs(event["rank_change"]) >= threshold:
            ret_5d = event.get("return_5d", 0) or 0
            if event["direction"] == "rising":
                # Rank improved — but only call it "momentum" if return is also positive
                label = "gaining momentum" if ret_5d > 0 else "gaining relative strength"
                trend = f"{label} — moved to #{event['rank_5d']} best sector in {event['market']} (was #{event['rank_20d']})"
            else:
                label = "losing momentum" if ret_5d < 0 else "losing relative strength"
                trend = f"{label} — dropped to #{event['rank_5d']} in {event['market']} (was #{event['rank_20d']})"
            alerts.append({
                "alert_type": "sector_rotation",
                "ticker": "",
                "sector": event["sector"],
                "market": event["market"],
                "name": f"{event['sector']} ({event['market']})",
                "signal": f"{trend}  ·  5d return: {event['return_5d']:.1%}",
                "rank_change": event["rank_change"],
            })

    if alerts:
        log.info(f"Sector rotation alerts: {len(alerts)}")
    return alerts


def check_cross_market_alerts(divergence_events):
    """Generate alerts from cross-market divergence events.

    Args:
        divergence_events: List of dicts from detect_cross_market_divergence()

    Returns:
        List of alert dicts.
    """
    rules = load_yaml("alert_rules.yaml")
    threshold = rules.get("cross_market_alert", {}).get("same_sector_divergence", 0.03)
    alerts = []

    for event in divergence_events:
        if abs(event["spread"]) >= threshold:
            alerts.append({
                "alert_type": "cross_market_divergence",
                "ticker": "",
                "sector": event["sector"],
                "market": "GLOBAL",
                "name": event["sector"],
                "signal": (
                    f"{event['best_market']} {event['best_return']:+.1%} vs "
                    f"{event['worst_market']} {event['worst_return']:+.1%}  "
                    f"spread: {event['spread']:.1%}"
                ),
                "spread": event["spread"],
            })

    if alerts:
        log.info(f"Cross-market alerts: {len(alerts)}")
    return alerts


def check_index_breakout_alerts(breakout_events):
    """Generate alerts from index breakout events.

    Args:
        breakout_events: List of dicts from detect_index_breakout()

    Returns:
        List of alert dicts.
    """
    alerts = []
    for event in breakout_events:
        direction = "high" if event["type"] == "breakout_high" else "low"
        emoji = "⬆" if direction == "high" else "⬇"
        alerts.append({
            "alert_type": "index_breakout",
            "ticker": event["symbol"],
            "market": event.get("market", ""),
            "name": event["name"],
            "signal": (
                f"Breaks {event['lookback_days']}d {direction} {emoji}  "
                f"price: {event['price']:,.2f}  prev {direction}: {event['threshold']:,.2f}"
            ),
            "price": event["price"],
        })

    if alerts:
        log.info(f"Index breakout alerts: {len(alerts)}")
    return alerts
