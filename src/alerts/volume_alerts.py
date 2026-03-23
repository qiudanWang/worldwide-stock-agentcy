import pandas as pd
from src.common.config import load_yaml
from src.common.logger import get_logger

log = get_logger("alerts.volume")


def check_volume_alerts(latest_signals):
    """Find stocks with volume spikes above threshold.

    Returns alerts with two tiers:
    - volume_spike: volume_ratio >= 2.0x (configurable)
    - volume_surge: volume_ratio >= 5.0x (configurable, extreme tier)
    """
    rules = load_yaml("alert_rules.yaml")
    threshold = rules["volume_alert"]["volume_ratio_threshold"]
    surge_threshold = rules["volume_alert"].get("volume_surge_threshold", 5.0)

    if latest_signals.empty:
        return pd.DataFrame()

    alerts = latest_signals[latest_signals["volume_ratio"] >= threshold].copy()
    alerts = alerts.sort_values("volume_ratio", ascending=False)

    # Tag each alert with its tier
    alerts["volume_tier"] = "volume_spike"
    alerts.loc[alerts["volume_ratio"] >= surge_threshold, "volume_tier"] = "volume_surge"

    if not alerts.empty:
        surge_count = (alerts["volume_tier"] == "volume_surge").sum()
        spike_count = len(alerts) - surge_count
        log.info(
            f"Volume alerts: {len(alerts)} stocks above {threshold}x "
            f"({surge_count} surges at {surge_threshold}x+, {spike_count} spikes)"
        )
        for _, row in alerts.iterrows():
            ret = row.get("return_1d", 0) or 0
            tier_label = "SURGE" if row["volume_tier"] == "volume_surge" else "spike"
            log.info(
                f"  [{tier_label}] {row['ticker']} ({row['market']}): "
                f"volume_ratio={row['volume_ratio']:.1f}x, "
                f"return_1d={ret:+.2%}"
            )
    else:
        log.info("No volume alerts today")

    return alerts
