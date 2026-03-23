import pandas as pd
from src.common.config import load_yaml
from src.common.logger import get_logger

log = get_logger("alerts.capital")


def check_capital_flow_alerts(northbound_flow_df):
    """Check for significant northbound capital flow."""
    rules = load_yaml("alert_rules.yaml")
    threshold = rules["capital_flow_alert"]["northbound_net_flow_threshold"]

    if northbound_flow_df.empty:
        return pd.DataFrame()

    if "net_flow" not in northbound_flow_df.columns:
        return pd.DataFrame()

    latest = northbound_flow_df.sort_values("date").tail(1)
    flow = latest["net_flow"].iloc[0]

    if abs(flow) >= threshold:
        direction = "inflow" if flow > 0 else "outflow"
        log.info(f"Capital flow alert: northbound {direction} {flow:,.0f}")
        return latest
    else:
        log.info("No capital flow alerts")
        return pd.DataFrame()
