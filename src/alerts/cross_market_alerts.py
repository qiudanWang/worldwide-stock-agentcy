"""Cross-market alerts — wraps sector_alerts for backward compatibility."""

from src.alerts.sector_alerts import (
    check_cross_market_alerts,
    check_index_breakout_alerts,
)

__all__ = ["check_cross_market_alerts", "check_index_breakout_alerts"]
