"""Macro-level alerts: VIX spikes, yield curve inversion, oil, gold, BTC, rates."""

from src.common.config import load_yaml
from src.common.logger import get_logger

log = get_logger("alerts.macro")


def check_macro_alerts(macro_latest, yield_curve=None):
    """Generate macro alerts from latest indicator values.

    Args:
        macro_latest: dict of indicator -> {value, change_pct, date}
        yield_curve: DataFrame with yield curve data (optional)

    Returns:
        List of alert dicts.
    """
    rules = load_yaml("alert_rules.yaml")
    macro_rules = rules.get("macro_alert", {})
    alerts = []

    # VIX spike (high threshold)
    vix = macro_latest.get("VIX", {})
    vix_spike_threshold = macro_rules.get("vix_spike_threshold", 25)
    vix_elevated_threshold = macro_rules.get("vix_elevated_threshold", 20)

    if vix and vix.get("value") is not None:
        if vix["value"] >= vix_spike_threshold:
            alerts.append({
                "alert_type": "macro_alert",
                "sub_type": "vix_spike",
                "ticker": "^VIX",
                "name": "VIX — Volatility Index",
                "market": "US",
                "signal": f"VIX at {vix['value']:.1f} — market fear is HIGH (above {vix_spike_threshold}). VIX measures expected S&P 500 volatility over the next 30 days; spikes signal panic or uncertainty.",
                "value": vix["value"],
            })
        elif vix["value"] >= vix_elevated_threshold:
            alerts.append({
                "alert_type": "macro_alert",
                "sub_type": "vix_elevated",
                "ticker": "^VIX",
                "name": "VIX — Volatility Index",
                "market": "US",
                "signal": f"VIX at {vix['value']:.1f} — elevated caution (above {vix_elevated_threshold}). Markets are becoming more uncertain; watch for increased volatility.",
                "value": vix["value"],
            })

    # DXY big move
    dxy = macro_latest.get("US Dollar Index (DXY)", {})
    dxy_threshold = macro_rules.get("dxy_move_threshold", 0.01)
    if dxy and dxy.get("change_pct") and abs(dxy["change_pct"]) >= dxy_threshold:
        direction = "up ↑" if dxy["change_pct"] > 0 else "down ↓"
        alerts.append({
            "alert_type": "macro_alert",
            "sub_type": "dxy_move",
            "ticker": "DXY",
            "name": "US Dollar Index",
            "market": "US",
            "signal": f"US Dollar Index moved {direction} {abs(dxy['change_pct']):.1%} to {dxy['value']:.1f}. A stronger dollar pressures commodities and emerging markets; a weaker dollar typically boosts them.",
            "value": dxy["value"],
        })

    # Oil move (CL=F)
    oil = macro_latest.get("Crude Oil (WTI)", {}) or macro_latest.get("CL=F", {})
    oil_threshold = macro_rules.get("oil_move_threshold", 0.03)
    if oil and oil.get("change_pct") and abs(oil["change_pct"]) >= oil_threshold:
        direction = "up ↑" if oil["change_pct"] > 0 else "down ↓"
        alerts.append({
            "alert_type": "macro_alert",
            "sub_type": "oil_move",
            "ticker": "CL=F",
            "name": "Crude Oil WTI",
            "market": "GLOBAL",
            "signal": f"Crude oil (WTI) moved {direction} {abs(oil['change_pct']):.1%} to ${oil['value']:.2f}/barrel. Large oil moves affect inflation, transport costs, and energy stocks globally.",
            "value": oil["value"],
        })

    # Gold move (GC=F)
    gold = macro_latest.get("Gold", {}) or macro_latest.get("GC=F", {})
    gold_threshold = macro_rules.get("gold_move_threshold", 0.02)
    if gold and gold.get("change_pct") and abs(gold["change_pct"]) >= gold_threshold:
        direction = "up ↑" if gold["change_pct"] > 0 else "down ↓"
        alerts.append({
            "alert_type": "macro_alert",
            "sub_type": "gold_move",
            "ticker": "GC=F",
            "name": "Gold (safe haven)",
            "market": "GLOBAL",
            "signal": f"Gold moved {direction} {abs(gold['change_pct']):.1%} to ${gold['value']:.2f}/oz. Gold rising often signals risk-off sentiment or inflation fears; falling gold suggests risk appetite is returning.",
            "value": gold["value"],
        })

    # Bitcoin move (BTC-USD)
    btc = macro_latest.get("Bitcoin", {}) or macro_latest.get("BTC-USD", {})
    btc_threshold = macro_rules.get("btc_move_threshold", 0.05)
    if btc and btc.get("change_pct") and abs(btc["change_pct"]) >= btc_threshold:
        direction = "up ↑" if btc["change_pct"] > 0 else "down ↓"
        alerts.append({
            "alert_type": "macro_alert",
            "sub_type": "btc_move",
            "ticker": "BTC-USD",
            "name": "Bitcoin",
            "market": "GLOBAL",
            "signal": f"Bitcoin moved {direction} {abs(btc['change_pct']):.1%} to ${btc['value']:,.0f}. Large BTC moves often reflect shifts in risk appetite and can lead tech/growth stock sentiment.",
            "value": btc["value"],
        })

    # 10Y Treasury rate move
    rates = macro_latest.get("US 10Y Treasury", {}) or macro_latest.get("^TNX", {})
    rates_threshold = macro_rules.get("rates_move_threshold", 0.05)
    if rates and rates.get("change_pct") and abs(rates["change_pct"]) >= rates_threshold:
        direction = "up ↑" if rates["change_pct"] > 0 else "down ↓"
        alerts.append({
            "alert_type": "macro_alert",
            "sub_type": "rates_move",
            "ticker": "^TNX",
            "name": "US 10Y Treasury Yield",
            "market": "US",
            "signal": f"US 10Y Treasury yield moved {direction} {abs(rates['change_pct']):.2%} to {rates['value']:.2f}%. Rising yields increase borrowing costs and pressure growth/tech valuations; falling yields do the opposite.",
            "value": rates["value"],
        })

    # Yield curve inversion
    if yield_curve is not None and not yield_curve.empty:
        latest = yield_curve.iloc[-1]
        if latest.get("inverted", False):
            alerts.append({
                "alert_type": "macro_alert",
                "sub_type": "yield_curve_inversion",
                "ticker": "YIELD_CURVE",
                "name": "Yield Curve (10Y-2Y)",
                "market": "US",
                "signal": f"Yield curve inverted: 10Y-2Y spread is {latest['spread']:.2f}%. When short-term rates exceed long-term rates, it historically signals a recession within 12–18 months.",
                "value": latest["spread"],
            })

    if alerts:
        log.info(f"Macro alerts: {len(alerts)} ({', '.join(a['sub_type'] for a in alerts)})")
    return alerts
