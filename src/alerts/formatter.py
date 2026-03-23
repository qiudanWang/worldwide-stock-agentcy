import json
from datetime import datetime
from src.common.config import get_data_path
from src.common.logger import get_logger
from src.common.stock_info import enrich_alerts

log = get_logger("alerts.formatter")


def format_volume_alerts(alerts_df):
    """Format volume alerts as a list of dicts.

    Handles both volume_spike and volume_surge tiers.
    """
    if alerts_df.empty:
        return []

    records = []
    for _, row in alerts_df.iterrows():
        ret = row.get("return_1d", 0) or 0
        mc = row.get("market_cap", None)
        tier = row.get("volume_tier", "volume_spike")
        records.append({
            "date": str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
            "ticker": row["ticker"],
            "market": row["market"],
            "alert_type": tier,
            "volume_ratio": round(row["volume_ratio"], 2),
            "return_1d": round(ret, 4),
            "close": round(row["close"], 2),
            "market_cap": float(mc) if mc is not None and mc == mc else None,
        })
    return records


def format_price_alerts(alerts):
    """Format price alerts for display."""
    records = []
    for a in alerts:
        ret = a.get("return_1d", 0)
        direction = "UP" if ret > 0 else "DOWN"
        records.append({
            **a,
            "display": (
                f"[{a.get('market', '')}] {a['ticker']}: "
                f"price {direction} {abs(ret):.1%} to {a.get('close', 0):.2f}"
            ),
        })
    return records


def format_gap_alerts(alerts):
    """Format gap alerts for display."""
    records = []
    for a in alerts:
        gap = a.get("gap_pct", 0)
        direction = "GAP UP" if gap > 0 else "GAP DOWN"
        records.append({
            **a,
            "display": (
                f"[{a.get('market', '')}] {a['ticker']}: "
                f"{direction} {abs(gap):.1%} "
                f"(prev close {a.get('prev_close', 0):.2f} -> "
                f"open {a.get('open', 0):.2f})"
            ),
        })
    return records


def format_correlation_break_alerts(alerts):
    """Format correlation break alerts for display."""
    records = []
    for a in alerts:
        records.append({
            **a,
            "display": (
                f"[CORRELATION] {a.get('pair', '')}: "
                f"correlation dropped from {a.get('historical_corr', 0):.2f} "
                f"to {a.get('current_corr', 0):.2f}"
            ),
        })
    return records


def format_macro_alerts(alerts):
    """Format macro alerts (all sub-types) for display."""
    records = []
    for a in alerts:
        sub = a.get("sub_type", "unknown")
        records.append({
            **a,
            "display": f"[MACRO/{sub.upper()}] {a.get('signal', '')}",
        })
    return records


def save_alerts(alerts_list, filename=None):
    """Save alerts to JSON file."""
    if not alerts_list:
        log.info("No alerts to save")
        return

    if filename is None:
        today = datetime.now().strftime("%Y%m%d")
        filename = f"alerts_{today}.json"

    path = get_data_path("snapshots", filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(alerts_list, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(alerts_list)} alerts to {path}")


def print_alerts(alerts_list):
    """Print alerts to console, handling all alert types."""
    if not alerts_list:
        print("\n=== No alerts today ===\n")
        return

    alerts_list = enrich_alerts(alerts_list)
    print(f"\n=== {len(alerts_list)} ALERTS ===\n")

    # Group by alert type for cleaner output
    by_type = {}
    for a in alerts_list:
        atype = a.get("alert_type", "unknown")
        by_type.setdefault(atype, []).append(a)

    for atype, group in sorted(by_type.items()):
        print(f"--- {atype.upper().replace('_', ' ')} ({len(group)}) ---")
        for a in group:
            _print_single_alert(a)
        print()

    print()


def _print_single_alert(a):
    """Print a single alert based on its type."""
    atype = a.get("alert_type", "")
    name = a.get("name", a.get("ticker", ""))
    market = a.get("market", "")
    board = a.get("board", "")

    if atype in ("volume_spike", "volume_surge"):
        ret = a.get("return_1d", 0)
        tier = "SURGE" if atype == "volume_surge" else "spike"
        print(
            f"  [{market}] {a.get('ticker', '')} {name} ({board}): "
            f"[{tier}] volume {a.get('volume_ratio', 0)}x | "
            f"return {ret:+.2%} | "
            f"close {a.get('close', 0)}"
        )

    elif atype == "price_alert":
        ret = a.get("return_1d", 0)
        direction = "UP" if ret > 0 else "DOWN"
        print(
            f"  [{market}] {a.get('ticker', '')} {name}: "
            f"price {direction} {abs(ret):.1%} to {a.get('close', 0):.2f}"
        )

    elif atype == "gap_alert":
        gap = a.get("gap_pct", 0)
        direction = "GAP UP" if gap > 0 else "GAP DOWN"
        print(
            f"  [{market}] {a.get('ticker', '')} {name}: "
            f"{direction} {abs(gap):.1%} "
            f"(prev {a.get('prev_close', 0):.2f} -> open {a.get('open', 0):.2f})"
        )

    elif atype == "macro_alert":
        sub = a.get("sub_type", "")
        print(f"  [MACRO/{sub}] {a.get('signal', '')}")

    elif atype == "correlation_break":
        print(
            f"  [CORRELATION] {a.get('pair', '')}: "
            f"corr {a.get('historical_corr', 0):.2f} -> {a.get('current_corr', 0):.2f}"
        )

    elif atype == "news_spike":
        print(
            f"  [{market}] {a.get('ticker', '')} {name}: "
            f"{a.get('news_count_keyword', 0)} keyword hits"
        )

    elif atype == "capital_flow":
        flow = a.get("net_flow")
        if flow is not None:
            print(f"  [{market}] {a.get('ticker', '')}: net flow {flow:,.0f}")
        else:
            print(f"  [{market}] {a.get('ticker', '')}: {a.get('signal', '')}")

    elif atype == "sector_rotation":
        print(f"  {a.get('signal', '')}")

    elif atype == "cross_market_divergence":
        print(f"  {a.get('signal', '')}")

    elif atype == "index_breakout":
        print(f"  {a.get('signal', '')}")

    elif atype == "peer_relative":
        print(f"  [{market}] {a.get('ticker', '')}: {a.get('signal', '')}")

    else:
        # Fallback for any unknown alert type
        print(f"  [{market}] {a.get('ticker', '')} {name}: {a.get('signal', atype)}")
