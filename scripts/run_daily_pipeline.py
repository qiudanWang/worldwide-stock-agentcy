"""Run the full daily pipeline: universe -> market data -> news -> capital flow -> alerts."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from datetime import datetime, timedelta
from src.common.config import get_data_path
from src.common.logger import get_logger
from src.universe.merge_universe import build_full_universe
from src.market_data.cn_market_data import fetch_cn_batch
from src.market_data.us_market_data import fetch_us_batch
from src.market_data.volume_signals import compute_volume_signals, get_latest_signals
from src.alerts.volume_alerts import check_volume_alerts
from src.alerts.news_alerts import check_news_alerts
from src.alerts.capital_flow_alerts import check_capital_flow_alerts
from src.alerts.peer_relative_alerts import check_peer_relative_alerts
from src.alerts.formatter import format_volume_alerts, save_alerts, print_alerts
from src.news.us_news import fetch_us_news_batch
from src.news.cn_news import fetch_cn_news_batch
from src.news.keyword_filter import filter_by_keywords, compute_news_counts
from src.news.world_monitor import fetch_geopolitical_context
from src.capital_flow.northbound import fetch_northbound_flow
from src.common.stock_info import enrich_alerts
from src.market_data.market_cap import fetch_all_market_caps, save_market_caps

log = get_logger("pipeline.daily")

# Key tickers for news (not all 600+)
CN_NEWS_TICKERS = [
    "002230", "688256", "000977", "603019", "301269",
    "300454", "688561", "600588", "600570", "688111",
    "688981", "603501", "000063", "000938", "688041",
]
US_NEWS_TICKERS = [
    "NVDA", "AMD", "MSFT", "GOOGL", "META", "PLTR",
    "ORCL", "PANW", "CRWD", "SMCI", "TSM", "AVGO",
]


def main():
    today = datetime.now().strftime("%Y%m%d")
    log.info(f"=== Daily Pipeline {today} ===")
    all_alerts = []

    # Step 1: Universe
    master_path = get_data_path("processed", "tech_universe_master.parquet")
    if not os.path.exists(master_path):
        log.info("Building universe...")
        universe = build_full_universe()
    else:
        universe = pd.read_parquet(master_path)
        log.info(f"Loaded universe: {len(universe)} stocks")

    cn_tickers = universe[universe["market"] == "CN"]["ticker"].tolist()
    us_tickers = universe[universe["market"] == "US"]["ticker"].tolist()

    # Step 2: Market data (skip if today's snapshot already exists)
    path = get_data_path("snapshots", f"market_daily_{today}.parquet")
    if os.path.exists(path):
        log.info(f"Today's snapshot already exists, loading from cache...")
        all_data = pd.read_parquet(path)
    else:
        log.info(f"Fetching CN market data ({len(cn_tickers)} stocks)...")
        cn_data = fetch_cn_batch(cn_tickers)

        log.info(f"Fetching US market data ({len(us_tickers)} stocks)...")
        us_data = fetch_us_batch(us_tickers)

        all_data = pd.concat([cn_data, us_data], ignore_index=True)
        all_data = compute_volume_signals(all_data)

        # Step 2b: Market cap — skip if cache is fresh (< 1 day old)
        cap_path = get_data_path("processed", "market_cap.parquet")
        cap_fresh = False
        if os.path.exists(cap_path):
            age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cap_path))
            if age < timedelta(days=1):
                cap_fresh = True
                log.info(f"Market cap cache is fresh ({age.seconds // 3600}h old), skipping fetch")

        if cap_fresh:
            cap_df = pd.read_parquet(cap_path)
            all_data = all_data.merge(cap_df, on="ticker", how="left")
        else:
            log.info("Fetching market cap data...")
            cap_df = fetch_all_market_caps(us_tickers, cn_tickers)
            if not cap_df.empty:
                save_market_caps(cap_df)
                all_data = all_data.merge(cap_df, on="ticker", how="left")
            else:
                all_data["market_cap"] = None

        all_data.to_parquet(path, index=False)
        log.info(f"Saved market snapshot: {all_data['ticker'].nunique()} tickers")

    # Step 3: Volume alerts (kept for Top Volume table, not added to alerts)
    latest = get_latest_signals(all_data)
    vol_alerts = check_volume_alerts(latest)

    # Build a ticker->latest lookup for enriching other alerts with price/market_cap
    latest_lookup = {}
    for _, row in latest.iterrows():
        latest_lookup[row["ticker"]] = {
            "close": round(row["close"], 2) if pd.notna(row.get("close")) else None,
            "market_cap": float(row["market_cap"]) if pd.notna(row.get("market_cap")) else None,
        }

    # Step 4: News
    log.info("Fetching news...")
    us_news = fetch_us_news_batch(US_NEWS_TICKERS)
    cn_news = fetch_cn_news_batch(CN_NEWS_TICKERS)
    news_df = pd.concat([us_news, cn_news], ignore_index=True)

    if not news_df.empty:
        news_df = filter_by_keywords(news_df)

        # Save news feed
        news_path = get_data_path("processed", "news_feed.parquet")
        news_df.to_parquet(news_path, index=False)
        log.info(f"Saved {len(news_df)} news items to {news_path}")

        news_counts = compute_news_counts(news_df)
        news_alerts = check_news_alerts(news_counts)
        if not news_alerts.empty:
            for _, row in news_alerts.iterrows():
                lk = latest_lookup.get(row["ticker"], {})
                all_alerts.append({
                    "date": today,
                    "ticker": row["ticker"],
                    "market": row["market"],
                    "alert_type": "news_spike",
                    "news_count_keyword": int(row.get("news_count_keyword", 0)),
                    "close": lk.get("close"),
                    "market_cap": lk.get("market_cap"),
                })

    # Step 5: World Monitor geopolitical context
    log.info("Fetching geopolitical context...")
    geo_context = fetch_geopolitical_context()
    if geo_context:
        path = get_data_path("processed", "geopolitical_context.json")
        with open(path, "w") as f:
            json.dump(geo_context, f, indent=2, default=str)

    # Step 6: Capital flow
    log.info("Fetching northbound flow...")
    nb_flow = fetch_northbound_flow()
    cap_alerts = check_capital_flow_alerts(nb_flow)
    if not cap_alerts.empty:
        all_alerts.append({
            "date": today,
            "ticker": "NORTHBOUND",
            "market": "CN",
            "alert_type": "capital_flow",
            "net_flow": float(cap_alerts["net_flow"].iloc[0]),
        })

    # Step 7: Peer-relative alerts
    peer_alerts = check_peer_relative_alerts(latest)
    if not peer_alerts.empty:
        for _, row in peer_alerts.iterrows():
            lk = latest_lookup.get(row["cn_ticker"], {})
            all_alerts.append({
                "date": today,
                "ticker": row["cn_ticker"],
                "market": "CN",
                "alert_type": "peer_relative",
                "signal": row["signal"],
                "close": lk.get("close"),
                "market_cap": lk.get("market_cap"),
            })

    # Enrich alerts with name + board, then save and print
    all_alerts = enrich_alerts(all_alerts)
    save_alerts(all_alerts)

    print(f"\n{'='*60}")
    print(f"  DAILY REPORT — {today}")
    print(f"{'='*60}")
    print(f"  Stocks tracked: {all_data['ticker'].nunique()}")
    print(f"  News items: {len(news_df) if not news_df.empty else 0}")
    print(f"  Total alerts: {len(all_alerts)}")
    print(f"{'='*60}\n")

    if all_alerts:
        for a in all_alerts:
            atype = a.get("alert_type", "unknown")
            ticker = a.get("ticker", "?")
            name = a.get("name", ticker)
            board = a.get("board", "")
            label = f"{ticker} {name} ({board})" if board else f"{ticker} {name}"
            if atype == "volume_spike":
                print(f"  [{atype}] {label}: {a['volume_ratio']}x volume, {a['return_1d']:+.2%}")
            elif atype == "news_spike":
                print(f"  [{atype}] {label}: {a.get('news_count_keyword', 0)} keyword hits")
            elif atype == "capital_flow":
                print(f"  [{atype}] Northbound: {a.get('net_flow', 0):,.0f}")
            elif atype == "peer_relative":
                print(f"  [{atype}] {a.get('signal', '')}")
            else:
                print(f"  [{atype}] {label}")
        print()
    else:
        print("  No alerts today.\n")

    log.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
