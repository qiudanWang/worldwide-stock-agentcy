"""Fetch market data for all stocks in the universe and compute signals."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime
from src.common.config import get_data_path
from src.common.logger import get_logger
from src.market_data.cn_market_data import fetch_cn_batch
from src.market_data.us_market_data import fetch_us_batch
from src.market_data.volume_signals import compute_volume_signals, get_latest_signals
from src.alerts.volume_alerts import check_volume_alerts
from src.alerts.formatter import format_volume_alerts, save_alerts, print_alerts

log = get_logger("script.market")


def main():
    # Load universe
    cn_path = get_data_path("processed", "cn_tech_universe.parquet")
    us_path = get_data_path("processed", "us_tech_universe.parquet")

    if not os.path.exists(cn_path) or not os.path.exists(us_path):
        print("Universe not built yet. Run: python scripts/build_universe.py")
        return

    cn_universe = pd.read_parquet(cn_path)
    us_universe = pd.read_parquet(us_path)

    # Fetch market data
    log.info(f"Fetching CN market data for {len(cn_universe)} stocks...")
    cn_data = fetch_cn_batch(cn_universe["ticker"].tolist())

    log.info(f"Fetching US market data for {len(us_universe)} stocks...")
    us_data = fetch_us_batch(us_universe["ticker"].tolist())

    # Combine
    all_data = pd.concat([cn_data, us_data], ignore_index=True)
    log.info(f"Total market data: {len(all_data)} rows, {all_data['ticker'].nunique()} tickers")

    # Compute signals
    all_data = compute_volume_signals(all_data)

    # Save full data
    today = datetime.now().strftime("%Y%m%d")
    path = get_data_path("snapshots", f"market_daily_{today}.parquet")
    all_data.to_parquet(path, index=False)
    log.info(f"Saved market data to {path}")

    # Get latest signals and check alerts
    latest = get_latest_signals(all_data)
    alerts_df = check_volume_alerts(latest)
    alerts_list = format_volume_alerts(alerts_df)
    save_alerts(alerts_list)
    print_alerts(alerts_list)


if __name__ == "__main__":
    main()
