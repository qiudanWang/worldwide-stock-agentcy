"""Fetch northbound capital flow data."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.common.config import get_data_path
from src.capital_flow.northbound import (
    fetch_northbound_flow,
    fetch_northbound_holdings,
    get_tech_northbound_changes,
)
from src.common.logger import get_logger

log = get_logger("script.capital")

if __name__ == "__main__":
    # Fetch northbound flow
    log.info("Fetching northbound flow...")
    flow = fetch_northbound_flow()
    if not flow.empty:
        path = get_data_path("processed", "northbound_flow.parquet")
        flow.to_parquet(path, index=False)
        latest = flow.sort_values("date").tail(5)
        print("\nRecent northbound flow:")
        print(latest.to_string())

    # Fetch northbound holdings
    log.info("Fetching northbound holdings...")
    holdings = fetch_northbound_holdings()
    if not holdings.empty:
        # Filter to tech universe
        cn_path = get_data_path("processed", "cn_tech_universe.parquet")
        if os.path.exists(cn_path):
            cn = pd.read_parquet(cn_path)
            tech_holdings = get_tech_northbound_changes(holdings, cn["ticker"].tolist())
            if not tech_holdings.empty:
                path = get_data_path("processed", "tech_northbound_holdings.parquet")
                tech_holdings.to_parquet(path, index=False)
                print(f"\nTech northbound holdings: {len(tech_holdings)} stocks")
                print(tech_holdings.head(10).to_string())
