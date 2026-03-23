"""Fetch and cache market cap data for all stocks."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.common.config import get_data_path
from src.market_data.market_cap import fetch_all_market_caps, save_market_caps

universe = pd.read_parquet(get_data_path("processed", "tech_universe_master.parquet"))
cn_tickers = universe[universe["market"] == "CN"]["ticker"].tolist()
us_tickers = universe[universe["market"] == "US"]["ticker"].tolist()

print(f"Fetching market cap for {len(cn_tickers)} CN + {len(us_tickers)} US stocks...")
cap_df = fetch_all_market_caps(us_tickers, cn_tickers)
save_market_caps(cap_df)
got = cap_df["market_cap"].notna().sum()
print(f"Done. Cached {got}/{len(cap_df)} market caps.")
