"""Fetch financial data for all stocks in the universe."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.common.config import get_data_path
from src.financials.normalize_financials import build_financial_snapshot

if __name__ == "__main__":
    cn_path = get_data_path("processed", "cn_tech_universe.parquet")
    us_path = get_data_path("processed", "us_tech_universe.parquet")

    if not os.path.exists(cn_path) or not os.path.exists(us_path):
        print("Universe not built yet. Run: python scripts/build_universe.py")
        sys.exit(1)

    cn = pd.read_parquet(cn_path)
    us = pd.read_parquet(us_path)

    snapshot = build_financial_snapshot(
        cn["ticker"].tolist(),
        us["ticker"].tolist(),
    )
    print(f"\nFinancial snapshot: {len(snapshot)} stocks")
    if not snapshot.empty:
        print(snapshot[["ticker", "market", "revenue", "gross_margin", "pe"]].head(10).to_string())
