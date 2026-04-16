#!/usr/bin/env python3
"""CLI script to refresh financials data for all markets (or a subset).

Usage:
    python refresh_financials.py               # all markets
    python refresh_financials.py CN            # CN only
    python refresh_financials.py CN US HK      # specific markets
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.financials.normalize_financials import refresh_all_financials, refresh_cn_financials, refresh_market_financials
from src.common.logger import get_logger

log = get_logger("refresh_financials")

if __name__ == "__main__":
    markets = sys.argv[1:] if len(sys.argv) > 1 else None

    if markets:
        log.info(f"Refreshing financials for: {markets}")
        for mkt in markets:
            if mkt == "CN":
                refresh_cn_financials()
            else:
                refresh_market_financials(mkt)
    else:
        log.info("Refreshing financials for ALL markets")
        refresh_all_financials()

    log.info("Done.")
