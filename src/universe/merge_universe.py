import pandas as pd
from src.common.config import get_data_path
from src.common.logger import get_logger
from src.universe.cn_universe import build_cn_universe
from src.universe.us_universe import build_us_universe

log = get_logger("universe.merge")


def build_full_universe():
    """Build and merge CN + US tech universes."""
    cn = build_cn_universe()
    us = build_us_universe()

    combined = pd.concat([cn, us], ignore_index=True)
    log.info(f"Full universe: {len(combined)} stocks (CN={len(cn)}, US={len(us)})")

    path = get_data_path("processed", "tech_universe_master.parquet")
    combined.to_parquet(path, index=False)
    log.info(f"Saved to {path}")
    return combined
