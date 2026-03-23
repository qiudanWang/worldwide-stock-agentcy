import akshare as ak
import pandas as pd
from src.common.config import load_yaml, get_data_path
from src.common.logger import get_logger

log = get_logger("universe.cn")


def build_cn_universe():
    """Fetch A-share tech stocks by industry from AKShare."""
    cfg = load_yaml("cn_industries.yaml")
    industries = cfg["industries"]

    all_stocks = []
    for industry in industries:
        log.info(f"Fetching industry: {industry}")
        try:
            df = ak.stock_board_industry_cons_em(symbol=industry)
            df = df[["代码", "名称"]].copy()
            df.columns = ["ticker", "name"]
            df["sector"] = industry
            df["market"] = "CN"
            all_stocks.append(df)
            log.info(f"  {industry}: {len(df)} stocks")
        except Exception as e:
            log.warning(f"  Failed to fetch {industry}: {e}")

    if not all_stocks:
        log.error("No CN stocks fetched")
        return pd.DataFrame()

    result = pd.concat(all_stocks, ignore_index=True)
    result = result.drop_duplicates(subset=["ticker"], keep="first")
    log.info(f"CN universe total: {len(result)} unique stocks")

    path = get_data_path("processed", "cn_tech_universe.parquet")
    result.to_parquet(path, index=False)
    log.info(f"Saved to {path}")
    return result
