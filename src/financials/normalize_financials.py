import pandas as pd
from src.common.config import get_data_path
from src.common.logger import get_logger
from src.financials.cn_financials import fetch_cn_financials_batch
from src.financials.us_financials import fetch_us_financials_batch

log = get_logger("financials.normalize")

COMMON_COLUMNS = [
    "ticker", "market", "report_date", "revenue", "revenue_growth",
    "gross_margin", "operating_margin", "net_margin", "eps", "pe",
    "ps", "market_cap",
]


def build_financial_snapshot(cn_tickers, us_tickers):
    """Fetch and normalize financials for CN + US stocks."""
    log.info(f"Fetching CN financials ({len(cn_tickers)} stocks)...")
    cn_df = fetch_cn_financials_batch(cn_tickers)

    log.info(f"Fetching US financials ({len(us_tickers)} stocks)...")
    us_df = fetch_us_financials_batch(us_tickers)

    frames = []
    if not cn_df.empty:
        for col in COMMON_COLUMNS:
            if col not in cn_df.columns:
                cn_df[col] = None
        frames.append(cn_df[COMMON_COLUMNS])

    if not us_df.empty:
        for col in COMMON_COLUMNS:
            if col not in us_df.columns:
                us_df[col] = None
        frames.append(us_df[COMMON_COLUMNS])

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    path = get_data_path("processed", "financial_snapshot.parquet")
    combined.to_parquet(path, index=False)
    log.info(f"Saved financial snapshot: {len(combined)} stocks to {path}")
    return combined
