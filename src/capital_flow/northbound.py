import akshare as ak
import pandas as pd
from src.common.config import get_data_path
from src.common.logger import get_logger

log = get_logger("capital.northbound")


def fetch_northbound_flow():
    """Fetch daily northbound (北向资金) net flow data.

    Uses the historical endpoint (stock_hsgt_hist_em) which has data back to
    ~2015. The live summary endpoint stopped providing flow values after
    Aug 2024, so we rely on the historical series and fall back to today's
    summary only if the hist data has a gap.
    """
    try:
        df = ak.stock_hsgt_hist_em(symbol="北向资金")
        if df is None or df.empty:
            return pd.DataFrame()

        # The column 当日成交净买额 is the daily net buy (hundred millions CNY)
        flow_col = "当日成交净买额"
        date_col = "日期"
        if flow_col not in df.columns or date_col not in df.columns:
            log.warning(f"Unexpected columns: {df.columns.tolist()}")
            return pd.DataFrame()

        df = df[[date_col, flow_col]].rename(columns={date_col: "date", flow_col: "net_flow"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["net_flow"])
        df = df.sort_values("date").reset_index(drop=True)

        if df.empty:
            log.warning("No valid northbound flow rows after filtering NaN")
            return pd.DataFrame()

        log.info(f"Fetched northbound flow: {len(df)} days (up to {df.iloc[-1]['date'].date()})")
        return df
    except Exception as e:
        log.warning(f"Failed to fetch northbound flow: {e}")
        return pd.DataFrame()


def fetch_northbound_holdings():
    """Fetch northbound holding data for individual stocks."""
    try:
        df = ak.stock_hsgt_hold_stock_em(market="北向", indicator="今日排行")
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.rename(columns={
            "代码": "ticker",
            "名称": "name",
            "今日持股-股数": "holding_shares",
            "今日持股-市值": "holding_value",
            "今日持股-占流通股比": "holding_pct",
            "今日增持估计-股数": "change_shares",
            "今日增持估计-市值": "change_value",
        })
        log.info(f"Fetched northbound holdings: {len(df)} stocks")
        return df
    except Exception as e:
        log.warning(f"Failed to fetch northbound holdings: {e}")
        return pd.DataFrame()


def get_tech_northbound_changes(holdings_df, tech_tickers):
    """Filter northbound holdings to tech universe only."""
    if holdings_df.empty:
        return pd.DataFrame()

    tech = holdings_df[holdings_df["ticker"].isin(tech_tickers)].copy()
    if "change_value" in tech.columns:
        tech = tech.sort_values("change_value", ascending=False)
    log.info(f"Tech northbound changes: {len(tech)} stocks")
    return tech
