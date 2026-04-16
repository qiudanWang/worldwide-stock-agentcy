import akshare as ak
import pandas as pd
from src.common.config import get_data_path
from src.common.logger import get_logger

log = get_logger("capital.northbound")


def fetch_northbound_flow():
    """Fetch daily northbound (北向资金) net flow data.

    NOTE: China's exchanges discontinued public northbound flow reporting after
    August 2024. The stock_hsgt_hist_em endpoint returns NaN for 当日成交净买额
    from Aug 17, 2024 onwards. No alternative akshare endpoint provides this
    data post-cutoff. The returned DataFrame will only contain data up to the
    last valid date (~Aug 16, 2024).
    """
    # Combine 沪股通 + 深股通 to reconstruct total northbound
    frames = []
    for symbol in ("沪股通", "深股通"):
        try:
            df = ak.stock_hsgt_hist_em(symbol=symbol)
            if df is None or df.empty:
                continue
            flow_col = "当日成交净买额"
            date_col = "日期"
            if flow_col not in df.columns or date_col not in df.columns:
                log.warning(f"[{symbol}] Unexpected columns: {df.columns.tolist()}")
                continue
            df = df[[date_col, flow_col]].rename(columns={date_col: "date", flow_col: "net_flow"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.dropna(subset=["net_flow"])
            frames.append(df)
        except Exception as e:
            log.warning(f"[{symbol}] fetch failed: {e}")

    if not frames:
        log.warning("No valid northbound flow data from either 沪股通 or 深股通")
        return pd.DataFrame()

    # Sum both channels per date
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.groupby("date", as_index=False)["net_flow"].sum()
    combined = combined.sort_values("date").reset_index(drop=True)

    last_date = combined.iloc[-1]["date"].date()
    log.info(f"Fetched northbound flow: {len(combined)} days (up to {last_date})")
    if (pd.Timestamp.today() - combined.iloc[-1]["date"]).days > 30:
        log.warning(
            f"CN northbound flow data ends at {last_date}. "
            "China exchanges discontinued northbound flow reporting after Aug 2024."
        )
    return combined


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
