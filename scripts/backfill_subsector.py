"""
One-time backfill: add subsector column to all existing universe.parquet files.

Usage:
    python scripts/backfill_subsector.py            # all markets
    python scripts/backfill_subsector.py CN US HK   # specific markets
"""
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

from src.agents.data_agent import _enrich_subsector_cn
from src.universe.yf_universe import enrich_subsector_yf
from src.common.config import get_data_path
from src.common.logger import get_logger

log = get_logger("backfill.subsector")


def backfill_market(market: str):
    path = get_data_path("markets", market, "universe.parquet")
    if not os.path.exists(path):
        log.warning(f"[{market}] No universe.parquet found, skipping")
        return

    df = pd.read_parquet(path)
    log.info(f"[{market}] Loaded {len(df)} stocks")

    # Already fully populated?
    if "subsector" in df.columns and (df["subsector"].fillna("") != "").mean() > 0.5:
        log.info(f"[{market}] subsector already populated ({(df['subsector'] != '').sum()}/{len(df)}), skipping")
        return

    # Ticker suffix needed for yfinance API calls (e.g. 700 → 700.HK)
    _SUFFIX = {"HK": ".HK", "JP": ".T", "AU": ".AX", "IN": ".NS",
               "KR": ".KS", "TW": ".TW", "DE": ".DE", "FR": ".PA",
               "UK": ".L", "BR": ".SA", "SA": ".SR"}

    if market == "CN":
        df = _enrich_subsector_cn(df)
        # If akshare failed, fall back to yfinance using .SS/.SZ suffixes
        if "subsector" not in df.columns or (df["subsector"].fillna("") == "").all():
            log.info("[CN] akshare failed, falling back to yfinance (.SS/.SZ)")
            df = df.copy()
            if "subsector" not in df.columns:
                df["subsector"] = ""

            def _cn_yf_symbol(ticker: str) -> str:
                t = str(ticker)
                if t.startswith("6"):
                    return f"{t}.SS"
                return f"{t}.SZ"

            df["_cn_yf"] = df["ticker"].apply(_cn_yf_symbol)
            df = enrich_subsector_yf(df, ticker_col="_cn_yf")
            df = df.drop(columns=["_cn_yf"], errors="ignore")
    elif "industry" in df.columns and (df["industry"].fillna("") != "").any():
        # Markets that already have yfinance industry populated (KR, TW, BR, DE, FR, UK, SA)
        df = df.copy()
        df["subsector"] = df["industry"].fillna("")
        log.info(f"[{market}] Copied industry → subsector ({(df['subsector'] != '').sum()}/{len(df)})")
    else:
        # Enrich from yfinance (US, HK, AU, JP, IN where industry is empty, etc.)
        suffix = _SUFFIX.get(market, "")
        if suffix and "yf_symbol" in df.columns:
            # Fix yf_symbol if it's missing the exchange suffix
            df = df.copy()
            mask = ~df["yf_symbol"].str.endswith(suffix, na=False)
            df.loc[mask, "yf_symbol"] = df.loc[mask, "yf_symbol"].astype(str) + suffix
        yf_col = "yf_symbol" if "yf_symbol" in df.columns else "ticker"
        df = enrich_subsector_yf(df, ticker_col=yf_col)

    df.to_parquet(path, index=False)
    filled = (df["subsector"].fillna("") != "").sum() if "subsector" in df.columns else 0
    log.info(f"[{market}] Saved. subsector filled: {filled}/{len(df)}")


if __name__ == "__main__":
    markets_arg = sys.argv[1:]
    if not markets_arg:
        data_dir = get_data_path("markets")
        markets_arg = sorted(os.listdir(data_dir)) if os.path.isdir(data_dir) else []

    log.info(f"Backfilling subsector for markets: {markets_arg}")
    for m in markets_arg:
        try:
            backfill_market(m)
        except Exception as e:
            log.error(f"[{m}] Failed: {e}")
    log.info("Done.")
