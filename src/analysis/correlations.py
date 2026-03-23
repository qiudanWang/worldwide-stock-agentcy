"""Cross-market correlation analysis."""

import pandas as pd
import numpy as np
from src.common.logger import get_logger

log = get_logger("analysis.corr")


def compute_index_correlations(index_data, window=20):
    """Compute rolling correlation matrix between market indices.

    Args:
        index_data: DataFrame with columns: date, symbol, close
        window: Rolling window in days.

    Returns:
        dict with: correlation_matrix (latest), lead_lag signals
    """
    if index_data.empty:
        return {"matrix": {}, "lead_lag": []}

    # Pivot to wide format: date x symbol
    pivot = index_data.pivot_table(
        index="date", columns="symbol", values="close"
    ).sort_index()

    if pivot.shape[1] < 2:
        return {"matrix": {}, "lead_lag": []}

    # Compute returns
    returns = pivot.pct_change().dropna()

    if len(returns) < window:
        return {"matrix": {}, "lead_lag": []}

    # Rolling correlation (latest window)
    recent = returns.tail(window)
    corr_matrix = recent.corr()

    # Convert to serializable format
    matrix = {}
    for col in corr_matrix.columns:
        matrix[col] = {
            idx: round(val, 3) for idx, val in corr_matrix[col].items()
        }

    # Lead-lag detection: correlate returns(t) with returns(t+1)
    lead_lag = []
    symbols = list(returns.columns)
    for i, s1 in enumerate(symbols):
        for s2 in symbols[i + 1:]:
            if s1 == s2:
                continue
            # s1 leading s2: corr(s1[t], s2[t+1])
            try:
                lead_corr = returns[s1].iloc[:-1].reset_index(drop=True).corr(
                    returns[s2].iloc[1:].reset_index(drop=True)
                )
                lag_corr = returns[s2].iloc[:-1].reset_index(drop=True).corr(
                    returns[s1].iloc[1:].reset_index(drop=True)
                )

                if abs(lead_corr) > 0.3 or abs(lag_corr) > 0.3:
                    if abs(lead_corr) > abs(lag_corr):
                        lead_lag.append({
                            "leader": s1,
                            "follower": s2,
                            "correlation": round(lead_corr, 3),
                            "direction": "positive" if lead_corr > 0 else "negative",
                        })
                    else:
                        lead_lag.append({
                            "leader": s2,
                            "follower": s1,
                            "correlation": round(lag_corr, 3),
                            "direction": "positive" if lag_corr > 0 else "negative",
                        })
            except Exception:
                continue

    log.info(f"Correlations: {len(symbols)} indices, {len(lead_lag)} lead-lag pairs")
    return {"matrix": matrix, "lead_lag": lead_lag}


def compute_sector_index_correlation(sector_returns, index_returns, window=20):
    """Compute correlation between sector returns and home market index.

    Args:
        sector_returns: DataFrame with: date, market, sector, return
        index_returns: DataFrame with: date, market, return

    Returns:
        DataFrame with: market, sector, correlation
    """
    if sector_returns.empty or index_returns.empty:
        return pd.DataFrame()

    results = []
    for (market, sector), group in sector_returns.groupby(["market", "sector"]):
        idx = index_returns[index_returns["market"] == market]
        if idx.empty or len(group) < window:
            continue

        merged = group[["date", "return"]].merge(
            idx[["date", "return"]].rename(columns={"return": "idx_return"}),
            on="date", how="inner"
        )

        if len(merged) >= window:
            corr = merged["return"].tail(window).corr(
                merged["idx_return"].tail(window)
            )
            results.append({
                "market": market,
                "sector": sector,
                "correlation": round(corr, 3) if pd.notna(corr) else None,
            })

    return pd.DataFrame(results)
