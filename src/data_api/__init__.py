"""
src/data_api — clean data access layer for the Global Tech Market pipeline.

Import from here rather than from submodules directly.

    from src.data_api import get_market_daily, get_macro_latest
    from src.data_api import run_data_agent, run_full_pipeline

Submodules:
    market   — per-market reads (universe, OHLCV, cap, indices, news, flow, alerts)
    global_  — cross-market reads (macro, sectors, correlations, global alerts)
    agents   — agent execution tools (run_data_agent, run_global_agent, etc.)
"""

from src.data_api.market import (
    get_universe,
    get_universe_tickers,
    get_market_daily,
    get_market_history,
    get_latest_signals,
    get_available_snapshot_dates,
    get_market_cap,
    get_market_cap_map,
    get_indices,
    get_latest_index_values,
    get_news,
    get_news_for_ticker,
    get_news_counts,
    get_capital_flow,
    get_latest_capital_flow,
    get_market_alerts,
    get_market_alerts_by_type,
)

from src.data_api.global_ import (
    list_markets,
    get_universe_master,
    get_universe_for_markets,
    search_universe,
    get_macro_indicators,
    get_macro_latest,
    get_macro_indicator_series,
    list_macro_indicators,
    get_sector_performance,
    get_top_sectors,
    get_sector_performance_for_ticker,
    get_correlations,
    get_correlation_pair,
    get_global_alerts,
    get_global_alert_summary,
    get_alerts_for_ticker,
    get_all_latest_signals,
    get_all_indices,
)

from src.data_api.agents import (
    AgentToolResult,
    run_data_agent,
    run_news_agent,
    run_signal_agent,
    run_all_agents_for_market,
    run_data_agents_all_markets,
    run_news_agents_all_markets,
    run_signal_agents_all_markets,
    run_global_agent,
    run_full_pipeline,
    get_agent_status,
    list_available_agents,
)

__all__ = [
    # market
    "get_universe", "get_universe_tickers",
    "get_market_daily", "get_market_history", "get_latest_signals",
    "get_available_snapshot_dates",
    "get_market_cap", "get_market_cap_map",
    "get_indices", "get_latest_index_values",
    "get_news", "get_news_for_ticker", "get_news_counts",
    "get_capital_flow", "get_latest_capital_flow",
    "get_market_alerts", "get_market_alerts_by_type",
    # global_
    "list_markets",
    "get_universe_master", "get_universe_for_markets", "search_universe",
    "get_macro_indicators", "get_macro_latest", "get_macro_indicator_series",
    "list_macro_indicators",
    "get_sector_performance", "get_top_sectors", "get_sector_performance_for_ticker",
    "get_correlations", "get_correlation_pair",
    "get_global_alerts", "get_global_alert_summary", "get_alerts_for_ticker",
    "get_all_latest_signals", "get_all_indices",
    # agents
    "AgentToolResult",
    "run_data_agent", "run_news_agent", "run_signal_agent",
    "run_all_agents_for_market",
    "run_data_agents_all_markets", "run_news_agents_all_markets",
    "run_signal_agents_all_markets",
    "run_global_agent", "run_full_pipeline",
    "get_agent_status", "list_available_agents",
]
