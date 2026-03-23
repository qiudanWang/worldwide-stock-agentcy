"""
Agent execution tools for the Global Tech Market pipeline.

These functions let the GlobalAgent (or any LLM-driven orchestrator) invoke
per-market agents as composable tools. Each function runs one agent for one
market and returns a structured AgentToolResult.

These are also the canonical LLM tool definitions: docstrings describe exactly
what each tool does, what arguments it accepts, and what it returns. An LLM with
access to these tools can reason about and drive the entire pipeline.

Example usage:
    from src.data_api.agents import run_data_agent, run_full_pipeline
    result = run_data_agent("JP")
    results = run_full_pipeline(markets=["US", "CN"])
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field


@dataclass
class AgentToolResult:
    """Structured result returned by every agent tool function."""
    market: str | None          # Market code, or None for the global agent.
    agent_type: str             # "data", "news", "signal", or "global".
    success: bool               # True if the agent completed without error.
    records_written: int = 0    # Number of records persisted to disk.
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    status_message: str = ""    # Human-readable status summary.

    @classmethod
    def from_agent_result(cls, market, agent_type, result) -> "AgentToolResult":
        msg = f"{result.records_written} records in {result.duration_seconds:.1f}s"
        if result.errors:
            msg += f" | errors: {'; '.join(result.errors)}"
        return cls(
            market=market,
            agent_type=agent_type,
            success=result.success,
            records_written=result.records_written,
            duration_seconds=result.duration_seconds,
            errors=result.errors,
            status_message=msg,
        )

    @classmethod
    def failure(cls, market, agent_type, error: str) -> "AgentToolResult":
        return cls(
            market=market,
            agent_type=agent_type,
            success=False,
            errors=[error],
            status_message=f"Failed: {error}",
        )


# ---------------------------------------------------------------------------
# Internal: build a single agent instance
# ---------------------------------------------------------------------------

def _build_agent(market: str, agent_type: str, force: bool = False):
    """Instantiate the correct agent class for a market + type."""
    from src.common.config import load_yaml
    markets_cfg = load_yaml("markets.yaml")["markets"]
    if market not in markets_cfg:
        raise ValueError(f"Unknown market: {market}. Available: {list(markets_cfg.keys())}")
    market_cfg = markets_cfg[market]

    if agent_type == "data":
        from src.agents.data_agent import DataAgent
        agent = DataAgent(name=f"{market}_data", market=market, market_config=market_cfg)
    elif agent_type == "news":
        from src.agents.news_agent import NewsAgent
        agent = NewsAgent(name=f"{market}_news", market=market, market_config=market_cfg)
    elif agent_type == "signal":
        from src.agents.signal_agent import SignalAgent
        agent = SignalAgent(name=f"{market}_signal", market=market, market_config=market_cfg)
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")

    agent.force = force
    return agent


# ---------------------------------------------------------------------------
# Per-market agent tools
# ---------------------------------------------------------------------------

def run_data_agent(market: str, force: bool = False) -> AgentToolResult:
    """Fetch and save market data for a single market.

    Runs the DataAgent which fetches the stock universe, OHLCV price history,
    market capitalisation, and index data for the specified market. Writes:
    universe.parquet, market_daily_{date}.parquet, market_cap.parquet, and
    indices.parquet under data/markets/{market}/.

    This is the first step in the pipeline for any market. NewsAgent and
    SignalAgent depend on it completing successfully.

    Args:
        market: Two-letter market code (e.g. "US", "CN", "JP").
        force:  If True, bypass the cache check and re-fetch even if
                today's snapshot already exists. Defaults to False.

    Returns:
        AgentToolResult with success status, record count, duration,
        and any error messages.
    """
    try:
        agent = _build_agent(market, "data", force=force)
        result = agent.execute()
        return AgentToolResult.from_agent_result(market, "data", result)
    except Exception as e:
        return AgentToolResult.failure(market, "data", str(e))


def run_news_agent(market: str, force: bool = False) -> AgentToolResult:
    """Fetch and save company news for a single market.

    Runs the NewsAgent which fetches recent news articles for key tickers,
    applies keyword filtering, and writes news.parquet under
    data/markets/{market}/. Depends on the universe file from DataAgent.

    Args:
        market: Two-letter market code (e.g. "US", "CN", "JP").
        force:  If True, re-fetch even if a fresh news file exists.

    Returns:
        AgentToolResult with success status, article count, duration,
        and any error messages.
    """
    try:
        agent = _build_agent(market, "news", force=force)
        result = agent.execute()
        return AgentToolResult.from_agent_result(market, "news", result)
    except Exception as e:
        return AgentToolResult.failure(market, "news", str(e))


def run_signal_agent(market: str, force: bool = False) -> AgentToolResult:
    """Compute capital flow and local alerts for a single market.

    Runs the SignalAgent which fetches capital flow data, detects volume
    spikes, news spikes, capital flow events, large price moves, and gap
    opens. Writes capital_flow.parquet and alerts.json under
    data/markets/{market}/. Depends on market_daily and news files.

    Args:
        market: Two-letter market code (e.g. "US", "CN", "JP").
        force:  If True, re-run even if today's alert file exists.

    Returns:
        AgentToolResult with success status, alert count, duration,
        and any error messages.
    """
    try:
        agent = _build_agent(market, "signal", force=force)
        result = agent.execute()
        return AgentToolResult.from_agent_result(market, "signal", result)
    except Exception as e:
        return AgentToolResult.failure(market, "signal", str(e))


# ---------------------------------------------------------------------------
# Multi-agent convenience tools
# ---------------------------------------------------------------------------

def run_all_agents_for_market(
    market: str, force: bool = False
) -> dict[str, AgentToolResult]:
    """Run DataAgent → NewsAgent → SignalAgent for a single market in order.

    Executes the full three-agent pipeline for one market sequentially.
    Stops early if a required upstream agent fails.

    Args:
        market: Two-letter market code (e.g. "US", "CN", "JP").
        force:  If True, bypass caches for all three agents.

    Returns:
        Dict mapping agent type string ("data", "news", "signal") to its
        AgentToolResult. All three keys are always present.
    """
    results: dict[str, AgentToolResult] = {}

    data_result = run_data_agent(market, force=force)
    results["data"] = data_result
    if not data_result.success:
        results["news"] = AgentToolResult.failure(market, "news", "Skipped: data agent failed")
        results["signal"] = AgentToolResult.failure(market, "signal", "Skipped: data agent failed")
        return results

    news_result = run_news_agent(market, force=force)
    results["news"] = news_result

    signal_result = run_signal_agent(market, force=force)
    results["signal"] = signal_result

    return results


def run_data_agents_all_markets(
    markets: list[str] | None = None,
    max_workers: int = 6,
    force: bool = False,
) -> dict[str, AgentToolResult]:
    """Run DataAgent for all (or selected) markets in parallel.

    Args:
        markets:     Market codes to run. None = all configured markets.
        max_workers: Maximum concurrent market agents. Defaults to 6.
        force:       If True, bypass caches for all markets.

    Returns:
        Dict mapping market code to its AgentToolResult.
    """
    return _run_parallel(run_data_agent, markets, max_workers, force)


def run_news_agents_all_markets(
    markets: list[str] | None = None,
    max_workers: int = 6,
    force: bool = False,
) -> dict[str, AgentToolResult]:
    """Run NewsAgent for all (or selected) markets in parallel.

    Args:
        markets:     Market codes to run. None = all configured markets.
        max_workers: Maximum concurrent market agents. Defaults to 6.
        force:       If True, bypass caches for all markets.

    Returns:
        Dict mapping market code to its AgentToolResult.
    """
    return _run_parallel(run_news_agent, markets, max_workers, force)


def run_signal_agents_all_markets(
    markets: list[str] | None = None,
    max_workers: int = 6,
    force: bool = False,
) -> dict[str, AgentToolResult]:
    """Run SignalAgent for all (or selected) markets in parallel.

    Args:
        markets:     Market codes to run. None = all configured markets.
        max_workers: Maximum concurrent market agents. Defaults to 6.
        force:       If True, bypass caches for all markets.

    Returns:
        Dict mapping market code to its AgentToolResult.
    """
    return _run_parallel(run_signal_agent, markets, max_workers, force)


# ---------------------------------------------------------------------------
# Global agent tool
# ---------------------------------------------------------------------------

def run_global_agent(force: bool = False) -> AgentToolResult:
    """Run the GlobalAgent to produce cross-market analysis.

    The GlobalAgent merges per-market universes, fetches macro indicators
    (FRED, World Bank, CN macro, commodities/FX), computes sector performance
    and index correlations, and generates the global alert feed. Writes all
    output under data/global/. Should be called after all market agents complete.

    Args:
        force: If True, bypass any cache checks in the GlobalAgent.

    Returns:
        AgentToolResult with success status, total records, duration,
        and any error messages. The market field is None.
    """
    try:
        from src.agents.global_agent import GlobalAgent
        from src.common.config import load_yaml
        markets_cfg = load_yaml("markets.yaml")["markets"]
        agent = GlobalAgent(name="global", depends_on=[])
        agent.force = force
        result = agent.execute()
        return AgentToolResult.from_agent_result(None, "global", result)
    except Exception as e:
        return AgentToolResult.failure(None, "global", str(e))


# ---------------------------------------------------------------------------
# Full pipeline tool
# ---------------------------------------------------------------------------

def run_full_pipeline(
    markets: list[str] | None = None,
    max_workers: int = 6,
    force: bool = False,
) -> dict[str, AgentToolResult]:
    """Run the complete pipeline: all market agents then the global agent.

    Executes in dependency order:
      1. All DataAgents in parallel
      2. All NewsAgents + SignalAgents in parallel (after their market's DataAgent)
      3. GlobalAgent last

    Args:
        markets:     Run only these markets (plus global). None = all 13.
        max_workers: Maximum concurrent market agents. Defaults to 6.
        force:       If True, bypass all caches across the entire pipeline.

    Returns:
        Dict mapping agent name (e.g. "US_data", "CN_signal", "global")
        to its AgentToolResult.
    """
    from src.common.config import load_yaml
    if markets is None:
        markets = list(load_yaml("markets.yaml")["markets"].keys())

    all_results: dict[str, AgentToolResult] = {}

    # Tier 1: data agents in parallel
    data_results = _run_parallel(run_data_agent, markets, max_workers, force)
    for market, r in data_results.items():
        all_results[f"{market}_data"] = r

    # Tier 2: news + signal in parallel (only for markets where data succeeded)
    successful_markets = [m for m, r in data_results.items() if r.success]
    failed_markets = [m for m in markets if m not in successful_markets]

    for m in failed_markets:
        all_results[f"{m}_news"] = AgentToolResult.failure(m, "news", "Skipped: data agent failed")
        all_results[f"{m}_signal"] = AgentToolResult.failure(m, "signal", "Skipped: data agent failed")

    if successful_markets:
        news_results = _run_parallel(run_news_agent, successful_markets, max_workers, force)
        signal_results = _run_parallel(run_signal_agent, successful_markets, max_workers, force)
        for market, r in news_results.items():
            all_results[f"{market}_news"] = r
        for market, r in signal_results.items():
            all_results[f"{market}_signal"] = r

    # Tier 3: global agent
    all_results["global"] = run_global_agent(force=force)

    return all_results


# ---------------------------------------------------------------------------
# Status and introspection tools
# ---------------------------------------------------------------------------

def get_agent_status(agent_name: str | None = None) -> dict:
    """Return the runtime status of one or all pipeline agents.

    Reads the shared agent_status.json file written during pipeline runs.

    Args:
        agent_name: Specific agent name (e.g. "US_data", "CN_signal", "global").
                    None = return all agents.

    Returns:
        Dict of agent name → status dict (keys: state, progress, last_run,
        duration_s, records, errors). Empty dict if file not found.
    """
    import json
    from src.common.config import get_data_path
    try:
        path = get_data_path("agent_status.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        agents = data.get("agents", {})
        if agent_name:
            return {agent_name: agents.get(agent_name, {})}
        return agents
    except Exception:
        return {}


def list_available_agents() -> list[str]:
    """Return the names of all agents registered in the pipeline.

    Returns:
        List of agent name strings: "{MARKET}_{type}" for market agents
        and "global" for the global agent. Example:
        ["CN_data", "CN_news", "CN_signal", "US_data", ..., "global"].
    """
    from src.common.config import load_yaml
    try:
        markets = list(load_yaml("markets.yaml")["markets"].keys())
        names = []
        for m in markets:
            names.extend([f"{m}_data", f"{m}_news", f"{m}_signal"])
        names.append("global")
        return names
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _all_markets() -> list[str]:
    from src.common.config import load_yaml
    try:
        return list(load_yaml("markets.yaml")["markets"].keys())
    except Exception:
        return []


def _run_parallel(fn, markets, max_workers, force) -> dict[str, AgentToolResult]:
    """Run fn(market, force) for each market in parallel."""
    if markets is None:
        markets = _all_markets()
    results: dict[str, AgentToolResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fn, m, force): m for m in markets}
        for future in as_completed(futures):
            market = futures[future]
            try:
                results[market] = future.result()
            except Exception as e:
                agent_type = fn.__name__.replace("run_", "").replace("_agent", "")
                results[market] = AgentToolResult.failure(market, agent_type, str(e))
    return results
