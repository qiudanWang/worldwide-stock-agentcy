"""Orchestrator: dependency-aware agent runner with parallel execution."""

import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.common.config import load_yaml
from src.common.logger import get_logger
from src.agents.base import AgentResult, update_pipeline_status

log = get_logger("orchestrator")


class Orchestrator:
    """Runs agents respecting dependency order, with parallel execution within tiers."""

    def __init__(self):
        self.agents = {}
        self.results = {}

    def register(self, agent):
        """Register an agent."""
        self.agents[agent.name] = agent

    def load_all_agents(self):
        """Load all 40 agents (13 markets x 3 + 1 global)."""
        from src.agents.data_agent import DataAgent
        from src.agents.news_agent import NewsAgent
        from src.agents.signal_agent import SignalAgent
        from src.agents.global_agent import GlobalAgent

        markets_cfg = load_yaml("markets.yaml")["markets"]

        market_agent_names = []
        for market_code, market_cfg in markets_cfg.items():
            # Data Agent
            data_name = f"{market_code}_data"
            self.register(DataAgent(
                name=data_name,
                market=market_code,
                market_config=market_cfg,
            ))

            # News Agent (depends on data)
            news_name = f"{market_code}_news"
            self.register(NewsAgent(
                name=news_name,
                market=market_code,
                market_config=market_cfg,
                depends_on=[data_name],
            ))

            # Signal Agent (depends on data + news)
            signal_name = f"{market_code}_signal"
            self.register(SignalAgent(
                name=signal_name,
                market=market_code,
                market_config=market_cfg,
                depends_on=[data_name, news_name],
            ))

            market_agent_names.extend([data_name, news_name, signal_name])

        # Global Agent (depends on all market agents)
        self.register(GlobalAgent(
            name="global",
            depends_on=market_agent_names,
        ))

        log.info(f"Loaded {len(self.agents)} agents")

    def _get_execution_tiers(self, agent_names=None):
        """Topological sort agents into parallel execution tiers."""
        if agent_names is None:
            agent_names = set(self.agents.keys())
        else:
            agent_names = set(agent_names)

        remaining = {n: set(self.agents[n].depends_on) & agent_names
                     for n in agent_names}
        tiers = []

        while remaining:
            # Find agents with all dependencies satisfied
            ready = {n for n, deps in remaining.items() if not deps}
            if not ready:
                log.error(f"Circular dependency detected: {remaining}")
                break

            tiers.append(sorted(ready))
            for n in ready:
                del remaining[n]
            for deps in remaining.values():
                deps -= ready

        return tiers

    def set_force(self, force: bool):
        """Enable or disable force mode (bypass all caches) for all agents."""
        for agent in self.agents.values():
            agent.force = force

    def run_all(self, max_workers=6):
        """Run all agents in dependency order."""
        return self._run_agents(list(self.agents.keys()), max_workers)

    def run_agent(self, name: str):
        """Run a single agent (ignoring dependencies)."""
        if name not in self.agents:
            log.error(f"Agent not found: {name}")
            return {}
        update_pipeline_status(
            state="running",
            started=datetime.now().isoformat(),
            completed_agents=0,
            total_agents=1,
        )
        agent = self.agents[name]
        result = agent.execute()
        self.results[name] = result
        update_pipeline_status(
            state="completed",
            completed_agents=1,
            total_agents=1,
        )
        return {name: result}

    def run_market(self, market: str, max_workers=3):
        """Run all agents for a specific market."""
        names = [n for n, a in self.agents.items()
                 if a.market == market or n == "global"]
        # Don't run global agent in market-specific mode
        names = [n for n in names if n != "global"]
        return self._run_agents(names, max_workers)

    def run_agent_type(self, agent_type: str, max_workers=6):
        """Run all agents of a given type (e.g., 'data', 'news', 'signal')."""
        names = [n for n, a in self.agents.items()
                 if a.agent_type == agent_type]
        return self._run_agents(names, max_workers)

    def _run_agents(self, agent_names, max_workers=6):
        """Execute a set of agents respecting dependencies."""
        tiers = self._get_execution_tiers(agent_names)
        total = len(agent_names)
        completed_count = 0

        update_pipeline_status(
            state="running",
            started=datetime.now().isoformat(),
            completed_agents=0,
            total_agents=total,
        )

        start = time.time()
        log.info(f"Starting pipeline: {total} agents in {len(tiers)} tiers")

        for tier_idx, tier in enumerate(tiers):
            log.info(f"Tier {tier_idx + 1}/{len(tiers)}: {tier}")

            # Set waiting status for agents in this tier
            for name in tier:
                deps = self.agents[name].depends_on
                if deps:
                    dep_str = ", ".join(deps)
                    self.agents[name].update_status(
                        state="waiting",
                        progress=f"Waiting for {dep_str}",
                    )

            # Execute tier in parallel — stagger starts by 2s to reduce yfinance rate limiting
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for i, name in enumerate(tier):
                    if i > 0:
                        time.sleep(2)
                    future = executor.submit(self.agents[name].execute)
                    futures[future] = name

                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        result = future.result()
                        self.results[name] = result
                    except Exception as e:
                        log.error(f"Agent {name} raised: {e}")
                        self.results[name] = AgentResult(
                            success=False, errors=[str(e)]
                        )

                    completed_count += 1
                    update_pipeline_status(
                        completed_agents=completed_count,
                    )

        duration = time.time() - start
        failed = [n for n, r in self.results.items() if not r.success]

        update_pipeline_status(
            state="completed" if not failed else "completed_with_errors",
            completed=datetime.now().isoformat(),
            duration_s=round(duration, 1),
            completed_agents=completed_count,
            failed_agents=failed,
        )

        log.info(
            f"Pipeline complete: {completed_count}/{total} agents, "
            f"{len(failed)} failed, {duration:.1f}s total"
        )

        return self.results

    def get_status(self) -> dict:
        """Return status of all registered agents."""
        return {name: agent.status() for name, agent in self.agents.items()}
