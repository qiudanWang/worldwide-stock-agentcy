"""Agent base class and result type for the multi-agent pipeline."""

import json
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from src.common.config import get_data_path
from src.common.logger import get_logger

_status_lock = threading.Lock()

log = get_logger("agents.base")


@dataclass
class AgentResult:
    """Result of an agent run."""
    success: bool
    records_written: int = 0
    duration_seconds: float = 0.0
    errors: list = field(default_factory=list)


class BaseAgent:
    """Base class for all pipeline agents.

    Each agent has a name, market scope, dependencies, and a run() method.
    The orchestrator uses depends_on to determine execution order.
    """

    name: str = "base"
    agent_type: str = "base"  # data, news, signal, global
    market: str = None  # None for global agent
    depends_on: list = []
    schedule: str = "daily"

    def __init__(self, name: str, agent_type: str, market: str = None,
                 depends_on: list = None, force: bool = False):
        self.name = name
        self.agent_type = agent_type
        self.market = market
        self.depends_on = depends_on or []
        self.force = force  # if True, bypass all cache checks
        self._status = {
            "state": "scheduled",
            "progress": "",
            "last_run": None,
            "duration_s": 0,
            "records": 0,
            "errors": [],
        }

    def should_run(self) -> bool:
        """Check if the agent should run (not cached, etc)."""
        return True

    def run(self) -> AgentResult:
        """Execute the agent. Subclasses must override."""
        raise NotImplementedError

    def validate(self) -> bool:
        """Verify output integrity after run."""
        return True

    def status(self) -> dict:
        """Return current status dict."""
        return self._status.copy()

    def update_status(self, **kwargs):
        """Update status fields and write to shared status file."""
        self._status.update(kwargs)
        _update_agent_status(self.name, self._status)

    def execute(self, max_retries: int = 2, retry_delay: int = 30) -> AgentResult:
        """Full execution wrapper: status tracking, timing, error handling, and retry."""
        self.update_status(state="running", progress="Starting...")
        start = time.time()

        try:
            if not self.force and not self.should_run():
                duration = time.time() - start
                self.update_status(
                    state="completed",
                    progress="Skipped (cached)",
                    duration_s=round(duration, 1),
                )
                log.info(f"[{self.name}] Skipped (cached)")
                return AgentResult(success=True, duration_seconds=duration)

            result = None
            for attempt in range(1, max_retries + 2):  # attempts: 1, 2, 3
                if attempt > 1:
                    wait = retry_delay * (attempt - 1)  # 30s, 60s
                    log.warning(f"[{self.name}] Retry {attempt - 1}/{max_retries} in {wait}s...")
                    self.update_status(
                        state="running",
                        progress=f"Retry {attempt - 1}/{max_retries} (waiting {wait}s)",
                    )
                    time.sleep(wait)

                try:
                    result = self.run()
                    if result.success:
                        break  # success — no retry needed
                    log.warning(f"[{self.name}] Attempt {attempt} failed: {result.errors}")
                except Exception as inner_e:
                    result = AgentResult(success=False, errors=[str(inner_e)])
                    log.warning(f"[{self.name}] Attempt {attempt} raised: {inner_e}")

            duration = time.time() - start
            result.duration_seconds = round(duration, 1)

            if result.success:
                self.update_status(
                    state="completed",
                    progress=f"{result.records_written} records",
                    duration_s=result.duration_seconds,
                    records=result.records_written,
                    last_run=datetime.now().isoformat(),
                )
                log.info(
                    f"[{self.name}] Completed: {result.records_written} records "
                    f"in {result.duration_seconds:.1f}s"
                )
            else:
                self.update_status(
                    state="failed",
                    progress=f"Failed after {max_retries + 1} attempts: {'; '.join(result.errors)}",
                    duration_s=result.duration_seconds,
                    errors=result.errors,
                )
                log.error(f"[{self.name}] Failed after {max_retries + 1} attempts: {result.errors}")

            return result

        except Exception as e:
            duration = time.time() - start
            self.update_status(
                state="failed",
                progress=f"Error: {str(e)}",
                duration_s=round(duration, 1),
                errors=[str(e)],
            )
            log.error(f"[{self.name}] Exception: {e}")
            return AgentResult(
                success=False,
                duration_seconds=round(duration, 1),
                errors=[str(e)],
            )


def _update_agent_status(agent_name: str, status: dict):
    """Update a single agent's entry in the shared agent_status.json file."""
    status_path = get_data_path("agent_status.json")
    with _status_lock:
        try:
            with open(status_path, "r") as f:
                all_status = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_status = {"agents": {}, "pipeline": {}}

        all_status["agents"][agent_name] = status

        with open(status_path, "w") as f:
            json.dump(all_status, f, indent=2, default=str)


def update_pipeline_status(**kwargs):
    """Update the pipeline-level status in agent_status.json."""
    status_path = get_data_path("agent_status.json")
    with _status_lock:
        try:
            with open(status_path, "r") as f:
                all_status = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_status = {"agents": {}, "pipeline": {}}

        all_status["pipeline"].update(kwargs)

        with open(status_path, "w") as f:
            json.dump(all_status, f, indent=2, default=str)
