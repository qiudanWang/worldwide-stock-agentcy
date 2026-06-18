"""openai-agents SDK based agent — replaces the hand-rolled LiteLLM loop.

Uses Agent + Runner with SQLiteSession for persistent conversation history.
Config is read fresh on every request so web UI changes take effect immediately.
Supports any OpenAI-compatible endpoint via base_url in llm_config.json.
"""

import json
import os

from agents import Agent, Runner, WebSearchTool
from agents.memory.sqlite_session import SQLiteSession
from agents.memory.session_settings import SessionSettings

from web.agent_tools import ALL_TOOLS
from src.common.tracing import observe
try:
    from traceroot import update_current_span
except ImportError:
    def update_current_span(**kwargs): pass

_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"

_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "agent_sessions.db",
)

# ---------------------------------------------------------------------------
# Config (read fresh each call so UI changes take effect without restart)
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    try:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "llm_config.json",
        )
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _resolve_model(cfg: dict):
    """Return a model string (OpenAI native) or OpenAIChatCompletionsModel (custom endpoint)."""
    model_name = cfg.get("model", "gpt-4o-mini").strip()
    api_key    = cfg.get("api_key", "").strip()
    base_url   = (cfg.get("base_url", "") or _OPENAI_DEFAULT_BASE_URL).strip()

    is_official_openai = base_url.rstrip("/") == _OPENAI_DEFAULT_BASE_URL.rstrip("/")

    if is_official_openai:
        # SDK handles auth via OPENAI_API_KEY env var
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        return model_name
    else:
        # Custom endpoint — wrap with OpenAIChatCompletionsModel
        from openai import AsyncOpenAI
        from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
        import httpx

        client = AsyncOpenAI(
            api_key=api_key or "sk-placeholder",
            base_url=base_url,
            http_client=httpx.AsyncClient(verify=False),
        )
        return OpenAIChatCompletionsModel(model=model_name, openai_client=client)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a professional global stock market analyst assistant.

Rules:
- ALWAYS use tools to get data before answering. Never invent prices, returns, or financial figures.
- RESPOND IN ENGLISH ONLY. All text — section headers, bullet points, table headers, labels — must be in English.
- Format numbers consistently: prices to 2 decimal places, returns as percentages (+4.85%), large figures with units (B/M/億).
- For peer comparisons or multi-stock queries, call get_stock_data for EACH ticker separately.
- If get_stock_data returns no data for a CN ticker, IMMEDIATELY use web search with the 6-digit ticker code as the query (e.g. "603986"). Do NOT ask the user for permission — just search automatically.
- After responses that include web search results, add a brief "## News Highlights" section (3–5 bullets).
- Be concise. Lead with the key finding, then supporting data in a table or bullets.
- You have a web search tool available. NEVER say you cannot search the web or access real-time information — call it immediately whenever you need current news, prices, or data you don't have.
- When the user asks which sector rose/fell the most, best/worst sector, or sector ranking for ANY market, call get_sector_ranking immediately — do NOT ask for ticker lists or sector names first.
- When the user asks about a specific stock (how is X, analyze X, what do you think about X), call deep_analysis as the primary tool.
"""

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@observe(name="agent_chat", type="llm")
def agent_chat(
    agent_type: str,
    market: str | None,
    message: str,
    data_dir: str,
    language: str = "English",
    context: dict | None = None,
    session_id: str | None = None,
) -> tuple[str, dict]:
    """Run the agent and return (response_text, updated_context)."""
    import asyncio

    context = dict(context or {})
    if market:
        context["market"] = market

    cfg   = _load_config()
    model = _resolve_model(cfg)
    session_limit = cfg.get("session_limit", 20)

    update_current_span(
        model=cfg.get("model", ""),
        input={"message": message, "market": market, "agent_type": agent_type},
    )

    system = SYSTEM_PROMPT
    if market:
        system += f"\nCurrent market context: {market} market."

    agent = Agent(
        name="stock-analyst",
        model=model,
        instructions=system,
        tools=[WebSearchTool(), *ALL_TOOLS],
    )

    session = SQLiteSession(
        session_id=session_id or "default",
        db_path=_DB_PATH,
        session_settings=SessionSettings(limit=session_limit),
    )

    async def _run():
        result = await Runner.run(agent, input=message, session=session)
        return result.final_output

    try:
        import contextvars
        ctx = contextvars.copy_context()
        response = ctx.run(asyncio.run, _run())
    except Exception as e:
        response = f"Agent error: {e}"

    update_current_span(output={"response": response[:1000]})
    return response, context
