"""LiteLLM-based agent loop — replaces agent_llm.py.

Uses native tool_use (function calling) so the LLM drives which tools to call
and when to stop, eliminating the hand-rolled JSON planner and all its edge cases.

Drop-in replacement: same signature as agent_llm.agent_chat.
"""

import json
import os

import litellm

from web.agent_tools import TOOL_SCHEMAS, execute_tool

litellm.suppress_debug_info = True
litellm.drop_params = True
litellm.set_verbose = False

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a professional global stock market analyst assistant.

Rules:
- ALWAYS use tools to get data before answering. Never invent prices, returns, or financial figures.
- RESPOND IN ENGLISH ONLY. All text — section headers, bullet points, table headers, labels — must be in English.
- Format numbers consistently: prices to 2 decimal places, returns as percentages (+4.85%), large figures with units (B/M/億).
- If a tool returns no data, say so clearly rather than guessing.
- For peer comparisons involving multiple stocks, call get_stock_data for each ticker.
- After responses that include web_search results, add a brief "## News Highlights" section (3–5 bullets).
- Be concise. Lead with the key finding, then supporting data in a table or bullets.
"""


# ---------------------------------------------------------------------------
# Config helpers
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


def _make_openai_client():
    """Build an OpenAI-compatible client from llm_config.json.

    Uses httpx.Client(verify=False) to match the existing codebase's SSL setup.
    Returns (client, model_name).
    """
    import httpx
    from openai import OpenAI

    cfg      = _load_config()
    api_key  = cfg.get("api_key", "sk-placeholder").strip() or "sk-placeholder"
    model    = cfg.get("model", "gpt-4o-mini").strip() or "gpt-4o-mini"
    base_url = (cfg.get("base_url", "") or "https://api.openai.com/v1").strip()

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(verify=False),
    )
    return client, model


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def agent_chat(
    agent_type: str,
    market: str | None,
    message: str,
    data_dir: str,
    chat_history: list | None = None,
    language: str = "English",
    context: dict | None = None,
) -> tuple[str, dict]:
    """Run the agent loop and return (response_text, updated_context).

    The LLM decides which tools to call and when it has enough data to answer.
    No JSON planner, no intent classification — just tool_use in a while loop.
    """
    context = dict(context or {})
    if market:
        context["market"] = market

    oai_client, model = _make_openai_client()

    system = SYSTEM_PROMPT
    if market:
        system += f"\nCurrent market context: {market} market."

    messages = list(chat_history or []) + [{"role": "user", "content": message}]

    max_iterations = 10  # safety cap — prevents infinite loops
    used_web_search = False

    for _ in range(max_iterations):
        response = oai_client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}] + messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            max_completion_tokens=4096,
        )

        choice  = response.choices[0]
        message_obj = choice.message

        # ── Done: LLM produced a final answer ──────────────────────────────
        if choice.finish_reason == "stop":
            text = message_obj.content or ""
            return text, context

        # ── Tool calls requested ────────────────────────────────────────────
        if choice.finish_reason == "tool_calls":
            tool_calls = message_obj.tool_calls or []

            # Append assistant turn (with tool_calls) to history
            messages.append({
                "role":       "assistant",
                "content":    message_obj.content,
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            # Execute each tool and append result
            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    tool_inputs = json.loads(tc.function.arguments)
                except Exception:
                    tool_inputs = {}

                if tool_name == "web_search":
                    used_web_search = True

                result = execute_tool(tool_name, tool_inputs, context, data_dir)

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      str(result),
                })

            continue  # back to top of loop — ask LLM again with tool results

        # Unexpected finish_reason — return whatever we have
        text = message_obj.content or "Unable to complete the analysis."
        return text, context

    return "Reached maximum tool iterations without a final answer. Please try a more specific question.", context
