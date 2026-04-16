"""
Step-by-step pipeline test for capital flow question:
  "看起来各个国家的Foreign Capital Flow 都在降低，钱去哪里了？"

Verifies each stage: planner → local data → web search → synthesis.
"""
import sys, json, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "web"))

from agent_llm import (
    _plan_message, _execute_plan, _load_global_context,
    _build_macro_search_query, _web_search, _call_openai,
    SYSTEM_PROMPTS,
)

with open("data/llm_config.json") as f:
    cfg = json.load(f)
API_KEY  = cfg["api_key"]
PROVIDER = "openai"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

MESSAGE    = "看起来各个国家的Foreign Capital Flow 都在降低，钱去哪里了？"
MARKET     = None        # global agent
AGENT_TYPE = "global"

SEP = "=" * 60


# ── Step 1: Macro search query mapping ──────────────────────────────────────
print(f"\n{SEP}\nSTEP 1: _build_macro_search_query\n{SEP}")
query = _build_macro_search_query(MESSAGE, MARKET)
print(f"Mapped query: {query!r}")
assert query, "Query must not be empty"


# ── Step 2: Local global context (capital flows) ────────────────────────────
print(f"\n{SEP}\nSTEP 2: LOCAL GLOBAL CONTEXT (capital flows first)\n{SEP}")
local_ctx = _load_global_context(DATA_DIR)
if not local_ctx or local_ctx.startswith("No global data"):
    print("WARNING: No global data — pipeline will be web-search only")
else:
    # Check capital flow section is present and populated
    cf_present = "CAPITAL FLOWS" in local_ctx
    print(f"Capital flow section present: {cf_present}")
    # Print just the capital flow section
    lines = local_ctx.split("\n")
    in_cf = False
    for line in lines:
        if "CAPITAL FLOWS" in line:
            in_cf = True
        if in_cf:
            print(line)
        if in_cf and line == "" and "CAPITAL FLOWS" not in line:
            # end of section after first blank line
            break


# ── Step 3: Planner ──────────────────────────────────────────────────────────
print(f"\n{SEP}\nSTEP 3: PLANNER\n{SEP}")
plan = _plan_message(MESSAGE, MARKET, AGENT_TYPE, PROVIDER, API_KEY)
print(json.dumps(plan, ensure_ascii=False, indent=2))

intent         = plan.get("intent", "")
tickers        = plan.get("tickers", [])
steps          = plan.get("steps", [])
response_focus = plan.get("response_focus", "")

print(f"\nIntent:  {intent}")
print(f"Tickers: {tickers}")
print(f"Tools:   {[s.get('tool') for s in steps]}")
print(f"Focus:   {response_focus}")

# Assertions
assert intent in ("macro", "general", "market_overview"), \
    f"Expected macro/general/market_overview intent, got {intent!r}"
assert not tickers, f"No tickers expected for this question, got {tickers}"
tools_chosen = [s.get("tool") for s in steps]
assert any(t in ("web_search", "local_data") for t in tools_chosen), \
    f"Expected web_search or local_data in plan, got {tools_chosen}"


# ── Step 4: Execute plan ─────────────────────────────────────────────────────
print(f"\n{SEP}\nSTEP 4: EXECUTE PLAN\n{SEP}")
tool_results = _execute_plan(plan, tickers, MARKET, AGENT_TYPE, DATA_DIR, MESSAGE)
for k, v in tool_results.items():
    print(f"\n-- {k} ({len(v)} chars) --")
    print(v[:600])


# ── Step 5: Check web search content quality ─────────────────────────────────
print(f"\n{SEP}\nSTEP 5: WEB SEARCH CONTENT CHECK\n{SEP}")
ws = tool_results.get("web_search", "")
if ws:
    # Check for signal words that indicate relevant content
    SIGNALS = ["gold", "treasury", "safe haven", "USD", "dollar", "bond",
               "outflow", "inflow", "capital", "emerging", "流出", "流入",
               "避险", "美元", "黄金", "国债"]
    found = [s for s in SIGNALS if s.lower() in ws.lower()]
    print(f"Relevant signal words found in web search: {found}")
    if len(found) < 2:
        print("WARNING: Web search result may lack relevant financial content")
    else:
        print("OK: Web search has relevant content")
else:
    print("WARNING: No web_search result in tool_results")


# ── Step 6: Check local_data capital flow coverage ───────────────────────────
print(f"\n{SEP}\nSTEP 6: LOCAL DATA CAPITAL FLOW COVERAGE\n{SEP}")
ld = tool_results.get("local_data", "")
if ld:
    cf_present = "CAPITAL FLOWS" in ld or "capital_flow" in ld.lower()
    print(f"Capital flow in local_data: {cf_present}")
    markets_with_flow = []
    import re
    for m in ["CN", "HK", "JP", "US", "KR", "TW", "IN", "UK", "DE", "FR", "AU", "BR"]:
        if f"  {m}:" in ld:
            markets_with_flow.append(m)
    print(f"Markets with flow data: {markets_with_flow}")
    if len(markets_with_flow) < 2:
        print("WARNING: Few markets have capital flow data — answer may be incomplete")
    else:
        print(f"OK: {len(markets_with_flow)} markets have flow data")
else:
    print("NOTE: local_data not in tool_results (may be loaded directly in synthesis)")


# ── Step 7: Synthesis ────────────────────────────────────────────────────────
print(f"\n{SEP}\nSTEP 7: SYNTHESIS\n{SEP}")

# Merge local context if not already in tool_results (global agent loads it separately)
if "local_data" not in tool_results:
    tool_results["local_data"] = local_ctx

context_parts = []
for tool_name, result in tool_results.items():
    label = tool_name.replace("_", " ").upper()
    context_parts.append(f"=== {label} ===\n{result}")
combined_context = "\n\n".join(context_parts)

base_system  = SYSTEM_PROMPTS["global"]
focus_block  = f"\n\nTask: {response_focus}" if response_focus else ""

data_notes = []
if tool_results.get("web_search"):
    data_notes.append("Web search has already been performed — results are in the DATA section below.")
data_notes_block = "\n".join(data_notes) + "\n\n" if data_notes else ""

full_system = (
    f"{data_notes_block}"
    f"{base_system}{focus_block}\n\n"
    f"--- DATA ---\n{combined_context}"
)

print(f"[synthesis] system_chars={len(full_system)}")

history_msgs = []   # fresh conversation, no history
# No max_tokens for synthesis — let the model decide (mirrors real agent_chat)
resp = _call_openai(API_KEY, full_system, history_msgs, MESSAGE)
print("\n--- FINAL RESPONSE ---")
print(resp)


# ── Step 8: Response quality check ──────────────────────────────────────────
print(f"\n{SEP}\nSTEP 8: RESPONSE QUALITY CHECK\n{SEP}")
if resp:
    QUALITY_SIGNALS = [
        "gold",       "黄金",
        "treasury",   "国债",
        "safe haven", "避险",
        "dollar",     "美元",
        "outflow",    "流出",
        "where",      "去向",
    ]
    found_q = [s for s in QUALITY_SIGNALS if s.lower() in resp.lower()]
    print(f"Quality signals in response: {found_q}")
    if len(found_q) >= 3:
        print("PASS: Response addresses 'where did the money go' with sufficient detail")
    else:
        print("FAIL: Response may not be answering the core question well")
    print(f"\nResponse length: {len(resp)} chars")
else:
    print("FAIL: Empty response")
