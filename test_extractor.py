"""Step-by-step test: web_search → extractor for 中科曙光 peer query."""
import sys, json, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "web"))

from agent_llm import _web_search, _extract_financials

# Load real API key
with open("data/llm_config.json") as f:
    cfg = json.load(f)
API_KEY = cfg["api_key"]
PROVIDER = "openai"

MESSAGE = "中科曙光 同类型的A股公司 2025 2026 营收表现"
PEERS = ["浪潮信息", "紫光股份", "华鲲振宇", "中国长城", "同方股份"]

# ── Step 1: web search ──────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: web_search")
print("=" * 60)
query = "浪潮信息 紫光股份 中科曙光 服务器算力 2024 2025 营收净利润"
ws = _web_search(query, max_results=6, body_chars=600, region="cn-zh")
print(ws[:2000])

# ── Step 2: extractor ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: extractor")
print("=" * 60)
tool_results = {"web_search": ws}
result = _extract_financials(
    tool_results=tool_results,
    companies_hint=PEERS,
    message=MESSAGE,
    provider=PROVIDER,
    api_key=API_KEY,
)
print(result if result else "(empty)")
