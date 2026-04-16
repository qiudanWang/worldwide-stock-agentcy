"""Full pipeline debug for: 中科曙光 peer comparison query."""
import sys, json, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "web"))

from agent_llm import (
    _plan_message, _execute_plan, _cn_peer_financials,
    _extract_financials, _web_search, _avail_tokens,
    _call_openai, SYSTEM_PROMPTS
)

with open("data/llm_config.json") as f:
    cfg = json.load(f)
API_KEY = cfg["api_key"]
PROVIDER = "openai"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

MESSAGE = (
    "https://finance.eastmoney.com/a/202603033660678118.html\n"
    "read this article. If 中科曙光 2024年，中科曙光业绩承压：营收为131.48亿元，同比下滑8.4%, "
    "那么应该有其他提供算力的公司业绩有所增长？你就看下最新的数据就好。"
    "和中科曙光 同类型的公司A股有哪些。他们最近一年，我指的2025，2026 的收益如何"
)

SEP = "=" * 60

# ── Step 1: Planner ──────────────────────────────────────────────────────────
print(f"\n{SEP}\nSTEP 1: PLANNER\n{SEP}")
plan = _plan_message(MESSAGE, "CN", "market", ["https://finance.eastmoney.com/a/202603033660678118.html"], PROVIDER, API_KEY)
print(json.dumps(plan, ensure_ascii=False, indent=2))

intent        = plan.get("intent", "peer_comparison")
tickers       = plan.get("tickers", ["603019"])
company_name  = plan.get("company_name", "中科曙光")
steps         = plan.get("steps", [])
response_focus = plan.get("response_focus", "")

# ── Step 2: Execute plan ─────────────────────────────────────────────────────
print(f"\n{SEP}\nSTEP 2: EXECUTE PLAN\n{SEP}")
tool_results = _execute_plan(plan, tickers, "CN", "market", DATA_DIR, MESSAGE)
for k, v in tool_results.items():
    print(f"\n-- {k} ({len(v)} chars) --")
    print(v[:800])

# ── Step 3: cn_peer_financials ───────────────────────────────────────────────
print(f"\n{SEP}\nSTEP 3: CN PEER FINANCIALS (akshare)\n{SEP}")
import re, pandas as pd
_is_cn_peer = intent == "peer_comparison" and tickers and tickers[0].isdigit()
if _is_cn_peer:
    anchor = tickers[0]
    try:
        uni = pd.read_parquet(f"{DATA_DIR}/markets/CN/universe.parquet")
        ws = tool_results.get("web_search", "")
        ws_tks = list(dict.fromkeys(
            t for t in re.findall(r"(?<![/\d])([0369]\d{5})(?!\d)", ws)
            if t != anchor
        ))[:5]
        COMPUTE_KW = ["信息", "服务器", "算力", "计算", "数创", "长城", "同方", "浪潮"]
        anchor_row = uni[uni["ticker"].astype(str) == anchor]
        sector_peers = []
        if not anchor_row.empty and "sector" in uni.columns:
            sector = anchor_row.iloc[0]["sector"]
            sector_df = uni[(uni["sector"] == sector) & (uni["ticker"].astype(str) != anchor)]
            relevant = sector_df[sector_df["name"].apply(lambda n: any(k in str(n) for k in COMPUTE_KW))]
            sector_peers = relevant["ticker"].astype(str).tolist()[:4]
        peer_tks = list(dict.fromkeys(ws_tks + sector_peers))[:6]
        print(f"Peers selected: {peer_tks}")
        ak_result = _cn_peer_financials(peer_tks, anchor, DATA_DIR)
        if ak_result:
            tool_results["cn_peer_financials"] = ak_result
            print(ak_result)
        else:
            print("(empty)")
    except Exception as e:
        print(f"ERROR: {e}")

# ── Step 4: Extractor ────────────────────────────────────────────────────────
print(f"\n{SEP}\nSTEP 4: EXTRACTOR\n{SEP}")
if tool_results.get("web_search") or tool_results.get("fetch_url"):
    companies_hint = [company_name] if company_name else []
    ws = tool_results.get("web_search", "")
    found_cn = list(dict.fromkeys(re.findall(r'[\u4e00-\u9fff]{2,6}(?:信息|科技|股份|系统|数据|网络|通信|电子|智能)?', ws)))
    companies_hint += [n for n in found_cn[:6] if n not in companies_hint]
    extracted = _extract_financials(tool_results, companies_hint, MESSAGE, PROVIDER, API_KEY)
    if extracted:
        tool_results["extracted_financials"] = extracted
    print(extracted or "(empty)")

# ── Step 5: Synthesis ────────────────────────────────────────────────────────
print(f"\n{SEP}\nSTEP 5: SYNTHESIS\n{SEP}")
WEB_TOOLS = {"web_search", "fetch_url", "local_data_from_web"}
priority = ["cn_peer_financials", "extracted_financials"]
ordered = {k: tool_results[k] for k in priority if k in tool_results}
for k, v in tool_results.items():
    if k not in priority:
        ordered[k] = v

context_parts = []
for tool_name, result in ordered.items():
    limit = 3000 if tool_name == "cn_peer_financials" else 2000 if tool_name == "extracted_financials" else 1000
    if len(result) > limit:
        result = result[:limit] + "\n...(truncated)"
    context_parts.append(f"=== {tool_name.upper()} ===\n{result}")
combined_context = "\n\n".join(context_parts)

base_system = SYSTEM_PROMPTS["market"].format(market="CN")
focus_block = f"\n\nTask: {response_focus}" if response_focus else ""
full_system = f"{base_system}{focus_block}\n\n--- DATA ---\n{combined_context}"

# Simulate 2 turns of chat history (real app caps at 2 turns when context > 4000 chars)
fake_history = [
    {"role": "user", "content": "中科曙光最近的消息"},
    {"role": "assistant", "content": "中科曙光(603019)近期股价承压，管理层拟减持，2024年营收131.48亿(-8.4%)，净利润19.11亿(+4.1%)。市场关注算力需求分化。"},
]
hist_msgs = [{"role": ("assistant" if m["role"]=="agent" else m["role"]), "content": m["content"][:600]} for m in fake_history]

avail = _avail_tokens(full_system, hist_msgs, MESSAGE)
print(f"[synthesis] system chars={len(full_system)}, history turns={len(hist_msgs)}, avail_tokens={avail}")

resp = _call_openai(API_KEY, full_system, hist_msgs, MESSAGE, max_tokens=avail)
print("\n--- FINAL RESPONSE ---")
print(resp)
