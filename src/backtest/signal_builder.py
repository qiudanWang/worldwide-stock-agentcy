"""
Signal Builder
==============
Converts natural language stock selection description -> Python select() function
using an LLM, with a self-reflection loop that catches bugs before returning.

The generated function signature:
    def select(universe_df: pd.DataFrame, history: dict, date) -> list[str]:
        ...
"""

from __future__ import annotations

import re
import threading
from typing import Callable, Optional

import pandas as pd
import numpy as np

from src.common.logger import get_logger

log = get_logger("backtest.signal_builder")

MAX_REFLECTION_ATTEMPTS = 3

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_GENERATOR = """You are a quantitative analyst writing Python stock selection functions.

The function must have this EXACT signature:
def select(universe_df, history, date):
    '''
    universe_df: pd.DataFrame with columns: ticker, name, sector, subsector
    history: dict mapping ticker -> pd.DataFrame with columns: date, open, high, low, close, volume
             All data is up to and including `date`
    date: pd.Timestamp signal date
    Returns: list of ticker strings to hold (empty list if none qualify)
    '''

Available imports (already in scope): pandas as pd, numpy as np
Rules:
- Handle missing tickers in history gracefully (use .get())
- Return at most 20 tickers
- Return empty list if data insufficient
- Do NOT import anything
- The DataFrame index is integers (0, 1, 2, ...) — NEVER use df.loc[date] or df.loc[timestamp]
- To filter up to the signal date: df = df[df['date'] <= date].copy()  — ALWAYS use .copy() to avoid modifying the original data
- NEVER do df = df[df['date'] <= date] without .copy() — this causes non-deterministic results across runs
- To get the latest row: df.iloc[-1]  (after filtering)
- Always filter by date before accessing any row
- When computing indicators (moving averages, ATR, etc.), always work on the .copy() result
- Use pct_change() for percentage returns, not diff() — diff() gives absolute price change which is scale-dependent
- When ranking/sorting stocks, ALWAYS use a numeric score (float), NEVER a boolean — boolean comparisons like (a > b) return True/False which makes sort meaningless
- For breakout magnitude: compute the numeric excess e.g. breakout_mag = price_change - 1.5 * atr, then check breakout_mag > 0 AND append breakout_mag as the score
- np.maximum() only accepts TWO arrays at a time — for three-way max (e.g. True Range) always nest: np.maximum(a, np.maximum(b, c)), NEVER np.maximum(a, b, c)
- When mixing ATR (absolute price units) with returns (percentage), ALWAYS normalize to the same unit before comparing or subtracting. Use atr_pct = atr / close.shift() to convert ATR to percentage, then compare with pct_change() returns
- ALWAYS filter by date first, THEN check len(df) < N, THEN compute indicators:
    df = df[df['date'] <= date].copy()   # 1. filter + copy
    if len(df) < N: continue             # 2. length guard (on the filtered slice)
    df['indicator'] = ...                # 3. compute indicators
  Do NOT check len(df) before the date filter — the full history may be long enough but the filtered slice may not be

Respond in this EXACT format — no JSON, no extra text:

EXPLANATION: <one sentence describing what the function does>
```python
def select(universe_df, history, date):
    ...
```
"""

SYSTEM_REFLECTOR = """You are a code reviewer checking Python stock selection functions for bugs.

Check for these specific issues:
1. Missing .copy() after DataFrame filter: df = df[...] without .copy() causes non-deterministic results
2. Boolean used as sort score: expressions like (a > b) return True/False, not a numeric score — sort becomes meaningless
3. Using diff() for returns instead of pct_change() — diff() is scale-dependent (absolute price change)
4. len(df) < N check placed BEFORE the date filter — must be AFTER: first do df = df[df['date'] <= date].copy(), then check len(df) < N. Otherwise the full history may have enough rows but the filtered slice does not, making indicators full of NaN.
5. len(df) < N check placed AFTER computing indicators — should be BEFORE indicators
6. Using df.loc[timestamp] to filter by date — index is integers, use df[df['date'] <= date]
7. np.maximum() called with more than two arguments — must be nested: np.maximum(a, np.maximum(b, c))
8. Mixing ATR (absolute price) with pct_change() returns (percentage) in the same expression — they have different units and must be normalized first (e.g. atr_pct = atr / close.shift())
9. Any syntax errors or obvious logic bugs

Respond in this EXACT format:

OK
or
ISSUES:
- <issue 1>
- <issue 2>
"""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_code(raw: str, description: str) -> Optional[tuple[str, str]]:
    """Extract (code, explanation) from LLM response. Returns None if not found."""
    # 1. ```python block (primary)
    m = re.search(r'```python\s*(.*?)```', raw, re.DOTALL)
    if m:
        code = m.group(1).strip()
        expl = re.search(r'EXPLANATION\s*:\s*(.+)', raw)
        return code, (expl.group(1).strip() if expl else description)

    # 2. Any fenced block containing def select
    m = re.search(r'```\s*(def select.*?)```', raw, re.DOTALL)
    if m:
        code = m.group(1).strip()
        expl = re.search(r'EXPLANATION\s*:\s*(.+)', raw)
        return code, (expl.group(1).strip() if expl else description)

    # 3. Bare def select (last resort)
    m = re.search(r'(def select\s*\(.*)', raw, re.DOTALL)
    if m:
        code = m.group(1).strip()
        expl = re.search(r'EXPLANATION\s*:\s*(.+)', raw)
        return code, (expl.group(1).strip() if expl else description)

    return None


def _extract_issues(raw: str) -> Optional[list[str]]:
    """Parse reflector response. Returns None if OK, list of issues otherwise."""
    text = raw.strip()
    if text.upper().startswith("OK"):
        return None
    # Extract bullet points after ISSUES:
    issues = re.findall(r'-\s*(.+)', text)
    return issues if issues else ["Unspecified issues found"]


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def _call_llm(client, model: str, messages: list) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1500,
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


def generate_signal_code(description: str, llm_config: dict) -> dict:
    """
    Call LLM to generate a select() function from natural language,
    then run a reflector loop (up to MAX_REFLECTION_ATTEMPTS) to fix issues.

    Returns: {
        "code": str,
        "explanation": str,
        "success": bool,
        "error": str,
        "attempts": int,
    }
    """
    api_key  = llm_config.get("api_key", "").strip()
    base_url = llm_config.get("base_url", "https://api.openai.com/v1") or "https://api.openai.com/v1"
    model    = llm_config.get("model", "gpt-4o-mini") or "gpt-4o-mini"

    if not api_key:
        return {"code": "", "explanation": "", "success": False, "error": "No API key configured", "attempts": 0}

    try:
        from openai import OpenAI
        import httpx
        client = OpenAI(api_key=api_key, base_url=base_url, http_client=httpx.Client(verify=False))

        gen_messages = [
            {"role": "system", "content": SYSTEM_GENERATOR},
            {"role": "user",   "content": f"Write a select() function that: {description}"},
        ]

        for attempt in range(1, MAX_REFLECTION_ATTEMPTS + 1):
            # ── Generate ──────────────────────────────────────────────────────
            raw = _call_llm(client, model, gen_messages)
            log.debug(f"[signal_builder] attempt {attempt} generator response: {raw[:200]}")

            parsed = _extract_code(raw, description)
            if parsed is None:
                return {"code": "", "explanation": "", "success": False,
                        "error": f"Could not extract code from LLM response: {raw[:200]}",
                        "attempts": attempt}

            code, explanation = parsed

            # ── Reflect ───────────────────────────────────────────────────────
            ref_raw = _call_llm(client, model, [
                {"role": "system",  "content": SYSTEM_REFLECTOR},
                {"role": "user",    "content": f"Review this code:\n\n```python\n{code}\n```"},
            ])
            log.debug(f"[signal_builder] attempt {attempt} reflector response: {ref_raw[:200]}")

            issues = _extract_issues(ref_raw)

            if issues is None:
                # Reflector approved — done
                log.info(f"[signal_builder] approved after {attempt} attempt(s)")
                return {"code": code, "explanation": explanation,
                        "success": True, "error": "", "attempts": attempt}

            # Feed issues back into generator conversation
            log.info(f"[signal_builder] attempt {attempt} issues: {issues}")
            issues_text = "\n".join(f"- {i}" for i in issues)
            gen_messages.append({"role": "assistant", "content": raw})
            gen_messages.append({
                "role": "user",
                "content": (
                    f"Your code has the following issues, please fix them:\n{issues_text}\n\n"
                    "Respond again in the same format (EXPLANATION: ... then ```python block)."
                ),
            })

        # Exhausted attempts — return last generated code with a warning
        log.warning(f"[signal_builder] max attempts reached, returning last code with known issues")
        return {"code": code, "explanation": explanation,
                "success": True, "error": f"Warning: reflector found issues after {MAX_REFLECTION_ATTEMPTS} attempts: {'; '.join(issues)}",
                "attempts": MAX_REFLECTION_ATTEMPTS}

    except Exception as e:
        log.error(f"[signal_builder] LLM call failed: {e}")
        return {"code": "", "explanation": "", "success": False, "error": str(e), "attempts": 0}


# ---------------------------------------------------------------------------
# Safe compile / execute
# ---------------------------------------------------------------------------

_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "filter": filter, "float": float, "format": format,
    "hasattr": hasattr, "int": int, "isinstance": isinstance, "issubclass": issubclass,
    "iter": iter, "len": len, "list": list, "map": map, "max": max, "min": min,
    "next": next, "print": print, "range": range, "reversed": reversed,
    "round": round, "set": set, "slice": slice, "sorted": sorted, "str": str,
    "sum": sum, "tuple": tuple, "type": type, "zip": zip,
    "True": True, "False": False, "None": None,
}

_BLOCKED_PATTERNS = [
    r"\bimport\b",
    r"\bopen\s*\(",
    r"\bexec\s*\(",
    r"\beval\s*\(",
    r"\b__import__\s*\(",
    r"\b__builtins__\b",
    r"\bos\.(?:path|system|popen|listdir|remove|rename|getcwd|environ)\b",
    r"\bsubprocess\b", r"\bshutil\b", r"\bpathlib\b",
]


def _check_blocked(code: str) -> Optional[str]:
    for pattern in _BLOCKED_PATTERNS:
        if re.search(pattern, code):
            return f"Code contains blocked pattern: {pattern!r}"
    return None


def safe_compile(code: str) -> tuple[Optional[Callable], str]:
    blocked_msg = _check_blocked(code)
    if blocked_msg:
        return None, blocked_msg

    if "def select(" not in code and "def select (" not in code:
        return None, "Code must define a 'select' function"

    namespace = {
        "__builtins__": _SAFE_BUILTINS,
        "pd": pd,
        "np": np,
    }

    try:
        exec(compile(code, "<signal>", "exec"), namespace)  # noqa: S102
    except SyntaxError as e:
        return None, f"Syntax error: {e}"
    except Exception as e:
        return None, f"Compilation error: {e}"

    fn = namespace.get("select")
    if fn is None or not callable(fn):
        return None, "No callable 'select' function found after compilation"

    return fn, ""


def safe_call_select(fn: Callable, universe_df: pd.DataFrame, history: dict, date) -> tuple[list, str]:
    result_container = [None]
    error_container  = [""]

    def _run():
        try:
            result = fn(universe_df, history, date)
            if isinstance(result, (list, tuple)):
                result_container[0] = [str(t) for t in result[:20]]
            else:
                error_container[0] = f"select() returned {type(result).__name__}, expected list"
        except KeyError as e:
            error_container[0] = (
                f"KeyError: {e}. "
                "Tip: don't use df.loc[date] — the index is integers. "
                "Use df[df['date'] <= date].iloc[-1] to get the latest row."
            )
        except Exception as e:
            error_container[0] = f"Runtime error: {type(e).__name__}: {e}"

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=30)

    if t.is_alive():
        return [], "select() timed out after 30 seconds"
    if error_container[0]:
        return [], error_container[0]
    return result_container[0] or [], ""


def validate_select_fn(code: str, universe_df: pd.DataFrame, history: dict, test_date) -> dict:
    fn, compile_error = safe_compile(code)
    if fn is None:
        return {"valid": False, "tickers": [], "error": compile_error}

    tickers, run_error = safe_call_select(fn, universe_df, history, test_date)
    if run_error:
        return {"valid": False, "tickers": [], "error": run_error}

    return {"valid": True, "tickers": tickers, "error": ""}
