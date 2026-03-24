"""
Signal Builder
==============
Converts natural language stock selection description -> Python select() function
using an LLM.

The generated function signature:
    def select(universe_df: pd.DataFrame, history: dict, date) -> list[str]:
        ...
"""

from __future__ import annotations

import json
import re
import textwrap
import threading
from typing import Callable, Optional

import pandas as pd
import numpy as np

from src.common.logger import get_logger

log = get_logger("backtest.signal_builder")

# ---------------------------------------------------------------------------
# LLM integration
# ---------------------------------------------------------------------------

SYSTEM = """You are a quantitative analyst writing Python stock selection functions.

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
- To filter up to the signal date: df = df[df['date'] <= date]
- To get the latest row: df.iloc[-1]  (after filtering)
- Always filter by date before accessing any row

Return JSON: {"code": "<python code>", "explanation": "<one sentence what it does>"}
"""


def generate_signal_code(description: str, llm_config: dict) -> dict:
    """
    Call LLM to generate a select() function from natural language.

    llm_config: {"api_key": str, "base_url": str, "model": str}

    Returns: {
        "code": str,        # Python code string
        "explanation": str, # LLM explanation of what the code does
        "success": bool,
        "error": str,       # if success=False
    }
    """
    api_key = llm_config.get("api_key", "").strip()
    base_url = llm_config.get("base_url", "https://api.openai.com/v1") or "https://api.openai.com/v1"
    model = llm_config.get("model", "gpt-4o-mini") or "gpt-4o-mini"

    if not api_key:
        return {"code": "", "explanation": "", "success": False, "error": "No API key configured"}

    user_message = f"Write a select() function that: {description}"

    try:
        from openai import OpenAI
        import httpx
        client = OpenAI(api_key=api_key, base_url=base_url, http_client=httpx.Client(verify=False))

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_message},
        ]

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1500,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or ""
        log.debug(f"[signal_builder] LLM raw response: {raw[:200]}")

        # 1. Prefer explicit ```python code block — most reliable
        code_match = re.search(r'```python\s*(.*?)```', raw, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            # Try to grab explanation from surrounding text
            expl_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', raw)
            explanation = expl_match.group(1) if expl_match else description
            return {"code": code, "explanation": explanation, "success": True, "error": ""}

        # 2. Try JSON — LLM may embed literal newlines in string values (invalid JSON),
        #    so encode them before parsing then decode back after.
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                # Fix literal control characters inside JSON string values
                def _fix_json(s):
                    # Replace literal newlines/tabs only inside string values
                    out, in_str, i = [], False, 0
                    while i < len(s):
                        c = s[i]
                        if c == '"' and (i == 0 or s[i-1] != '\\'):
                            in_str = not in_str
                            out.append(c)
                        elif in_str and c == '\n':
                            out.append('\\n')
                        elif in_str and c == '\t':
                            out.append('\\t')
                        elif in_str and c == '\r':
                            out.append('\\r')
                        else:
                            out.append(c)
                        i += 1
                    return ''.join(out)
                try:
                    parsed = json.loads(_fix_json(json_str))
                except json.JSONDecodeError as e:
                    return {"code": "", "explanation": "", "success": False,
                            "error": f"Could not parse LLM response: {e}"}

            code = parsed.get("code", "").strip()
            explanation = parsed.get("explanation", "").strip()
            if code:
                return {"code": code, "explanation": explanation, "success": True, "error": ""}

        return {"code": "", "explanation": "", "success": False,
                "error": f"Could not extract code from LLM response: {raw[:200]}"}

    except Exception as e:
        log.error(f"[signal_builder] LLM call failed: {e}")
        return {"code": "", "explanation": "", "success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Safe compile / execute
# ---------------------------------------------------------------------------

# Allowed built-in names for sandboxed execution
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
    r"\bopen\s*\(",        # open() as a function call, not as a column name string
    r"\bexec\s*\(",
    r"\beval\s*\(",
    r"\b__import__\s*\(",
    r"\b__builtins__\b",
    r"\bos\.(?:path|system|popen|listdir|remove|rename|getcwd|environ)\b",
    r"\bsubprocess\b", r"\bshutil\b", r"\bpathlib\b",
]


def _check_blocked(code: str) -> Optional[str]:
    """Check for blocked patterns. Returns error message or None."""
    for pattern in _BLOCKED_PATTERNS:
        if re.search(pattern, code):
            return f"Code contains blocked pattern: {pattern!r}"
    return None


def safe_compile(code: str) -> tuple[Optional[Callable], str]:
    """
    Safely compile the select() function code.
    Returns (fn, error_message). If error, fn is None.

    Only allows: pd, np, built-ins (list, dict, sorted, len, range, etc.)
    Blocks: import, open, exec, eval, __import__, os, sys
    """
    # Security check
    blocked_msg = _check_blocked(code)
    if blocked_msg:
        return None, blocked_msg

    # Ensure code defines a select() function
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
    """
    Safely call fn(universe_df, history, date) with a 30-second timeout.
    Returns (tickers_list, error_message).
    """
    result_container = [None]
    error_container = [""]

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
    """
    Compile + run the function on test data to validate it works.
    Returns {"valid": bool, "tickers": [...], "error": str}
    """
    fn, compile_error = safe_compile(code)
    if fn is None:
        return {"valid": False, "tickers": [], "error": compile_error}

    tickers, run_error = safe_call_select(fn, universe_df, history, test_date)
    if run_error:
        return {"valid": False, "tickers": [], "error": run_error}

    return {"valid": True, "tickers": tickers, "error": ""}
