"""SQLite-backed chat history with context compaction.

Replaces the frontend-passed history list so conversations survive page refreshes
and backend controls the context window budget.

Schema:
  chat_messages  — one row per message (user or agent)
  chat_summary   — one compacted summary per session (covers old messages)
"""

import os
import sqlite3

_DB_PATH = None


def _db_path() -> str:
    global _DB_PATH
    if _DB_PATH is None:
        here = os.path.dirname(os.path.abspath(__file__))
        _DB_PATH = os.path.join(here, "..", "data", "chat_history.db")
    return _DB_PATH


def _conn() -> sqlite3.Connection:
    db = sqlite3.connect(_db_path())
    db.row_factory = sqlite3.Row
    return db


def init_db():
    os.makedirs(os.path.dirname(_db_path()), exist_ok=True)
    with _conn() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                role        TEXT    NOT NULL,   -- 'user' | 'agent'
                content     TEXT    NOT NULL,
                ts          DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_session
                ON chat_messages(session_id, id);

            CREATE TABLE IF NOT EXISTS chat_summary (
                session_id          TEXT PRIMARY KEY,
                summary             TEXT    NOT NULL,
                summarized_through  INTEGER NOT NULL   -- id of last summarized message
            );
        """)


def save_turn(session_id: str, user_msg: str, agent_msg: str):
    """Append one user+agent turn to the history."""
    with _conn() as db:
        db.execute(
            "INSERT INTO chat_messages (session_id, role, content) VALUES (?, 'user',  ?)",
            (session_id, user_msg),
        )
        db.execute(
            "INSERT INTO chat_messages (session_id, role, content) VALUES (?, 'agent', ?)",
            (session_id, agent_msg),
        )


# ── how many recent turns to keep as full messages ──────────────────────────
_KEEP_TURNS = 6        # 12 messages (6 user + 6 agent)
_COMPACT_AFTER = 10    # compact once history exceeds this many turns


def load_history(session_id: str) -> list[dict]:
    """Return history as list of {role, content} dicts for the LLM.

    Recent _KEEP_TURNS turns are returned verbatim.
    Older turns are replaced by a compact summary (if one has been generated).
    """
    if not session_id:
        return []

    with _conn() as db:
        rows = db.execute(
            "SELECT id, role, content FROM chat_messages "
            "WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()

    if not rows:
        return []

    messages = [{"id": r["id"], "role": r["role"], "content": r["content"]} for r in rows]
    keep = _KEEP_TURNS * 2  # messages, not turns

    if len(messages) <= keep:
        return _to_llm_fmt(messages)

    # There are older messages beyond the keep window
    recent   = messages[-keep:]
    older    = messages[:-keep]

    history = []

    # Inject summary of older messages if available
    with _conn() as db:
        row = db.execute(
            "SELECT summary FROM chat_summary WHERE session_id = ?",
            (session_id,),
        ).fetchone()

    if row:
        history.append({
            "role":    "assistant",
            "content": f"[Summary of earlier conversation]: {row['summary']}",
        })
    else:
        # No summary yet — include a brief hint from the oldest visible turn
        if older:
            m = older[-1]
            role = "user" if m["role"] == "user" else "assistant"
            history.append({"role": role, "content": m["content"][:300] + " …"})

    history.extend(_to_llm_fmt(recent))
    return history


def needs_compaction(session_id: str) -> bool:
    """True if this session has more than _COMPACT_AFTER turns without a summary."""
    if not session_id:
        return False
    with _conn() as db:
        total = db.execute(
            "SELECT COUNT(*) FROM chat_messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        already = db.execute(
            "SELECT summarized_through FROM chat_summary WHERE session_id = ?",
            (session_id,),
        ).fetchone()

    turns = total // 2
    if turns <= _COMPACT_AFTER:
        return False
    if already:
        # Only recompact if significant new messages have arrived since last summary
        # (avoid recompacting on every turn)
        with _conn() as db:
            latest_id = db.execute(
                "SELECT MAX(id) FROM chat_messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
        return (latest_id - already["summarized_through"]) >= _KEEP_TURNS * 2
    return True


def save_summary(session_id: str, summary: str):
    """Store a compacted summary, covering everything up to (but not including)
    the most recent _KEEP_TURNS turns."""
    with _conn() as db:
        rows = db.execute(
            "SELECT id FROM chat_messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    if not rows:
        return
    keep = _KEEP_TURNS * 2
    older = rows[:-keep] if len(rows) > keep else []
    if not older:
        return
    through_id = older[-1]["id"]
    with _conn() as db:
        db.execute(
            "INSERT OR REPLACE INTO chat_summary "
            "(session_id, summary, summarized_through) VALUES (?, ?, ?)",
            (session_id, summary, through_id),
        )


def clear_session(session_id: str):
    with _conn() as db:
        db.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        db.execute("DELETE FROM chat_summary  WHERE session_id = ?", (session_id,))


# ── helpers ─────────────────────────────────────────────────────────────────

def _to_llm_fmt(messages: list[dict]) -> list[dict]:
    return [
        {"role": "user" if m["role"] == "user" else "assistant", "content": m["content"]}
        for m in messages
    ]
