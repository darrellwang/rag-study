"""
会话历史持久化（SQLite）

替换进程内存字典，重启服务不丢失会话。每张表按 session_id 存消息序列。
库文件默认在 config.DB_DIR/sessions.db。
"""
import os
import sqlite3
import json

from config import BASE_DIR

DB_PATH = os.path.join(BASE_DIR, "sessions.db")


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            session_id TEXT,
            turn INTEGER,
            role TEXT,
            content TEXT,
            ts REAL,
            PRIMARY KEY (session_id, turn)
        )
        """
    )
    conn.commit()
    return conn


def append_message(session_id: str, role: str, content: str) -> int:
    """追加一条消息，返回当前轮次（turn）。"""
    conn = _conn()
    try:
        cur = conn.execute("SELECT COALESCE(MAX(turn), -1) FROM messages WHERE session_id=?", (session_id,))
        turn = cur.fetchone()[0] + 1
        conn.execute(
            "INSERT INTO messages (session_id, turn, role, content, ts) VALUES (?,?,?,?,?)",
            (session_id, turn, role, content, __import__("time").time()),
        )
        conn.commit()
        return turn
    finally:
        conn.close()


def get_messages(session_id: str, last_n: int = 6) -> list:
    """返回最近 last_n 轮的消息（保持时间序）。"""
    conn = _conn()
    try:
        cur = conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY turn DESC LIMIT ?",
            (session_id, last_n),
        )
        rows = cur.fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]
    finally:
        conn.close()


def history_text(session_id: str, last_n: int = 6) -> str:
    """把会话历史拼成文本（最近 last_n 轮）。"""
    msgs = get_messages(session_id, last_n)
    pairs = []
    for m in msgs:
        role = "用户" if m["role"] == "user" else "助手"
        pairs.append(f"{role}: {m['content']}")
    return "\n".join(pairs)


def delete_session(session_id: str) -> None:
    conn = _conn()
    try:
        conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        conn.commit()
    finally:
        conn.close()
