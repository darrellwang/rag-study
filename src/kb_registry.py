"""
知识库注册表与用户权限（SQLite）

为每个知识库登记 owner（创建者）。owner 由请求携带的 API Key 派生：
  - 未启用鉴权（KB_API_KEY 为空）时，owner 记为 "anonymous"，不做权限校验
  - 启用鉴权时，owner = 该 API Key 对应的用户标识（这里直接用 Key 本身哈希前 8 位作为 user_id）

库文件默认在 config.DB_DIR/kb_registry.db。
"""
import os
import sqlite3
import hashlib

from config import BASE_DIR, SERVICE_API_KEY

DB_PATH = os.path.join(BASE_DIR, "kb_registry.db")


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS kb_registry (
            kb_id TEXT PRIMARY KEY,
            owner TEXT,
            created_at REAL
        )
        """
    )
    conn.commit()
    return conn


def user_id_from_key(api_key: str = "") -> str:
    """由 API Key 派生 user_id。"""
    if not SERVICE_API_KEY:
        return "anonymous"
    if not api_key:
        return "anonymous"
    return "u_" + hashlib.sha256(api_key.encode()).hexdigest()[:8]


def register_kb(kb_id: str, owner: str) -> None:
    conn = _conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO kb_registry (kb_id, owner, created_at) VALUES (?,?,?)",
            (kb_id, owner, __import__("time").time()),
        )
        conn.commit()
    finally:
        conn.close()


def get_owner(kb_id: str) -> str:
    conn = _conn()
    try:
        cur = conn.execute("SELECT owner FROM kb_registry WHERE kb_id=?", (kb_id,))
        row = cur.fetchone()
        return row[0] if row else "anonymous"
    finally:
        conn.close()


def is_owner(kb_id: str, user: str) -> bool:
    """校验 user 是否为该 kb 的 owner。未启用鉴权时恒为 True。"""
    if not SERVICE_API_KEY:
        return True
    return get_owner(kb_id) == user


def delete_registry(kb_id: str) -> None:
    conn = _conn()
    try:
        conn.execute("DELETE FROM kb_registry WHERE kb_id=?", (kb_id,))
        conn.commit()
    finally:
        conn.close()
