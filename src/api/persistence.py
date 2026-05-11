"""
PostgreSQL-backed session persistence for GooverContext chat history.

Schema (auto-created on init):
  chat_sessions(id, title, history, created_at, updated_at)
  history column: JSONB array of {role, content} objects
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Generator

import psycopg2
import psycopg2.pool

from src.chatbot_goover_context import GooverTurn

# ---------------------------------------------------------------------------
# Connection pool (module-level singleton)
# ---------------------------------------------------------------------------

_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        dsn = os.environ.get("DATABASE_URL", "")
        if not dsn:
            raise RuntimeError("DATABASE_URL 환경변수가 설정되지 않았습니다.")
        _pool = psycopg2.pool.ThreadedConnectionPool(minconn=1, maxconn=10, dsn=dsn)
    return _pool


@contextmanager
def _conn() -> Generator:
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


# ---------------------------------------------------------------------------
# Schema init
# ---------------------------------------------------------------------------

def init_db() -> None:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id          TEXT PRIMARY KEY,
                    title       TEXT,
                    history     JSONB NOT NULL DEFAULT '[]',
                    created_at  DOUBLE PRECISION NOT NULL,
                    updated_at  DOUBLE PRECISION NOT NULL
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated_at
                ON chat_sessions (updated_at DESC)
            """)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def save_session(session_id: str, title: str | None, history: list[GooverTurn]) -> None:
    now = time.time()
    history_json = json.dumps(
        [{"role": t.role, "content": t.content} for t in history],
        ensure_ascii=False,
    )
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chat_sessions (id, title, history, created_at, updated_at)
                VALUES (%s, %s, %s::jsonb, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    title      = EXCLUDED.title,
                    history    = EXCLUDED.history,
                    updated_at = EXCLUDED.updated_at
            """, (session_id, title, history_json, now, now))


def load_session(session_id: str) -> list[GooverTurn] | None:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT history FROM chat_sessions WHERE id = %s",
                (session_id,),
            )
            row = cur.fetchone()
    if row is None:
        return None
    turns_raw = row[0] if isinstance(row[0], list) else json.loads(row[0])
    return [GooverTurn(role=t["role"], content=t["content"]) for t in turns_raw]


def list_sessions(limit: int = 100) -> list[dict]:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, title, created_at, updated_at,
                       jsonb_array_length(history)
                FROM chat_sessions
                ORDER BY updated_at DESC
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "title": r[1] or "새 대화",
            "created_at": r[2],
            "updated_at": r[3],
            "turn_count": (r[4] or 0) // 2,
        }
        for r in rows
    ]


def delete_session(session_id: str) -> bool:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chat_sessions WHERE id = %s", (session_id,))
            return cur.rowcount > 0


def get_session_history(session_id: str) -> dict | None:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, title, history FROM chat_sessions WHERE id = %s",
                (session_id,),
            )
            row = cur.fetchone()
    if row is None:
        return None
    turns_raw = row[2] if isinstance(row[2], list) else json.loads(row[2])
    return {
        "id": row[0],
        "title": row[1] or "새 대화",
        "turns": turns_raw,
    }
