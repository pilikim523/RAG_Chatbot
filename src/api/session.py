"""In-memory + PostgreSQL-backed session store for multi-turn GooverContext sessions."""
from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from typing import Callable

from src.chatbot_goover_context import GooverContext
from src.api import persistence


class SessionStore:
    """Thread-safe LRU store mapping session_id → GooverContext.

    - In-memory cache for active sessions (fast path).
    - PostgreSQL backing store for persistence across restarts.
    - On cache miss, attempts to restore history from DB.
    - persist() is called after each chat turn to save history.
    """

    def __init__(
        self,
        factory: Callable[[], GooverContext],
        max_sessions: int = 200,
    ) -> None:
        self._factory = factory
        self._max = max_sessions
        self._store: OrderedDict[str, GooverContext] = OrderedDict()
        self._lock = threading.Lock()
        try:
            persistence.init_db()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("DB init failed (sessions won't persist): %s", e)

    def get_or_create(self, session_id: str | None) -> tuple[str, GooverContext]:
        """Return (session_id, context). Restores from DB on cache miss."""
        with self._lock:
            if session_id and session_id in self._store:
                self._store.move_to_end(session_id)
                return session_id, self._store[session_id]

            new_id = session_id or str(uuid.uuid4())
            ctx = self._factory()

            # Try to restore history from PostgreSQL on cache miss
            if session_id:
                try:
                    turns = persistence.load_session(session_id)
                    if turns:
                        ctx.restore_history(turns)
                except Exception:
                    pass

            if len(self._store) >= self._max:
                self._store.popitem(last=False)
            self._store[new_id] = ctx
            return new_id, ctx

    def persist(self, session_id: str) -> None:
        """Save current history of a session to PostgreSQL."""
        with self._lock:
            ctx = self._store.get(session_id)
            if ctx is None:
                return
            history = ctx.history

        if not history:
            return

        user_turns = [t for t in history if t.role == "user"]
        title = user_turns[0].content[:40] if user_turns else None
        try:
            persistence.save_session(session_id, title, history)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Failed to persist session %s: %s", session_id, e)

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)
        try:
            persistence.delete_session(session_id)
        except Exception:
            pass

    def reset(self, session_id: str) -> bool:
        """Reset history for an existing session. Returns True if session existed."""
        with self._lock:
            ctx = self._store.get(session_id)
            if ctx is None:
                try:
                    if not persistence.load_session(session_id):
                        return False
                except Exception:
                    return False
            else:
                ctx.reset()
        try:
            persistence.save_session(session_id, None, [])
        except Exception:
            pass
        return True

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)
