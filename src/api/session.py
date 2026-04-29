"""In-memory session store for multi-turn GooverContext sessions."""
from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from typing import Callable

from src.chatbot_goover_context import GooverContext


class SessionStore:
    """Thread-safe LRU store mapping session_id → GooverContext.

    When max_sessions is reached, the least-recently-used session is evicted.
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

    def get_or_create(self, session_id: str | None) -> tuple[str, GooverContext]:
        """Return (session_id, context). Creates a new session if id is unknown."""
        with self._lock:
            if session_id and session_id in self._store:
                self._store.move_to_end(session_id)
                return session_id, self._store[session_id]

            new_id = session_id or str(uuid.uuid4())
            ctx = self._factory()
            if len(self._store) >= self._max:
                self._store.popitem(last=False)
            self._store[new_id] = ctx
            return new_id, ctx

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)

    def reset(self, session_id: str) -> bool:
        """Reset history for an existing session. Returns True if session existed."""
        with self._lock:
            ctx = self._store.get(session_id)
            if ctx is None:
                return False
            ctx.reset()
            return True

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)
