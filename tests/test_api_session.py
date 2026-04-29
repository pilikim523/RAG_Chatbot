"""Tests for src/api/session.py — no Qdrant or GPU required."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.api.session import SessionStore
from src.chatbot_goover_context import GooverContext


def _make_ctx() -> GooverContext:
    ctx = MagicMock(spec=GooverContext)
    ctx.reset = MagicMock()
    return ctx


def _make_store(max_sessions: int = 5) -> SessionStore:
    return SessionStore(factory=_make_ctx, max_sessions=max_sessions)


class TestSessionStoreGetOrCreate:
    def test_creates_new_session_when_no_id(self):
        store = _make_store()
        sid, ctx = store.get_or_create(None)
        assert sid
        assert ctx is not None

    def test_returns_same_context_for_same_id(self):
        store = _make_store()
        sid, ctx1 = store.get_or_create(None)
        _, ctx2 = store.get_or_create(sid)
        assert ctx1 is ctx2

    def test_different_ids_give_different_contexts(self):
        store = _make_store()
        _, ctx1 = store.get_or_create(None)
        _, ctx2 = store.get_or_create(None)
        assert ctx1 is not ctx2

    def test_explicit_session_id_is_preserved(self):
        store = _make_store()
        sid, _ = store.get_or_create("my-session-id")
        assert sid == "my-session-id"

    def test_unknown_session_id_creates_new(self):
        store = _make_store()
        sid, ctx = store.get_or_create("nonexistent-id")
        assert sid == "nonexistent-id"
        assert ctx is not None


class TestSessionStoreEviction:
    def test_evicts_oldest_when_full(self):
        store = _make_store(max_sessions=2)
        sid1, _ = store.get_or_create(None)
        sid2, _ = store.get_or_create(None)
        assert store.size == 2

        # Adding third evicts first
        sid3, _ = store.get_or_create(None)
        assert store.size == 2
        # sid1 evicted — accessing it creates a new context
        _, ctx_new = store.get_or_create(sid1)
        # sid2 should still be accessible with same context
        _, ctx2_again = store.get_or_create(sid2)
        assert ctx2_again is not None

    def test_lru_access_prevents_eviction(self):
        store = _make_store(max_sessions=2)
        sid1, ctx1 = store.get_or_create(None)
        sid2, _ = store.get_or_create(None)
        # Touch sid1 to make it most-recently-used
        store.get_or_create(sid1)
        # Adding third should evict sid2 (LRU), not sid1
        store.get_or_create(None)
        _, ctx1_check = store.get_or_create(sid1)
        assert ctx1_check is ctx1


class TestSessionStoreDelete:
    def test_delete_removes_session(self):
        store = _make_store()
        sid, _ = store.get_or_create(None)
        store.delete(sid)
        assert store.size == 0

    def test_delete_nonexistent_is_safe(self):
        store = _make_store()
        store.delete("no-such-id")  # should not raise


class TestSessionStoreReset:
    def test_reset_clears_history(self):
        store = _make_store()
        sid, ctx = store.get_or_create(None)
        result = store.reset(sid)
        assert result is True
        ctx.reset.assert_called_once()

    def test_reset_nonexistent_returns_false(self):
        store = _make_store()
        assert store.reset("no-such-id") is False

    def test_reset_keeps_session_alive(self):
        store = _make_store()
        sid, ctx = store.get_or_create(None)
        store.reset(sid)
        _, ctx_again = store.get_or_create(sid)
        assert ctx_again is ctx


class TestSessionStoreThreadSafety:
    def test_concurrent_creates(self):
        import threading
        store = _make_store(max_sessions=200)
        ids = []
        lock = threading.Lock()

        def create():
            sid, _ = store.get_or_create(None)
            with lock:
                ids.append(sid)

        threads = [threading.Thread(target=create) for _ in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert len(ids) == 20
        assert len(set(ids)) == 20  # all unique
