from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rushdb_hermes_memory.outbox import DurableOutbox
from rushdb_hermes_memory.provider import RushDBMemoryProvider


class FakeRecord(dict):
    score = 0.91


class FakeRecords:
    def __init__(self) -> None:
        self.upserts: list[dict[str, Any]] = []
        self.searches: list[dict[str, Any]] = []

    def upsert(self, **kwargs: Any) -> None:
        self.upserts.append(kwargs)

    def vector_search(self, query: dict[str, Any]) -> Any:
        self.searches.append(query)
        property_name = query["propertyName"]
        if property_name == "summary":
            return SimpleNamespace(
                data=[
                    FakeRecord(
                        eventId="episode-1",
                        summary="The user selected TypeScript.",
                    )
                ]
            )
        return SimpleNamespace(data=[])


class FakeDB:
    def __init__(self) -> None:
        self.records = FakeRecords()


@pytest.fixture
def provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RUSHDB_API_KEY", "test-key")
    db = FakeDB()
    instance = RushDBMemoryProvider(db_factory=lambda *args, **kwargs: db)
    instance.initialize(
        "session-1",
        hermes_home=str(tmp_path),
        platform="cli",
        agent_identity="coder",
        agent_context="primary",
    )
    yield instance, db, tmp_path
    instance.shutdown()


def test_sync_turn_is_remote_non_blocking_and_durable(provider) -> None:
    instance, db, tmp_path = provider
    started = time.monotonic()
    instance.sync_turn("Use TypeScript", "Understood", session_id="session-1")
    elapsed = time.monotonic() - started
    assert elapsed < 0.1
    instance.on_session_end([])
    assert len(db.records.upserts) == 1
    write = db.records.upserts[0]
    assert write["label"] == "EPISODE"
    assert write["options"]["mergeBy"] == ["eventId"]
    assert list((tmp_path / "rushdb-memory" / "outbox").glob("*.json")) == []


def test_prefetch_applies_all_scope_constraints(provider) -> None:
    instance, db, _ = provider
    content = instance.prefetch("Which language did we choose?", limit=4)
    assert "TypeScript" in content
    where = db.records.searches[0]["where"]
    assert where == {
        "agentId": "coder",
        "profileId": "coder",
        "privacyScope": "private",
        "participantScopeHash": instance._participant_scope_hash,
        "sandboxEligible": False,
    }
    assert instance.recall_status().count == 1


def test_memory_write_mirrors_only_explicit_add_or_replace(provider) -> None:
    instance, db, _ = provider
    instance.on_memory_write("remove", "memory", "secret")
    instance.on_memory_write("add", "memory", "Prefer concise answers")
    instance.on_session_end([])
    assert len(db.records.upserts) == 1
    assert db.records.upserts[0]["label"] == "MEMORY_FACT"
    assert db.records.upserts[0]["data"]["active"] is True


def test_registration_entrypoint_uses_real_hermes_shape() -> None:
    from rushdb_hermes_memory import register

    registered = []
    ctx = SimpleNamespace(register_memory_provider=registered.append)
    register(ctx)
    assert len(registered) == 1
    assert registered[0].name == "rushdb"


def test_tool_result_is_json(provider) -> None:
    instance, _, _ = provider
    result = json.loads(
        instance.handle_tool_call(
            "rushdb_memory_recall",
            {"query": "language", "limit": 2},
        )
    )
    assert result["count"] == 1
    assert "TypeScript" in result["content"]


def test_startup_replays_a_surviving_outbox_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUSHDB_API_KEY", "test-key")
    event = {
        "schemaVersion": 1,
        "eventType": "episode",
        "eventId": "a" * 64,
        "runtime": "hermes",
        "agentId": "coder",
        "profileId": "coder",
        "externalSessionId": "session-crashed",
        "turnIndex": 3,
        "userText": "Remember this",
        "assistantText": "Stored",
        "summary": "A replayable episode",
        "conversationKind": "direct",
        "privacyScope": "private",
        "participantScopeHash": "b" * 64,
        "sandboxEligible": False,
        "visibility": "participant",
        "trustClass": "mixed",
        "originClass": "conversation",
        "observedAt": "2026-08-18T00:00:00Z",
        "provenance": "hermes:sync_turn",
    }
    outbox = DurableOutbox(tmp_path / "rushdb-memory" / "outbox")
    outbox.put(event)
    db = FakeDB()
    instance = RushDBMemoryProvider(db_factory=lambda *args, **kwargs: db)
    instance.initialize(
        "session-new",
        hermes_home=str(tmp_path),
        platform="cli",
        agent_identity="coder",
        agent_context="primary",
    )
    instance.on_session_end([])
    instance.shutdown()
    assert db.records.upserts[0]["data"]["eventId"] == "a" * 64
    assert outbox.pending() == []


def test_non_primary_agent_context_never_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUSHDB_API_KEY", "test-key")
    db = FakeDB()
    instance = RushDBMemoryProvider(db_factory=lambda *args, **kwargs: db)
    instance.initialize(
        "session-child",
        hermes_home=str(tmp_path),
        platform="cli",
        agent_identity="coder",
        agent_context="subagent",
    )
    instance.sync_turn("child task", "child result")
    instance.on_memory_write("add", "memory", "must not persist")
    instance.shutdown()
    assert db.records.upserts == []
