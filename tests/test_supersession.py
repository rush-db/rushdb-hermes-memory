from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rushdb_hermes_memory.outbox import DurableOutbox
from rushdb_hermes_memory.provider import RushDBMemoryProvider


class StoredFact(dict):
    @property
    def id(self) -> str:
        return self["__id"]


class FactStore:
    """Applies upserts, finds and updates to MEMORY_FACT records in memory."""

    def __init__(self) -> None:
        self.facts: list[StoredFact] = []
        self.upserts: list[dict[str, Any]] = []
        self.updates: list[tuple[str, dict[str, Any]]] = []

    def upsert(
        self, *, data: dict[str, Any], label: str, options: dict[str, Any], **_: Any
    ) -> None:
        self.upserts.append({"data": dict(data), "label": label})
        key = options["mergeBy"][0]
        for fact in self.facts:
            if fact.get(key) == data[key]:
                fact.update(data)
                return
        self.facts.append(StoredFact(data, __id=f"rec-{len(self.facts) + 1}"))

    def find(self, query: dict[str, Any]) -> list[StoredFact]:
        where = query["where"]
        return [f for f in self.facts if all(f.get(k) == v for k, v in where.items())]

    def update(self, target: str, data: dict[str, Any], label: str | None = None) -> None:
        self.updates.append((target, data))
        next(f for f in self.facts if f["__id"] == target).update(data)

    def active_texts(self) -> list[str]:
        return [f["text"] for f in self.facts if f["active"]]


class FakeDB:
    def __init__(self) -> None:
        self.records = FactStore()


def start(tmp_path: Path, db: FakeDB, session: str = "session-1") -> RushDBMemoryProvider:
    instance = RushDBMemoryProvider(db_factory=lambda *args, **kwargs: db)
    instance.initialize(
        session,
        hermes_home=str(tmp_path),
        platform="cli",
        agent_identity="coder",
        agent_context="primary",
    )
    return instance


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RUSHDB_API_KEY", "test-key")
    db = FakeDB()
    instance = start(tmp_path, db)
    yield instance, db.records, tmp_path
    instance.shutdown()


def test_replace_deactivates_the_replaced_fact_and_links_it(setup) -> None:
    instance, store, _ = setup
    instance.on_memory_write("add", "memory", "Billing service language: TypeScript")
    instance.on_memory_write("add", "memory", "Deploys go out on Tuesdays")
    instance.on_session_end([])
    old = next(f for f in store.facts if "TypeScript" in f["text"])

    instance.on_memory_write(
        "replace",
        "memory",
        "Billing service language: Go",
        {"previous_content": "Billing service language: TypeScript", "old_text": "TypeScript"},
    )
    instance.on_session_end([])

    assert store.active_texts() == ["Deploys go out on Tuesdays", "Billing service language: Go"]
    new = next(f for f in store.facts if "Go" in f["text"])
    assert new["supersedesFactId"] == old["factId"]
    assert old["active"] is False
    assert old["validUntil"] == new["validFrom"]
    assert all("_supersedesText" not in u["data"] for u in store.upserts)


def test_replace_without_previous_content_keeps_the_earlier_fact(setup) -> None:
    instance, store, _ = setup
    instance.on_memory_write("add", "memory", "Billing service language: TypeScript")
    # Older Hermes callers send only old_text, which is not an identity.
    instance.on_memory_write(
        "replace", "memory", "Billing service language: Go", {"old_text": "TypeScript"}
    )
    instance.on_session_end([])
    assert store.updates == []
    assert len(store.active_texts()) == 2
    assert all("supersedesFactId" not in f for f in store.facts)


def test_replace_only_matches_the_same_target_and_exact_entry(setup) -> None:
    instance, store, _ = setup
    instance.on_memory_write("add", "user", "Billing service language: TypeScript")
    instance.on_memory_write("add", "memory", "Billing service language: TypeScript, mostly")
    instance.on_session_end([])
    instance.on_memory_write(
        "replace",
        "memory",
        "Billing service language: Go",
        {"previous_content": "Billing service language: TypeScript"},
    )
    instance.on_session_end([])
    assert store.updates == []
    assert len(store.active_texts()) == 3


def test_remove_deactivates_the_removed_fact_without_writing_one(setup) -> None:
    instance, store, _ = setup
    instance.on_memory_write("add", "memory", "Deploys go out on Tuesdays")
    instance.on_memory_write("add", "memory", "Invoices are kept for 10 years")
    instance.on_session_end([])
    upserts = len(store.upserts)

    instance.on_memory_write(
        "remove", "memory", "", {"previous_content": "Deploys go out on Tuesdays"}
    )
    instance.on_session_end([])

    assert len(store.upserts) == upserts
    assert store.active_texts() == ["Invoices are kept for 10 years"]
    removed = next(f for f in store.facts if "Tuesdays" in f["text"])
    assert removed["validUntil"]


def test_remove_without_previous_content_does_nothing(setup) -> None:
    instance, store, _ = setup
    instance.on_memory_write("add", "memory", "Deploys go out on Tuesdays")
    instance.on_memory_write("remove", "memory", "", {"old_text": "Tuesdays"})
    instance.on_session_end([])
    assert store.updates == []
    assert store.active_texts() == ["Deploys go out on Tuesdays"]


def test_a_replayed_replace_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUSHDB_API_KEY", "test-key")
    db = FakeDB()
    store = db.records
    first = start(tmp_path, db)
    first.on_memory_write("add", "memory", "Billing service language: TypeScript")
    first.on_session_end([])
    old = store.facts[0]

    outbox = DurableOutbox(tmp_path / "rushdb-memory" / "outbox")
    first.on_memory_write(
        "replace",
        "memory",
        "Billing service language: Go",
        {"previous_content": "Billing service language: TypeScript"},
    )
    first.on_session_end([])
    first.shutdown()
    replacement = next(u["data"] for u in store.upserts if "Go" in u["data"]["text"])

    # The process died before acknowledging: the same event is back in the outbox.
    event = dict(replacement)
    event.pop("supersedesFactId")
    event["_supersedesText"] = "Billing service language: TypeScript"
    outbox.put(event)
    second = start(tmp_path, db, session="session-2")
    second.on_session_end([])
    second.shutdown()

    assert len(store.facts) == 2
    assert store.active_texts() == ["Billing service language: Go"]
    assert store.facts[1]["supersedesFactId"] == old["factId"]
    assert outbox.pending() == []
