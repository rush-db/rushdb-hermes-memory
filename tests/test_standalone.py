from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from test_provider import FakeRecords

from rushdb_hermes_memory import provider as provider_module
from rushdb_hermes_memory.embedder import OpenAICompatibleEmbedder, embedder_from_env
from rushdb_hermes_memory.provider import RushDBMemoryProvider


class FakeIndexes:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def find(self) -> Any:
        return SimpleNamespace(data=[])

    def create(self, params: dict[str, Any]) -> None:
        self.created.append(params)


class FakeDB:
    def __init__(self) -> None:
        self.records = FakeRecords()
        self.ai = SimpleNamespace(indexes=FakeIndexes())


class FakeEmbedder:
    dimensions = 3
    similarity = "cosine"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.fixture
def standalone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in ("RUSHDB_API_KEY", "RUSHDB_API_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RUSHDB_ENGINE_URL", "http://127.0.0.1:3000")
    monkeypatch.setenv("RUSHDB_ENGINE_TOKEN", "engine-token")
    monkeypatch.setattr(provider_module, "embedder_from_env", lambda: (FakeEmbedder(), ""))
    db = FakeDB()
    calls: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> FakeDB:
        calls.append(kwargs)
        return db

    instance = RushDBMemoryProvider(db_factory=factory)
    instance.initialize(
        "session-1", hermes_home=str(tmp_path), platform="cli", agent_identity="coder"
    )
    yield instance, db, calls
    instance.shutdown()


def test_connects_to_the_engine_and_declares_external_indexes(standalone) -> None:
    _, db, calls = standalone
    assert calls == [
        {
            "api_key": "engine-token",
            "direct": True,
            "engine_url": "http://127.0.0.1:3000",
            "scope": None,
        }
    ]
    assert [
        (index["label"], index["external"], index["dimensions"]) for index in db.ai.indexes.created
    ] == [
        ("EPISODE", True, 3),
        ("MEMORY_FACT", True, 3),
    ]


def test_writes_carry_their_vector_and_recall_searches_by_vector(standalone) -> None:
    instance, db, _ = standalone
    instance.sync_turn("Use TypeScript", "Understood", session_id="session-1")
    instance.on_session_end([])
    assert db.records.upserts[0]["vectors"] == [
        {"propertyName": "summary", "vector": [0.1, 0.2, 0.3], "similarityFunction": "cosine"}
    ]
    instance.prefetch("Which language did we choose?", limit=4)
    assert db.records.searches[0]["queryVector"] == [0.1, 0.2, 0.3]
    assert "query" not in db.records.searches[0]


def test_a_standalone_engine_without_embeddings_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUSHDB_ENGINE_URL", "http://127.0.0.1:3000")
    monkeypatch.setenv("RUSHDB_ENGINE_TOKEN", "engine-token")
    for name in (
        "RUSHDB_EMBEDDING_API_KEY",
        "RUSHDB_EMBEDDING_MODEL",
        "RUSHDB_EMBEDDING_DIMENSIONS",
    ):
        monkeypatch.delenv(name, raising=False)
    instance = RushDBMemoryProvider(db_factory=lambda **kwargs: FakeDB())
    assert not instance.is_available()
    assert "no managed embeddings" in instance.unavailable_reason()


def test_incomplete_embedding_settings_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUSHDB_EMBEDDING_MODEL", "m")
    monkeypatch.delenv("RUSHDB_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("RUSHDB_EMBEDDING_DIMENSIONS", raising=False)
    embedder, error = embedder_from_env()
    assert embedder is None and "RUSHDB_EMBEDDING_DIMENSIONS" in error


def test_embedder_requests_what_rushdb_cloud_requests() -> None:
    embedder = OpenAICompatibleEmbedder(
        api_key="key", model="m", dimensions=2, base_url="https://provider.example/v1/"
    )
    sent: dict[str, Any] = {}

    class Response:
        def read(self) -> bytes:
            return json.dumps(
                {"data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]}
            ).encode()

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

    def fake_urlopen(request: Any, timeout: float) -> Response:
        sent["url"] = request.full_url
        sent["body"] = json.loads(request.data)
        return Response()

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        assert embedder.embed(["a", "b"]) == [[1, 0], [0, 1]]
    assert sent == {
        "url": "https://provider.example/v1/embeddings",
        "body": {"model": "m", "input": ["a", "b"], "dimensions": 2},
    }
