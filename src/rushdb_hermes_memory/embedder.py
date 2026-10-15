"""Client-side embeddings for memory on a standalone RushDB engine.

RushDB Cloud embeds memories with its managed model. A standalone engine has
no managed embeddings, so the provider embeds summaries and queries itself
through any OpenAI-compatible embeddings endpoint, requesting what RushDB
Cloud requests: ``POST <base_url>/embeddings`` with
``{"model", "input", "dimensions"}``.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class OpenAICompatibleEmbedder:
    api_key: str
    model: str
    dimensions: int
    base_url: str = "https://api.openai.com/v1"
    similarity: str = "cosine"
    timeout: float = 30.0

    def embed(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in order."""
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/embeddings",
            data=json.dumps(
                {"model": self.model, "input": texts, "dimensions": self.dimensions}
            ).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
            body = json.loads(response.read().decode("utf-8"))
        rows = sorted(body.get("data") or [], key=lambda row: row.get("index", 0))
        vectors = [row.get("embedding") or [] for row in rows]
        if len(vectors) != len(texts) or any(len(vector) != self.dimensions for vector in vectors):
            raise ValueError(
                f"embedding provider returned {len(vectors)} vectors for {len(texts)} texts, "
                f"expected {self.dimensions} dimensions each"
            )
        return vectors


def embedder_from_env() -> tuple[OpenAICompatibleEmbedder | None, str]:
    """The embedder the environment configures, or the reason it is incomplete.

    Reads ``RUSHDB_EMBEDDING_API_KEY``, ``RUSHDB_EMBEDDING_MODEL``,
    ``RUSHDB_EMBEDDING_DIMENSIONS`` and ``RUSHDB_EMBEDDING_BASE_URL``. None of
    them set: no embedder and no error.
    """
    api_key = os.environ.get("RUSHDB_EMBEDDING_API_KEY", "")
    model = os.environ.get("RUSHDB_EMBEDDING_MODEL", "")
    dimensions = os.environ.get("RUSHDB_EMBEDDING_DIMENSIONS", "")
    if not (api_key or model or dimensions):
        return None, ""
    if not (api_key and model and dimensions.isdigit() and int(dimensions) > 0):
        return None, (
            "Set RUSHDB_EMBEDDING_API_KEY, RUSHDB_EMBEDDING_MODEL and RUSHDB_EMBEDDING_DIMENSIONS "
            "(a positive integer) together"
        )
    base_url = os.environ.get("RUSHDB_EMBEDDING_BASE_URL") or "https://api.openai.com/v1"
    return OpenAICompatibleEmbedder(
        api_key=api_key, model=model, dimensions=int(dimensions), base_url=base_url
    ), ""
