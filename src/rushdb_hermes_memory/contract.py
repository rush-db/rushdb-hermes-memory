from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_scope(*parts: str, salt: str = "") -> str:
    return sha256(f"{salt}\x1e" + "\x1f".join(part.strip() for part in parts))


def episode_event_id(event: dict[str, Any]) -> str:
    return sha256(
        stable_json(
            {
                "schemaVersion": SCHEMA_VERSION,
                "eventType": "episode",
                "runtime": event["runtime"],
                "agentId": event["agentId"],
                "profileId": event["profileId"],
                "externalSessionId": event["externalSessionId"],
                "sourceEventId": event.get("sourceEventId", ""),
                "turnIndex": event["turnIndex"],
                "userText": event["userText"],
                "assistantText": event["assistantText"],
            }
        )
    )


def fact_event_id(event: dict[str, Any]) -> str:
    return sha256(
        stable_json(
            {
                "schemaVersion": SCHEMA_VERSION,
                "eventType": "fact",
                "runtime": event["runtime"],
                "agentId": event["agentId"],
                "profileId": event["profileId"],
                "participantScopeHash": event["participantScopeHash"],
                "subjectKey": event["subjectKey"],
                "kind": event["kind"],
                "sourceEventId": event["sourceEventId"],
                "text": event["text"],
            }
        )
    )


def bounded_text(value: str, limit: int = 6000) -> str:
    normalized = value.replace("\x00", "").strip()
    return normalized if len(normalized) <= limit else normalized[:limit] + "…"
