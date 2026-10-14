from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_SAFE_ID = re.compile(r"^[a-f0-9]{64}$")


class DurableOutbox:
    """Small profile-local write-ahead outbox with atomic file replacement."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, event: dict[str, Any]) -> Path:
        event_id = str(event.get("eventId") or event.get("factId") or "")
        if not _SAFE_ID.fullmatch(event_id):
            raise ValueError("memory event has an invalid deterministic id")

        destination = self.root / f"{event_id}.json"
        temporary = self.root / f".{event_id}.{os.getpid()}.tmp"
        payload = json.dumps(event, ensure_ascii=False, sort_keys=True)

        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        return destination

    def pending(self) -> list[Path]:
        return sorted(self.root.glob("[0-9a-f]" + "[0-9a-f]" * 63 + ".json"))

    @staticmethod
    def read(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def acknowledge(path: Path) -> None:
        path.unlink(missing_ok=True)
