from __future__ import annotations

import json
from pathlib import Path

from rushdb_hermes_memory.contract import episode_event_id


def test_event_id_matches_shared_contract_fixture() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "conformance.v1.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert episode_event_id(fixture["episodeInput"]) == fixture["expectedEventId"]
