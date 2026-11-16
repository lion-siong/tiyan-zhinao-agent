"""Verify that the exported source package carries a public-safe test fixture."""

from __future__ import annotations

import json
from pathlib import Path


def test_public_fixture_is_explicitly_non_research() -> None:
    path = Path(__file__).resolve().parent / "fixtures" / "public_fixture_corpus.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert "不是真实论文" in payload["fixture_notice"]
    assert len(payload["records"]) >= 2
    for record in payload["records"]:
        assert record["fixture_id"].startswith("fixture-")
        assert "SYNTHETIC TEST FIXTURE TEXT ONLY" in record["evidence"]
