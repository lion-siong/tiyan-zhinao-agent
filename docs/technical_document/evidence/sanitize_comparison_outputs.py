"""Replace full copyrighted evidence quotes in saved experiment inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "controversy-comparison"


def main() -> int:
    count = 0
    for path in ROOT.glob("question-*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("input_evidence", []):
            if "quote" not in item:
                continue
            quote = str(item.pop("quote"))
            item["quote_sha256"] = hashlib.sha256(
                quote.encode("utf-8")
            ).hexdigest()
            item["quote_preview"] = quote[:160]
            count += 1
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"sanitized_evidence_records={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
