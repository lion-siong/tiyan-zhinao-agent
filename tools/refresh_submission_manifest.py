"""Refresh the exported submission manifest after acceptance checks."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(
            part in {
                ".git",
                ".venv",
                "__pycache__",
                ".pytest_cache",
                ".cache",
                ".tmp",
            }
            for part in relative.parts
        ):
            continue
        if not path.is_file() or path.name == "SUBMISSION_MANIFEST.json":
            continue
        if path.suffix.casefold() in {".pyc", ".pyo", ".log"}:
            continue
        files[relative.as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    acceptance_path = root / "SUBMISSION_ACCEPTANCE.json"
    acceptance = (
        json.loads(acceptance_path.read_text(encoding="utf-8"))
        if acceptance_path.is_file()
        else {}
    )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file_count": len(files),
        "source_tree_sha256": acceptance.get("source_tree_sha256", ""),
        "acceptance_passed": acceptance.get("passed", False),
        "files": files,
    }
    (root / "SUBMISSION_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "root": str(root),
                "file_count": len(files),
                "source_tree_sha256": manifest["source_tree_sha256"],
                "acceptance_passed": manifest["acceptance_passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
