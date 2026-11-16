"""Copy an explicit, privacy-safe competition source staging tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROOT_FILES = (
    "README.md",
    "LICENSE",
    "requirements.txt",
    "requirements-dev.txt",
    "pytest.ini",
    ".env.example",
    ".gitignore",
    "app.py",
    "config.py",
    "main.py",
    "技术文档.md",
)

INCLUDED_DIRECTORIES = (
    "agents",
    "researchmind",
    "services",
    "tools",
    "tests",
    "docs",
    "demo",
    "assets",
    ".streamlit",
)

IGNORED_NAMES = {
    ".env",
    "__pycache__",
    ".pytest_cache",
    ".cache",
    ".tmp",
    ".venv",
    "backups",
    "technical_document_backups",
    "pytest-full-verbose.txt",
    "pytest-results.xml",
    "test-evidence.json",
    "papers-39.csv",
    "papers-41-all-database.csv",
    "evidence-3281-core-domain.csv",
    "evidence-3296-all-database.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ignore_generated(directory: str, names: list[str]) -> set[str]:
    ignored = {
        name
        for name in names
        if name in IGNORED_NAMES
        or name.endswith((".pyc", ".pyo", ".log", ".sqlite", ".sqlite3"))
    }
    return ignored


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(
            f"输出目录已存在，请使用新的目录名：{output}"
        )
    output.mkdir(parents=True)

    copied: list[Path] = []
    for name in ROOT_FILES:
        source = PROJECT_ROOT / name
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)

    for relative in INCLUDED_DIRECTORIES:
        source = PROJECT_ROOT / relative
        destination = output / relative
        if not source.is_dir():
            raise FileNotFoundError(source)
        shutil.copytree(
            source,
            destination,
            ignore=ignore_generated,
        )
        copied.extend(path for path in destination.rglob("*") if path.is_file())

    for name in (
        "knowledge_base/README.md",
        "knowledge_base/manifest.json",
        "knowledge_base/framework.json",
        "knowledge_base_adolescent/README.md",
        "data/README.md",
        "outputs/README.md",
    ):
        source = PROJECT_ROOT / name
        if not source.is_file():
            continue
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)

    public_pdf_dir = PROJECT_ROOT / "knowledge_base" / "pdfs"
    if public_pdf_dir.is_dir():
        destination = output / "knowledge_base" / "pdfs"
        shutil.copytree(
            public_pdf_dir,
            destination,
            ignore=ignore_generated,
        )
        copied.extend(path for path in destination.rglob("*") if path.is_file())

    public_index_dir = PROJECT_ROOT / "knowledge_base" / "index"
    if public_index_dir.is_dir():
        destination = output / "knowledge_base" / "index"
        shutil.copytree(
            public_index_dir,
            destination,
            ignore=ignore_generated,
        )
        copied.extend(path for path in destination.rglob("*") if path.is_file())

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file_count": len(copied),
        "files": {
            path.relative_to(output).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(copied)
        },
    }
    manifest_path = output / "SUBMISSION_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "file_count": len(copied),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
