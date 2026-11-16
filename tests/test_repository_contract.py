"""Repository-level contracts required by the competition submission."""

from __future__ import annotations

import re
from pathlib import Path

from tools.prepare_submission import (
    INCLUDED_DIRECTORIES,
    IGNORED_NAMES,
    ROOT_FILES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_competition_documents_and_readme_links_exist() -> None:
    required = (
        "README.md",
        "requirements.txt",
        "requirements-dev.txt",
        ".env.example",
        ".gitignore",
        "LICENSE",
        "pytest.ini",
        "docs/ARCHITECTURE.md",
        "docs/DEPLOYMENT.md",
        "docs/API.md",
        "docs/FUNCTIONALITY.md",
        "docs/CODEBASE_AUDIT_REPORT.md",
        "docs/FINAL_ACCEPTANCE_REPORT.md",
        "docs/COMPETITION_SUBMISSION_CHECKLIST.md",
        "demo/README.md",
    )
    assert all((PROJECT_ROOT / name).is_file() for name in required)

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", readme)
    missing = []
    for target in links:
        clean = target.strip().split("#", 1)[0]
        if not clean or "://" in clean or clean.startswith("mailto:"):
            continue
        if not (PROJECT_ROOT / clean).resolve().exists():
            missing.append(target)
    assert not missing, f"README 本地链接失效：{missing}"


def test_env_example_has_no_nonempty_secret_values() -> None:
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    secret_names = (
        "OPENAI_API_KEY",
        "SEMANTIC_SCHOLAR_API_KEY",
        "PUBMED_API_KEY",
        "ZOTERO_API_KEY",
    )
    for name in secret_names:
        match = re.search(rf"^{name}=(.+)$", text, flags=re.MULTILINE)
        assert match is None or not match.group(1).strip(), (
            f"{name} 在示例配置中不得包含真实值"
        )

    assert ".env.example" in ROOT_FILES
    assert ".env" not in ROOT_FILES
    assert ".env" in IGNORED_NAMES
    assert "docs" in INCLUDED_DIRECTORIES
