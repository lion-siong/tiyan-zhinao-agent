"""青少年专题论文库的隔离、备份和完整恢复测试。"""

from __future__ import annotations

import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from config import Settings
from researchmind.adolescent_library import (
    create_adolescent_workflow,
    create_library_backup,
    list_library_backups,
    restore_library_backup,
)
from researchmind.models import PaperRecord, ResearchDomain
from tools.knowledge_base import INDEX_FILE_PATH


def _settings(
    tmp_path: Path,
    *,
    legacy_db: Path | None = None,
    topic_db: Path | None = None,
    topic_kb: Path | None = None,
) -> Settings:
    return Settings.from_env(
        {
            "RESEARCHMIND_DB_PATH": str(
                legacy_db or tmp_path / "legacy" / "researchmind.sqlite3"
            ),
            "ADOLESCENT_RESEARCHMIND_DB_PATH": str(
                topic_db or tmp_path / "topic" / "researchmind.sqlite3"
            ),
            "ADOLESCENT_KNOWLEDGE_BASE_DIR": str(
                topic_kb or tmp_path / "topic-kb"
            ),
            "OPENAI_API_KEY": "",
        }
    )


def _test_paper(domain: ResearchDomain, pdf_path: Path) -> PaperRecord:
    return PaperRecord(
        paper_id="paper-topic-test",
        domain_id=domain.domain_id,
        title="青少年体育锻炼干预测试论文",
        authors=("测试作者",),
        year="2026",
        journal="测试期刊",
        doi="10.0000/topic-test.2026.001",
        source_url="https://example.org/topic-test",
        abstract="该记录仅用于自动化测试，不代表真实学术论文。",
        keywords=("青少年体质健康", "体育锻炼干预"),
        pdf_path=str(pdf_path),
        source="自动化测试",
        verification_status="测试记录",
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_topic_library_is_isolated_from_legacy_library(tmp_path: Path) -> None:
    config = _settings(tmp_path)
    workflow = create_adolescent_workflow(config)

    assert workflow.store.db_path == config.adolescent_topic_db_path.resolve()
    assert workflow.knowledge_base_dir == (
        config.adolescent_topic_knowledge_base_dir.resolve()
    )
    assert workflow.store.db_path != config.legacy_db_path.resolve()
    assert workflow.knowledge_base_dir != config.legacy_knowledge_base_dir.resolve()
    assert workflow.store.db_path.is_file()
    assert (workflow.knowledge_base_dir / "pdfs").is_dir()
    assert (workflow.knowledge_base_dir / "index").is_dir()
    assert (workflow.knowledge_base_dir / "chroma").is_dir()
    assert (workflow.knowledge_base_dir / "backups").is_dir()
    assert not config.legacy_db_path.exists()


@pytest.mark.parametrize("crossing", ["database", "knowledge_base"])
def test_topic_library_rejects_crossed_legacy_paths(
    tmp_path: Path,
    crossing: str,
) -> None:
    legacy_db = tmp_path / "legacy" / "researchmind.sqlite3"
    if crossing == "database":
        config = _settings(tmp_path, topic_db=legacy_db)
    else:
        config = _settings(
            tmp_path,
            legacy_db=legacy_db,
            topic_kb=legacy_db.parent / "topic-kb",
        )

    with pytest.raises(ValueError, match="不能|交叉"):
        create_adolescent_workflow(config)


def test_backup_restore_rebuilds_a_complete_topic_library_mirror(
    tmp_path: Path,
) -> None:
    workflow = create_adolescent_workflow(_settings(tmp_path))
    domain = workflow.ensure_default_domain()
    root = workflow.knowledge_base_dir

    pdf_path = root / "pdfs" / "topic-test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test fixture")
    index_path = root / INDEX_FILE_PATH
    index_path.write_bytes(b'{"fixture": "original-index"}')
    chroma_path = root / "chroma" / "chroma.sqlite3"
    chroma_path.write_bytes(b"original-chroma")
    original_paper = _test_paper(domain, pdf_path)
    workflow.store.upsert_paper(original_paper)

    backup = create_library_backup(workflow, reason="自动化恢复测试")
    assert backup.path.is_file()
    with zipfile.ZipFile(backup.path) as archive:
        names = set(archive.namelist())
    assert "database/researchmind.sqlite3" in names
    assert "knowledge_base/pdfs/topic-test.pdf" in names
    assert "knowledge_base/index/evidence_chunks.json" in names
    assert "knowledge_base/chroma/chroma.sqlite3" in names
    assert not any("/backups/" in name for name in names)

    pdf_path.unlink()
    index_path.write_bytes(b'{"fixture": "changed"}')
    chroma_path.write_bytes(b"changed-chroma")
    (root / "pdfs" / "orphan.pdf").write_bytes(b"orphan")
    (root / "chroma" / "orphan.bin").write_bytes(b"orphan")
    workflow.store.upsert_paper(
        replace(original_paper, title="恢复前被修改的标题")
    )

    safety_backup = restore_library_backup(workflow, backup.path)

    assert safety_backup.path.is_file()
    assert safety_backup.reason == "恢复前自动备份"
    assert len(list_library_backups(workflow)) >= 2
    assert workflow.store.get_paper(original_paper.paper_id) == original_paper
    assert pdf_path.read_bytes() == b"%PDF-1.4 test fixture"
    assert index_path.read_bytes() == b'{"fixture": "original-index"}'
    assert chroma_path.read_bytes() == b"original-chroma"
    assert not (root / "pdfs" / "orphan.pdf").exists()
    assert not (root / "chroma" / "orphan.bin").exists()
