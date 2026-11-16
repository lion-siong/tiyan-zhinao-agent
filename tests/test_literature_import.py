"""文献自动采集入库、去重、回收站和永久删除的数据库测试。

测试中的候选论文和研究结论均为明确标注的自动化测试夹具，不代表真实论文。
测试只写入 pytest 临时目录，不会改动正式专题库或旧论文库。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from config import Settings
from researchmind.adolescent_library import create_adolescent_workflow
from researchmind.agents import PaperReader
from researchmind.literature_import import (
    import_candidates,
    permanently_delete_papers,
    restore_papers,
    soft_delete_papers,
)
from researchmind.models import CandidatePaper, EvidenceRef


FIXTURE_NOTE = (
    "自动化测试夹具，仅用于验证数据库流程，不代表真实论文或真实研究结论。"
    "研究对象为青少年，内容涉及体育锻炼干预与体质健康。"
)


def _settings(tmp_path: Path) -> Settings:
    """为每个测试创建与正式库、旧库完全隔离的临时专题库。"""

    return Settings.from_env(
        {
            "RESEARCHMIND_DB_PATH": str(
                tmp_path / "legacy" / "researchmind.sqlite3"
            ),
            "ADOLESCENT_RESEARCHMIND_DB_PATH": str(
                tmp_path / "topic" / "researchmind.sqlite3"
            ),
            "ADOLESCENT_KNOWLEDGE_BASE_DIR": str(tmp_path / "topic-kb"),
            "OPENAI_API_KEY": "",
        }
    )


def _candidate(
    key: str,
    *,
    title: str | None = None,
    doi: str | None = None,
    authors: tuple[str, ...] = ("测试作者",),
    journal: str = "",
    abstract: str = FIXTURE_NOTE,
    keywords: tuple[str, ...] = (
        "青少年",
        "体育锻炼",
        "体质健康",
    ),
) -> CandidatePaper:
    """构造明确标注为测试用途的候选论文。"""

    return CandidatePaper(
        candidate_id=f"fixture-{key}",
        title=title or f"青少年体育锻炼干预自动化测试夹具-{key}",
        authors=authors,
        year="2024",
        journal=journal,
        doi=doi if doi is not None else f"10.0000/fixture.{key}",
        source_url=f"https://example.org/fixture/{key}",
        abstract=abstract,
        keywords=keywords,
        source="自动化测试（非真实学术来源）",
    )


def _evidence(paper_id: str) -> EvidenceRef:
    """创建只用于自动化测试的正文证据。"""

    return EvidenceRef(
        chunk_id=f"{paper_id}-chunk-1",
        document_id=paper_id,
        document_title="自动化测试论文",
        file_name="fixture.pdf",
        page_number=2,
        quote="自动化测试原文片段，仅用于验证页码和证据查询链路。",
    )


def _attach_test_pdf(workflow, paper_id: str) -> Path:
    """为测试论文挂接真实存在的临时 PDF 文件路径。"""

    paper = workflow.store.get_paper(paper_id)
    assert paper is not None
    pdf_path = (
        workflow.knowledge_base_dir
        / "pdfs"
        / f"{paper_id}-fixture.pdf"
    )
    pdf_path.write_bytes(b"%PDF-1.4 automated test fixture")
    workflow.store.upsert_paper(
        replace(paper, pdf_path=str(pdf_path))
    )
    return pdf_path


def test_single_and_batch_import_write_actual_sqlite_rows(
    tmp_path: Path,
) -> None:
    workflow = create_adolescent_workflow(_settings(tmp_path))
    domain = workflow.ensure_default_domain()

    single = import_candidates(
        workflow,
        domain,
        (_candidate("single"),),
    )
    batch = import_candidates(
        workflow,
        domain,
        (_candidate("batch-a"), _candidate("batch-b")),
    )

    assert single.imported == 1
    assert batch.imported == 2
    assert workflow.store.paper_count(domain.domain_id) == 3
    with workflow.store.connection() as connection:
        database_count = connection.execute(
            "SELECT COUNT(*) AS count FROM papers WHERE domain_id = ?",
            (domain.domain_id,),
        ).fetchone()["count"]
    assert database_count == 3
    assert all(
        paper.parse_status == "仅元数据"
        for paper in workflow.store.list_papers(domain.domain_id)
    )


def test_doi_duplicate_is_detected_across_import_calls(
    tmp_path: Path,
) -> None:
    workflow = create_adolescent_workflow(_settings(tmp_path))
    domain = workflow.ensure_default_domain()
    first = _candidate(
        "doi-first",
        title="青少年体育锻炼干预自动化测试夹具-DOI甲",
        doi="10.0000/Fixture.DOI",
    )
    duplicate = _candidate(
        "doi-second",
        title="青少年体育锻炼干预自动化测试夹具-DOI乙",
        doi="https://doi.org/10.0000/fixture.doi.",
    )

    assert import_candidates(workflow, domain, (first,)).imported == 1
    result = import_candidates(workflow, domain, (duplicate,))

    assert result.imported == 0
    assert result.skipped_duplicates == 1
    assert workflow.store.paper_count(domain.domain_id) == 1


def test_normalized_title_duplicate_handles_case_and_punctuation(
    tmp_path: Path,
) -> None:
    workflow = create_adolescent_workflow(_settings(tmp_path))
    domain = workflow.ensure_default_domain()
    first = _candidate(
        "title-first",
        title="青少年体育锻炼干预：自动化测试题名",
        doi="10.0000/fixture.title-first",
    )
    duplicate = _candidate(
        "title-second",
        title=" 青少年体育锻炼干预 - 自动化测试题名 ",
        doi="",
    )

    import_candidates(workflow, domain, (first,))
    result = import_candidates(workflow, domain, (duplicate,))

    assert result.skipped_duplicates == 1
    assert workflow.store.paper_count(domain.domain_id) == 1


def test_duplicate_import_only_fills_empty_metadata(
    tmp_path: Path,
) -> None:
    workflow = create_adolescent_workflow(_settings(tmp_path))
    domain = workflow.ensure_default_domain()
    first = _candidate(
        "merge",
        authors=(),
        journal="",
        abstract="原始测试摘要，不应被重复导入覆盖。",
    )
    import_candidates(workflow, domain, (first,))

    second = replace(
        first,
        candidate_id="fixture-merge-second",
        authors=("补充作者",),
        journal="补充测试期刊",
        abstract="后来得到的摘要不应覆盖已有摘要。",
    )
    result = import_candidates(workflow, domain, (second,))

    paper = workflow.store.list_papers(domain.domain_id)[0]
    assert result.updated == 1
    assert paper.authors == ("补充作者",)
    assert paper.journal == "补充测试期刊"
    assert paper.abstract == "原始测试摘要，不应被重复导入覆盖。"


def test_unrelated_college_learning_paper_is_not_imported(
    tmp_path: Path,
) -> None:
    workflow = create_adolescent_workflow(_settings(tmp_path))
    domain = workflow.ensure_default_domain()
    candidate = _candidate(
        "unrelated",
        title="大学生在线学习行为与自我效能研究",
        abstract="研究对象为大学生，讨论在线学习行为。",
        keywords=("大学生", "在线学习", "自我效能"),
    )

    result = import_candidates(workflow, domain, (candidate,))

    assert result.imported == 0
    assert result.skipped_ineligible == 1
    assert workflow.store.paper_count(domain.domain_id, include_deleted=True) == 0


def test_soft_delete_hides_metadata_and_evidence_then_restores(
    tmp_path: Path,
) -> None:
    workflow = create_adolescent_workflow(_settings(tmp_path))
    domain = workflow.ensure_default_domain()
    paper_id = import_candidates(
        workflow,
        domain,
        (_candidate("recycle"),),
    ).paper_ids[0]
    _attach_test_pdf(workflow, paper_id)
    workflow.store.replace_evidence(paper_id, (_evidence(paper_id),))

    deletion = soft_delete_papers(
        workflow.store,
        domain.domain_id,
        (paper_id,),
    )

    assert deletion.paper_ids == (paper_id,)
    assert workflow.store.list_papers(domain.domain_id) == ()
    assert workflow.store.list_evidence(paper_id) == ()
    assert workflow.store.list_evidence(
        paper_id,
        include_deleted=True,
    )
    assert workflow.store.get_paper(paper_id) is None
    assert workflow.store.get_paper(
        paper_id,
        include_deleted=True,
    ).is_deleted

    restored = restore_papers(
        workflow.store,
        domain.domain_id,
        (paper_id,),
    )

    assert restored == (paper_id,)
    assert [paper.paper_id for paper in workflow.store.list_papers(
        domain.domain_id
    )] == [paper_id]
    assert workflow.store.list_evidence(paper_id)


def test_permanent_delete_requires_recycle_bin_backup_and_tombstone(
    tmp_path: Path,
) -> None:
    workflow = create_adolescent_workflow(_settings(tmp_path))
    domain = workflow.ensure_default_domain()
    paper_id = import_candidates(
        workflow,
        domain,
        (_candidate("permanent"),),
    ).paper_ids[0]

    with pytest.raises(ValueError, match="回收站"):
        permanently_delete_papers(workflow, domain, (paper_id,))

    soft_delete_papers(workflow.store, domain.domain_id, (paper_id,))
    result = permanently_delete_papers(workflow, domain, (paper_id,))

    assert result.paper_ids == (paper_id,)
    assert result.backup.path.is_file()
    assert result.backup.reason == "永久删除前自动备份"
    assert workflow.store.get_paper(
        paper_id,
        include_deleted=True,
    ) is None
    with workflow.store.connection() as connection:
        tombstone = connection.execute(
            """
            SELECT title, reason FROM paper_tombstones
            WHERE paper_id = ?
            """,
            (paper_id,),
        ).fetchone()
    assert tombstone is not None
    assert tombstone["reason"] == "用户在回收站永久删除"


def test_permanent_delete_preserves_pdf_shared_by_another_paper(
    tmp_path: Path,
) -> None:
    workflow = create_adolescent_workflow(_settings(tmp_path))
    domain = workflow.ensure_default_domain()
    paper_ids = import_candidates(
        workflow,
        domain,
        (_candidate("shared-a"), _candidate("shared-b")),
    ).paper_ids
    shared_pdf = workflow.knowledge_base_dir / "pdfs" / "shared-fixture.pdf"
    shared_pdf.write_bytes(b"%PDF-1.4 shared test fixture")
    for paper_id in paper_ids:
        paper = workflow.store.get_paper(paper_id)
        assert paper is not None
        workflow.store.upsert_paper(
            replace(paper, pdf_path=str(shared_pdf))
        )

    soft_delete_papers(workflow.store, domain.domain_id, (paper_ids[0],))
    result = permanently_delete_papers(
        workflow,
        domain,
        (paper_ids[0],),
    )

    assert shared_pdf.is_file()
    assert result.preserved_shared_files == (shared_pdf.name,)
    assert result.removed_pdf_files == ()
    assert workflow.store.get_paper(paper_ids[1]) is not None


def test_permanent_delete_removes_exclusive_pdf_not_in_manifest(
    tmp_path: Path,
) -> None:
    workflow = create_adolescent_workflow(_settings(tmp_path))
    domain = workflow.ensure_default_domain()
    paper_id = import_candidates(
        workflow,
        domain,
        (_candidate("exclusive-pdf"),),
    ).paper_ids[0]
    pdf_path = _attach_test_pdf(workflow, paper_id)

    soft_delete_papers(workflow.store, domain.domain_id, (paper_id,))
    result = permanently_delete_papers(workflow, domain, (paper_id,))

    assert pdf_path.is_file() is False
    assert result.removed_pdf_files == (pdf_path.name,)
    assert result.preserved_shared_files == ()
    assert workflow.store.get_paper(
        paper_id,
        include_deleted=True,
    ) is None
    assert result.backup.path.is_file()


def test_manifest_cleanup_keeps_imported_papers_and_deleted_state(
    tmp_path: Path,
) -> None:
    workflow = create_adolescent_workflow(_settings(tmp_path))
    domain = workflow.ensure_default_domain()
    paper_id = import_candidates(
        workflow,
        domain,
        (_candidate("manifest-safe"),),
    ).paper_ids[0]

    removed = workflow.store.delete_papers_except(
        domain.domain_id,
        keep_paper_ids=(),
    )
    assert removed == 0
    assert workflow.store.get_paper(paper_id) is not None

    soft_delete_papers(workflow.store, domain.domain_id, (paper_id,))
    deleted = workflow.store.get_paper(paper_id, include_deleted=True)
    assert deleted is not None
    workflow.store.upsert_paper(
        replace(
            deleted,
            title="清单同步不应恢复回收站论文",
            is_deleted=False,
            deleted_at="",
        )
    )

    still_deleted = workflow.store.get_paper(
        paper_id,
        include_deleted=True,
    )
    assert still_deleted is not None
    assert still_deleted.is_deleted is True
    assert workflow.store.list_papers(domain.domain_id) == ()


def test_metadata_only_paper_is_never_marked_as_parsed(
    tmp_path: Path,
) -> None:
    workflow = create_adolescent_workflow(_settings(tmp_path))
    domain = workflow.ensure_default_domain()
    metadata_id, full_text_id = import_candidates(
        workflow,
        domain,
        (_candidate("metadata-only"), _candidate("full-text")),
    ).paper_ids
    _attach_test_pdf(workflow, full_text_id)
    workflow.store.replace_evidence(
        full_text_id,
        (_evidence(full_text_id),),
    )
    reader = PaperReader(workflow.store)

    analyses = reader.read_domain(domain.domain_id)

    assert [analysis.paper_id for analysis in analyses] == [full_text_id]
    metadata_paper = workflow.store.get_paper(metadata_id)
    assert metadata_paper is not None
    assert metadata_paper.parse_status == "仅元数据"
    assert workflow.store.get_analysis(metadata_id) is None

    direct = reader.read_paper(metadata_id)
    metadata_paper = workflow.store.get_paper(metadata_id)
    assert "证据不足" in direct.status
    assert metadata_paper is not None
    assert metadata_paper.parse_status == "仅元数据"
    assert metadata_paper.analysis_status == "待补充全文"


def test_pipeline_summary_reports_metadata_only_papers(
    tmp_path: Path,
) -> None:
    workflow = create_adolescent_workflow(_settings(tmp_path))
    domain = workflow.ensure_default_domain()
    import_candidates(
        workflow,
        domain,
        (_candidate("metadata-summary"),),
    )

    pipeline = workflow.run_analysis_pipeline(
        domain.domain_id,
        generate_questions=False,
    )

    assert pipeline.analyses == ()
    assert "完成 0 篇全文结构化阅读" in pipeline.summary
    assert "跳过 1 篇纯题录" in pipeline.summary
