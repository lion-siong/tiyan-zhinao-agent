"""论文库指纹、派生分析和自动更新状态测试。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time

from config import Settings
from researchmind.library_state import (
    get_analysis_state,
    graph_fingerprint,
    library_fingerprint,
    refresh_domain_analysis_now,
    schedule_analysis_refresh,
)
from researchmind.models import EvidenceRef, PaperRecord
from researchmind.workflow import ResearchMindWorkflow
from researchmind.storage import ResearchMindStore


def _workflow(tmp_path: Path) -> ResearchMindWorkflow:
    config = Settings.from_env(
        {
            "RESEARCHMIND_DB_PATH": str(tmp_path / "legacy.sqlite3"),
            "ADOLESCENT_RESEARCHMIND_DB_PATH": str(
                tmp_path / "topic.sqlite3"
            ),
            "ADOLESCENT_KNOWLEDGE_BASE_DIR": str(tmp_path / "kb"),
            "OPENAI_API_KEY": "",
        }
    )
    return ResearchMindWorkflow(
        store=ResearchMindStore(config.adolescent_topic_db_path),
        config=config,
        knowledge_base_dir=tmp_path / "kb",
        auto_sync_bundled=False,
        default_debate_topic="青少年体育锻炼干预与体质健康",
    )


def _paper(domain_id: str, suffix: str) -> PaperRecord:
    return PaperRecord(
        paper_id=f"paper-{suffix}",
        domain_id=domain_id,
        title=f"青少年体育锻炼干预证据测试 {suffix}",
        authors=("Test Author",),
        year="2025",
        journal="Test Journal",
        doi=f"10.0000/state.{suffix}",
        source_url="https://example.org/state",
        abstract="青少年体育锻炼干预与体质健康研究。",
        keywords=("青少年", "体育锻炼", "体质健康"),
        pdf_path=f"{suffix}.pdf",
        source="状态回归测试",
        verification_status="测试记录",
        created_at="2026-01-01T00:00:00+00:00",
    )


def _evidence(paper: PaperRecord) -> EvidenceRef:
    return EvidenceRef(
        chunk_id=f"{paper.paper_id}-chunk",
        document_id=paper.paper_id,
        document_title=paper.title,
        file_name=f"{paper.paper_id}.pdf",
        page_number=2,
        quote=(
            "研究对象为青少年。采用随机对照试验比较体育锻炼干预，"
            "结果显示体质健康指标得到显著改善。结论认为需要进一步复核长期效果。"
        ),
    )


def test_import_evidence_and_delete_invalidate_analysis(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)
    domain = workflow.ensure_default_domain()
    first = _paper(domain.domain_id, "a")
    second = _paper(domain.domain_id, "b")

    workflow.store.upsert_paper(first)
    workflow.store.replace_evidence(first.paper_id, (_evidence(first),))
    initial_version = library_fingerprint(workflow.store, domain.domain_id)
    initial = get_analysis_state(
        workflow.store,
        domain.domain_id,
        workflow.default_debate_topic,
    )
    assert initial.status in {"待更新", "分析中"}
    assert not initial.is_current

    refreshed = refresh_domain_analysis_now(
        workflow,
        domain,
        workflow.default_debate_topic,
    )
    assert refreshed.status == "已更新"
    assert refreshed.is_current
    assert refreshed.latest_paper_included
    assert refreshed.analyzed_paper_ids == (first.paper_id,)
    assert graph_fingerprint(
        workflow.store,
        domain.domain_id,
    )

    workflow.store.upsert_paper(second)
    workflow.store.replace_evidence(second.paper_id, (_evidence(second),))
    changed_version = library_fingerprint(workflow.store, domain.domain_id)
    stale = get_analysis_state(
        workflow.store,
        domain.domain_id,
        workflow.default_debate_topic,
    )
    assert changed_version != initial_version
    assert stale.status == "待更新"
    assert not stale.is_current
    assert workflow.store.latest_debate(domain.domain_id) is None

    updated = refresh_domain_analysis_now(
        workflow,
        domain,
        workflow.default_debate_topic,
    )
    debate = workflow.store.latest_debate(domain.domain_id)
    synthesis = workflow.store.get_latest_synthesis(domain.domain_id)
    assert updated.status == "已更新"
    assert debate is not None
    assert synthesis is not None
    assert set(debate.considered_paper_ids) == {
        first.paper_id,
        second.paper_id,
    }
    assert set(synthesis.source_paper_ids) == {
        first.paper_id,
        second.paper_id,
    }

    workflow.store.soft_delete_papers(
        domain.domain_id,
        (second.paper_id,),
    )
    after_delete = get_analysis_state(
        workflow.store,
        domain.domain_id,
        workflow.default_debate_topic,
    )
    assert after_delete.status == "待更新"
    assert workflow.store.latest_debate(domain.domain_id) is None

    final = refresh_domain_analysis_now(
        workflow,
        domain,
        workflow.default_debate_topic,
    )
    synthesis = workflow.store.get_latest_synthesis(domain.domain_id)
    assert final.status == "已更新"
    assert final.paper_count == 1
    assert synthesis is not None
    assert synthesis.source_paper_ids == (first.paper_id,)


def test_debate_coverage_and_claim_origins(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)
    domain = workflow.ensure_default_domain()
    papers = (_paper(domain.domain_id, "a"), _paper(domain.domain_id, "b"))
    for paper in papers:
        workflow.store.upsert_paper(paper)
        workflow.store.replace_evidence(paper.paper_id, (_evidence(paper),))

    refresh_domain_analysis_now(
        workflow,
        domain,
        workflow.default_debate_topic,
    )
    debate = workflow.store.latest_debate(domain.domain_id)

    assert debate is not None
    assert set(debate.considered_paper_ids) == {
        paper.paper_id for paper in papers
    }
    assert set(debate.evidence_covered_paper_ids) == {
        paper.paper_id for paper in papers
    }
    assert all(viewpoint.claim_origin == "论文原文摘录" for viewpoint in debate.viewpoints)
    assert all(
        "规则推断" in viewpoint.stance_origin
        for viewpoint in debate.viewpoints
    )


def test_metadata_update_changes_fingerprint_without_deleting_paper(
    tmp_path: Path,
) -> None:
    workflow = _workflow(tmp_path)
    domain = workflow.ensure_default_domain()
    paper = _paper(domain.domain_id, "metadata")
    workflow.store.upsert_paper(paper)
    before = library_fingerprint(workflow.store, domain.domain_id)

    workflow.store.upsert_paper(
        replace(paper, title="修改后的青少年体育锻炼干预论文题名")
    )
    after = library_fingerprint(workflow.store, domain.domain_id)

    assert before != after
    assert workflow.store.get_paper(paper.paper_id) is not None
    assert get_analysis_state(
        workflow.store,
        domain.domain_id,
        workflow.default_debate_topic,
    ).status == "待更新"


def test_background_schedule_updates_persisted_state(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)
    domain = workflow.ensure_default_domain()
    paper = _paper(domain.domain_id, "background")
    workflow.store.upsert_paper(paper)
    workflow.store.replace_evidence(paper.paper_id, (_evidence(paper),))

    assert schedule_analysis_refresh(
        workflow,
        domain,
        workflow.default_debate_topic,
    )

    deadline = time.monotonic() + 12
    state = get_analysis_state(
        workflow.store,
        domain.domain_id,
        workflow.default_debate_topic,
    )
    while time.monotonic() < deadline:
        state = get_analysis_state(
            workflow.store,
            domain.domain_id,
            workflow.default_debate_topic,
        )
        if state.is_current or state.status == "失败":
            break
        time.sleep(0.1)

    assert state.status == "已更新"
    assert state.is_current
    assert state.paper_count == 1
    assert state.analyzed_paper_ids == (paper.paper_id,)
    assert not schedule_analysis_refresh(
        workflow,
        domain,
        workflow.default_debate_topic,
    )
