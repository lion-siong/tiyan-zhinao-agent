"""学术共识与争议升级的可追溯逻辑测试。

测试论文和结果句均明确标注为自动化夹具，只写入 pytest 临时数据库。
"""

from __future__ import annotations

from pathlib import Path

from researchmind.agents import AcademicDebateAnalyzer, PaperReader
from researchmind.evidence_quality import (
    RESEARCH_QUESTION_BY_ID,
    evidence_quality_flags,
)
from researchmind.models import EvidenceRef, PaperRecord, ResearchDomain
from researchmind.storage import ResearchMindStore


def _domain() -> ResearchDomain:
    return ResearchDomain.create(
        "证据综合测试领域",
        "自动化测试夹具。",
        ("青少年", "体育锻炼"),
        "2026-11-15T00:00:00+00:00",
    )


def _paper(
    paper_id: str,
    domain_id: str,
    title: str,
    abstract: str,
) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        domain_id=domain_id,
        title=title,
        authors=("Fixture Author",),
        year="2025",
        journal="Automated Test Fixture",
        doi=f"10.0000/{paper_id}",
        source_url=f"https://example.org/{paper_id}",
        abstract=abstract,
        keywords=("青少年", "体育锻炼", "体质健康"),
        pdf_path=f"{paper_id}.pdf",
        source="自动化测试夹具",
        verification_status="测试夹具",
        created_at="2026-11-15T00:00:00+00:00",
        parse_status="已解析",
        analysis_status="未分析",
    )


def _evidence(
    paper: PaperRecord,
    quote: str,
    *,
    chunk_id: str,
    page: int = 3,
) -> EvidenceRef:
    return EvidenceRef(
        chunk_id=chunk_id,
        document_id=paper.paper_id,
        document_title=paper.title,
        file_name=paper.pdf_path,
        page_number=page,
        quote=quote,
        source_type="全文",
        section_hint="result",
    )


def test_copyright_and_reference_text_are_not_research_evidence() -> None:
    copyright_text = (
        "Open Access © The Author(s) 2025. This article is licensed under a "
        "Creative Commons Attribution 4.0 International License. "
        "The intervention significantly improved physical fitness."
    )
    reference_text = (
        "References [1] Smith 2020; [2] Jones 2021; [3] Lee 2022; "
        "[4] Wang 2023. doi:10.1/a doi:10.1/b doi:10.1/c"
    )

    assert evidence_quality_flags(copyright_text)[0] is False
    assert evidence_quality_flags(reference_text)[0] is False


def test_two_same_direction_papers_form_grouped_consensus(tmp_path: Path) -> None:
    store = ResearchMindStore(tmp_path / "consensus.sqlite3")
    domain = _domain()
    store.upsert_domain(domain)
    papers = (
        _paper(
            "fixture-a",
            domain.domain_id,
            "Adolescent physical activity and body composition: a systematic review and meta-analysis",
            "本研究评估体育锻炼与青少年身体成分。",
        ),
        _paper(
            "fixture-b",
            domain.domain_id,
            "Exercise training for body composition in adolescents: a meta-analysis",
            "本研究评估运动训练与青少年身体成分。",
        ),
    )
    for paper in papers:
        store.upsert_paper(paper)
    store.replace_evidence(
        "fixture-a",
        (
            _evidence(
                papers[0],
                "The exercise intervention significantly reduced body fat and "
                "improved body composition in adolescents (p < 0.05).",
                chunk_id="fixture-a-result",
            ),
        ),
    )
    store.replace_evidence(
        "fixture-b",
        (
            _evidence(
                papers[1],
                "Exercise training reduced body fat and improved body composition "
                "in the adolescent sample (p = 0.01).",
                chunk_id="fixture-b-result",
            ),
        ),
    )

    debate = AcademicDebateAnalyzer(store).analyze(
        domain.domain_id,
        "青少年体育锻炼与体质健康",
        save=False,
    )
    body = next(
        item
        for item in debate.question_syntheses
        if item.question_id == "question:body-composition"
    )

    assert body.supporting_paper_ids == ("fixture-a", "fixture-b")
    assert body.status in {"初步共识", "有一致研究发现"}
    assert debate.conflicts == ()
    assert any(
        set(viewpoint.supporting_papers) == {"fixture-a", "fixture-b"}
        for viewpoint in debate.viewpoints
    )


def test_opposing_same_question_results_form_one_traceable_conflict(
    tmp_path: Path,
) -> None:
    store = ResearchMindStore(tmp_path / "conflict.sqlite3")
    domain = _domain()
    store.upsert_domain(domain)
    positive = _paper(
        "fixture-positive",
        domain.domain_id,
        "Exercise intervention and body composition in adolescents: a randomized controlled trial",
        "随机对照研究评估青少年身体成分。",
    )
    negative = _paper(
        "fixture-negative",
        domain.domain_id,
        "Physical activity and body composition among adolescents: a cross-sectional study",
        "横断面研究评估青少年身体成分。",
    )
    for paper in (positive, negative):
        store.upsert_paper(paper)
    store.replace_evidence(
        positive.paper_id,
        (
            _evidence(
                positive,
                "The intervention significantly reduced body fat and improved "
                "body composition (p < 0.05).",
                chunk_id="positive-result",
            ),
        ),
    )
    store.replace_evidence(
        negative.paper_id,
        (
            _evidence(
                negative,
                "The exercise group showed increased body fat and worsened body "
                "composition compared with the control group (p = 0.04).",
                chunk_id="negative-result",
            ),
        ),
    )

    debate = AcademicDebateAnalyzer(store).analyze(
        domain.domain_id,
        "青少年体育锻炼与体质健康",
        save=True,
    )
    conflict = next(
        item
        for item in debate.conflicts
        if item.question_id == "question:body-composition"
    )

    assert conflict.paper_ids_a == ("fixture-positive",)
    assert conflict.paper_ids_b == ("fixture-negative",)
    assert conflict.evidence_a
    assert conflict.evidence_b
    assert conflict.review_status == "待体育科研教师复核"
    assert "未达统计显著不视为方向相反" in conflict.significance_note
    assert conflict.divergence_reasons
    assert conflict.current_conclusion
    assert not _question_mismatch(conflict.question_id)

    restored = store.latest_debate(domain.domain_id)
    assert restored is not None
    assert restored.engine_version
    assert restored.question_syntheses
    assert restored.conflicts
    assert restored.conflicts[0].evidence_a


def test_positive_and_non_significant_results_do_not_form_conflict(
    tmp_path: Path,
) -> None:
    store = ResearchMindStore(tmp_path / "null-not-opposing.sqlite3")
    domain = _domain()
    store.upsert_domain(domain)
    positive = _paper(
        "fixture-positive",
        domain.domain_id,
        "Exercise intervention and body composition in adolescents: a randomized controlled trial",
        "随机对照研究评估青少年身体成分。",
    )
    null = _paper(
        "fixture-null",
        domain.domain_id,
        "Physical activity and body composition among adolescents: a cross-sectional study",
        "横断面研究评估青少年身体成分。",
    )
    for paper in (positive, null):
        store.upsert_paper(paper)
    store.replace_evidence(
        positive.paper_id,
        (
            _evidence(
                positive,
                "The intervention significantly reduced body fat and improved "
                "body composition (p < 0.05).",
                chunk_id="positive-result",
            ),
        ),
    )
    store.replace_evidence(
        null.paper_id,
        (
            _evidence(
                null,
                "There was no significant difference in body composition after "
                "exercise training (p = 0.42).",
                chunk_id="null-result",
            ),
        ),
    )

    debate = AcademicDebateAnalyzer(store).analyze(
        domain.domain_id,
        "青少年体育锻炼与体质健康",
        save=False,
    )
    body = next(
        item
        for item in debate.question_syntheses
        if item.question_id == "question:body-composition"
    )

    assert debate.conflicts == ()
    assert body.opposing_paper_ids == ()
    assert body.null_paper_ids == ("fixture-null",)
    assert "fixture-null" in body.inconclusive_paper_ids
    assert "未达统计显著" in body.conclusion


def test_abstract_only_analysis_is_explicitly_labelled(tmp_path: Path) -> None:
    store = ResearchMindStore(tmp_path / "abstract.sqlite3")
    domain = _domain()
    store.upsert_domain(domain)
    paper = _paper(
        "fixture-abstract",
        domain.domain_id,
        "Physical activity and mental health in adolescents: a systematic review",
        (
            "Objective: This systematic review examined physical activity and "
            "mental health in adolescents. Results: Exercise interventions "
            "improved mental health and reduced depression in adolescents. "
            "Conclusion: Physical activity may improve mental health outcomes."
        ),
    )
    store.upsert_paper(paper)

    analysis = PaperReader(store).read_paper(paper.paper_id)

    assert analysis.all_evidence
    assert all(item.source_type == "摘要" for item in analysis.all_evidence)
    assert all(item.location_label.startswith("摘要") for item in analysis.all_evidence)
    assert "摘要" in analysis.extraction_mode
    assert store.get_paper(paper.paper_id).parse_status == "摘要已解析"


def _question_mismatch(question_id: str) -> bool:
    return question_id not in RESEARCH_QUESTION_BY_ID
