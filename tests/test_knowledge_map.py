"""五层知识图谱的数据、证据和报告测试。"""

from __future__ import annotations

from researchmind.knowledge_map import (
    CORE_DIRECTION,
    CORE_DIRECTION_ID,
    THEMES,
    build_knowledge_map,
    build_markdown_report,
)
from researchmind.models import EvidenceRef, PaperAnalysis, PaperRecord


def _evidence(
    paper_id: str,
    title: str,
    chunk_id: str,
    page: int,
    quote: str,
) -> EvidenceRef:
    return EvidenceRef(
        chunk_id=chunk_id,
        document_id=paper_id,
        document_title=title,
        file_name=f"{paper_id}.pdf",
        page_number=page,
        quote=quote,
    )


def _paper(
    paper_id: str,
    title: str,
    *,
    abstract: str = "",
    doi: str = "",
) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        domain_id="domain-test",
        title=title,
        authors=("测试作者",),
        year="2025",
        journal="测试期刊",
        doi=doi,
        source_url="",
        abstract=abstract,
        keywords=(),
        pdf_path=f"knowledge_base/pdfs/{paper_id}.pdf",
        source="测试数据",
        verification_status="测试",
        created_at="2025-01-01T00:00:00+00:00",
    )


def test_college_papers_do_not_enter_adolescent_themes() -> None:
    paper = _paper(
        "paper-college",
        "大学生体育锻炼行为研究",
        abstract="研究对象为某高校大学生，调查体育锻炼行为。",
    )
    evidence = _evidence(
        paper.paper_id,
        paper.title,
        "chunk-college",
        2,
        "研究对象为某高校大学生，共 320 人。",
    )
    analysis = PaperAnalysis(
        paper_id=paper.paper_id,
        population=(evidence,),
        status="已提取",
    )

    knowledge_map = build_knowledge_map(
        domain_name="测试领域",
        papers=(paper,),
        analyses={paper.paper_id: analysis},
        evidence_by_paper={paper.paper_id: (evidence,)},
    )

    assert len(knowledge_map.default_nodes) == 7
    assert all(node.level <= 2 for node in knowledge_map.default_nodes)
    assert not knowledge_map.in_scope_papers
    assert knowledge_map.out_of_scope_papers[0].paper_id == paper.paper_id
    assert "大学生" in knowledge_map.out_of_scope_papers[0].scope_reason
    assert all(stat.paper_count == 0 for stat in knowledge_map.theme_statistics)


def test_adolescent_paper_builds_traceable_four_level_branches() -> None:
    paper = _paper(
        "paper-teen",
        "青少年体质健康与不同强度有氧运动干预研究",
        abstract="针对超重青少年开展随机对照运动干预，评价身体素质和肥胖指标。",
        doi="10.0000/test.2025.001",
    )
    population = _evidence(
        paper.paper_id,
        paper.title,
        "chunk-population",
        2,
        "研究对象为 12 至 15 岁青少年，包含超重和肥胖学生。",
    )
    methods = _evidence(
        paper.paper_id,
        paper.title,
        "chunk-methods",
        3,
        "采用随机对照试验，比较中等强度有氧运动处方与常规体育课。",
    )
    results = _evidence(
        paper.paper_id,
        paper.title,
        "chunk-results",
        5,
        "干预后身体素质、心肺适能和体重指数产生变化，采用统一测量指标和问卷评价。",
    )
    conclusions = _evidence(
        paper.paper_id,
        paper.title,
        "chunk-conclusions",
        6,
        "中等强度有氧运动可能改善青少年体质健康，但长期效果仍需追踪。",
    )
    limitations = _evidence(
        paper.paper_id,
        paper.title,
        "chunk-limitations",
        7,
        "本研究干预周期较短，样本来自单一学校。",
    )
    analysis = PaperAnalysis(
        paper_id=paper.paper_id,
        population=(population,),
        methods=(methods,),
        results=(results,),
        conclusions=(conclusions,),
        limitations=(limitations,),
        status="已提取",
    )

    knowledge_map = build_knowledge_map(
        domain_name="测试领域",
        papers=(paper,),
        analyses={paper.paper_id: analysis},
        evidence_by_paper={
            paper.paper_id: (
                population,
                methods,
                results,
                conclusions,
                limitations,
            )
        },
    )

    assert len(knowledge_map.in_scope_papers) == 1
    assert len(knowledge_map.default_nodes) == 7
    assert len(knowledge_map.nodes) > len(knowledge_map.default_nodes)
    assert any(node.level == 3 for node in knowledge_map.nodes)
    assert any(node.level == 4 for node in knowledge_map.nodes)
    assert any(node.node_type == "研究方向" for node in knowledge_map.nodes)
    assert any(node.node_type == "研究变量" for node in knowledge_map.nodes)
    assert all(node.level == 3 for node in knowledge_map.variable_nodes)
    assert any(node.level == 5 for node in knowledge_map.nodes)
    assert knowledge_map.graph_statistics.synthesis_node_count > 0
    assert {
        "心肺耐力",
        "BMI",
    }.issubset({node.label for node in knowledge_map.variable_nodes})
    assert {
        "theme:health-assessment",
        "theme:exercise-intervention",
        "theme:exercise-prescription",
        "theme:outcome-evaluation",
    }.issubset(knowledge_map.in_scope_papers[0].theme_ids)
    assert all(edge.evidence for edge in knowledge_map.edges)
    assert all(
        edge.paper_ids
        for edge in knowledge_map.edges
        if edge.edge_kind == "hierarchy"
    )
    paper_node = next(
        node for node in knowledge_map.nodes if node.node_type == "论文"
    )
    assert paper_node.parent_ids
    assert paper_node.evidence


def test_research_variable_relationship_requires_result_evidence() -> None:
    paper = _paper(
        "paper-relation",
        "青少年有氧运动干预与心肺耐力随机对照研究",
        abstract="研究有氧运动对青少年心肺耐力的影响。",
    )
    population = _evidence(
        paper.paper_id,
        paper.title,
        "relation-population",
        2,
        "研究对象为 120 名青少年。",
    )
    methods = _evidence(
        paper.paper_id,
        paper.title,
        "relation-method",
        3,
        "采用有氧运动干预，并测量心肺耐力和最大摄氧量。",
    )
    results = _evidence(
        paper.paper_id,
        paper.title,
        "relation-result",
        5,
        "有氧运动显著改善心肺耐力并提高最大摄氧量（p < 0.05）。",
    )
    conclusions = _evidence(
        paper.paper_id,
        paper.title,
        "relation-conclusion",
        6,
        "结论认为有氧运动与青少年心肺耐力改善呈正向关联。",
    )
    analysis = PaperAnalysis(
        paper_id=paper.paper_id,
        population=(population,),
        methods=(methods,),
        results=(results,),
        conclusions=(conclusions,),
        status="已提取",
    )

    knowledge_map = build_knowledge_map(
        domain_name="测试领域",
        papers=(paper,),
        analyses={paper.paper_id: analysis},
        evidence_by_paper={
            paper.paper_id: (
                population,
                methods,
                results,
                conclusions,
            )
        },
    )

    relation = next(
        edge
        for edge in knowledge_map.edges
        if edge.edge_kind == "research_relation"
        and edge.source_id
        == next(
            node.node_id
            for node in knowledge_map.variable_nodes
            if node.label == "体育锻炼"
        )
        and edge.target_id
        == next(
            node.node_id
            for node in knowledge_map.variable_nodes
            if node.label == "心肺耐力"
        )
    )
    assert relation.relation_status == "正向关联"
    assert relation.evidence
    assert relation.paper_ids == (paper.paper_id,)
    assert "不自动解释为因果" in relation.relation


def test_graph_recomputes_when_a_paper_is_removed() -> None:
    first = _paper(
        "paper-remove-a",
        "青少年体育锻炼有氧运动干预与心肺耐力研究",
        abstract="有氧运动改善青少年心肺耐力。",
    )
    second = _paper(
        "paper-remove-b",
        "青少年体育锻炼有氧运动干预与心肺耐力系统综述",
        abstract="有氧运动与心肺耐力改善相关。",
    )
    evidence_by_paper = {}
    analyses = {}
    for paper in (first, second):
        evidence = _evidence(
            paper.paper_id,
            paper.title,
            f"{paper.paper_id}-result",
            5,
            "有氧运动显著改善心肺耐力（p < 0.05）。",
        )
        evidence_by_paper[paper.paper_id] = (evidence,)
        analyses[paper.paper_id] = PaperAnalysis(
            paper_id=paper.paper_id,
            results=(evidence,),
            status="已提取",
        )

    full_map = build_knowledge_map(
        domain_name="测试领域",
        papers=(first, second),
        analyses=analyses,
        evidence_by_paper=evidence_by_paper,
    )
    reduced_map = build_knowledge_map(
        domain_name="测试领域",
        papers=(first,),
        analyses={first.paper_id: analyses[first.paper_id]},
        evidence_by_paper={first.paper_id: evidence_by_paper[first.paper_id]},
    )
    cardio_label = "心肺耐力"
    full_direction = next(
        node for node in full_map.direction_nodes if node.label == cardio_label
    )
    reduced_direction = next(
        node for node in reduced_map.direction_nodes if node.label == cardio_label
    )

    assert full_direction.paper_count == 2
    assert reduced_direction.paper_count == 1
    assert reduced_map.graph_statistics.paper_node_count == 1
    assert f"paper:{second.paper_id}" not in {
        node.node_id for node in reduced_map.nodes
    }


def test_report_marks_missing_evidence_and_keeps_real_metadata() -> None:
    paper = _paper(
        "paper-college",
        "大学生运动干预测试论文",
        doi="10.0000/college.2025.001",
    )
    evidence = _evidence(
        paper.paper_id,
        paper.title,
        "chunk-college",
        2,
        "研究对象为大学生。",
    )
    knowledge_map = build_knowledge_map(
        domain_name="测试领域",
        papers=(paper,),
        analyses={
            paper.paper_id: PaperAnalysis(
                paper_id=paper.paper_id,
                population=(evidence,),
                status="已提取",
            )
        },
        evidence_by_paper={paper.paper_id: (evidence,)},
    )

    report = build_markdown_report(knowledge_map)

    assert CORE_DIRECTION in report
    assert paper.title in report
    assert paper.doi in report
    assert "证据不足，待进一步验证" in report
    assert "不能据此断定整个" in report
    assert "10.9999/fabricated" not in report


def test_report_contains_six_theme_evidence_sections() -> None:
    knowledge_map = build_knowledge_map(
        domain_name="空专题测试",
        papers=(),
        analyses={},
        evidence_by_paper={},
    )

    report = build_markdown_report(knowledge_map)

    assert "六大研究主题证据摘要" in report
    for theme in THEMES:
        assert theme.label in report
    assert "证据不足，暂不能判断" in report


def test_empty_themes_keep_structure_without_fabricated_relationships() -> None:
    knowledge_map = build_knowledge_map(
        domain_name="空专题测试",
        papers=(),
        analyses={},
        evidence_by_paper={},
    )

    assert len(knowledge_map.default_nodes) == 1 + len(THEMES)
    assert all(
        node.parent_ids == (CORE_DIRECTION_ID,)
        for node in knowledge_map.default_nodes
        if node.node_type == "研究主题"
    )
    assert knowledge_map.edges == ()
