"""知识图谱页面、载荷和导出功能的回归测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from researchmind.knowledge_map import (
    CORE_DIRECTION_ID,
    THEMES,
    build_knowledge_map,
)
from researchmind.knowledge_map_ui import build_graph_payload
from researchmind.models import EvidenceRef, PaperAnalysis, PaperRecord
from researchmind.ui import RESEARCHMIND_PAGES


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _paper(paper_id: str, title: str) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        domain_id="domain-ui-test",
        title=title,
        authors=("测试作者",),
        year="2025",
        journal="测试期刊",
        doi=f"10.0000/{paper_id}",
        source_url="https://example.org/paper",
        abstract="",
        keywords=(),
        pdf_path=f"knowledge_base/pdfs/{paper_id}.pdf",
        source="测试数据",
        verification_status="测试",
        created_at="2025-01-01T00:00:00+00:00",
    )


def _out_of_scope_map():
    paper = _paper("paper-college", "大学生体育锻炼行为研究")
    evidence = _evidence(
        paper.paper_id,
        paper.title,
        "chunk-college",
        2,
        "研究对象为某高校大学生，共 320 人。",
    )
    return build_knowledge_map(
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


def test_knowledge_map_summary_page_is_registered() -> None:
    assert "知识图谱" in RESEARCHMIND_PAGES
    assert "知识图谱总结" in RESEARCHMIND_PAGES
    assert "科研知识地图" not in RESEARCHMIND_PAGES


def test_default_graph_payload_stays_compact_and_excludes_out_of_scope_paper_nodes() -> None:
    payload = build_graph_payload(_out_of_scope_map())

    assert payload["core_id"] == CORE_DIRECTION_ID
    assert len(payload["themes"]) == len(THEMES)
    assert len(payload["nodes"]) == 1 + len(THEMES)
    assert len(payload["nodes"]) <= 20
    assert all(node["level"] <= 2 for node in payload["nodes"])
    assert payload["papers"] == []
    assert all(item["paper_count"] == 0 for item in payload["themes"])


def test_graph_payload_exposes_directions_variables_relations_and_filters() -> None:
    paper = _paper(
        "paper-multilevel",
        "青少年有氧运动干预与心肺耐力随机对照研究",
    )
    population = _evidence(
        paper.paper_id,
        paper.title,
        "multilevel-population",
        2,
        "研究对象为 120 名青少年。",
    )
    methods = _evidence(
        paper.paper_id,
        paper.title,
        "multilevel-methods",
        3,
        "采用有氧运动干预并测量心肺耐力和最大摄氧量。",
    )
    results = _evidence(
        paper.paper_id,
        paper.title,
        "multilevel-results",
        5,
        "有氧运动显著改善心肺耐力并提高最大摄氧量（p < 0.05）。",
    )
    analysis = PaperAnalysis(
        paper_id=paper.paper_id,
        population=(population,),
        methods=(methods,),
        results=(results,),
        status="已提取",
    )
    knowledge_map = build_knowledge_map(
        domain_name="测试领域",
        papers=(paper,),
        analyses={paper.paper_id: analysis},
        evidence_by_paper={
            paper.paper_id: (population, methods, results)
        },
    )

    payload = build_graph_payload(knowledge_map)
    node_ids = [node["id"] for node in payload["all_nodes"]]
    edge_ids = [edge["id"] for edge in payload["all_edges"]]

    assert len(node_ids) == len(set(node_ids))
    assert len(edge_ids) == len(set(edge_ids))
    assert any(node["type"] == "研究方向" for node in payload["all_nodes"])
    assert any(node["type"] == "研究变量" for node in payload["all_nodes"])
    assert any(edge["kind"] == "research_relation" for edge in payload["all_edges"])
    assert all(
        edge["evidence"]
        for edge in payload["all_edges"]
        if edge["kind"] == "research_relation"
    )
    assert payload["years"] == ["2025"]
    assert payload["methods"]
    assert payload["evidence_types"] == ["元数据", "全文"]
    assert payload["graph_statistics"]["paper_node_count"] == 1
    assert payload["graph_statistics"]["research_question_count"] >= 1
    assert payload["graph_statistics"]["synthesis_node_count"] >= 0


@pytest.mark.parametrize("page", ["知识图谱", "知识图谱总结"])
def test_knowledge_map_pages_render_without_exception(page: str) -> None:
    app = AppTest.from_file(
        str(PROJECT_ROOT / "app.py"),
        default_timeout=60,
    )
    app.session_state["navigation"] = page
    app.session_state["sidebar_navigation"] = page

    app.run(timeout=60)

    assert not app.exception, [str(item.value) for item in app.exception]
    assert any(title.value == page for title in app.title)


def test_legacy_knowledge_graph_page_names_are_aliased() -> None:
    app = AppTest.from_file(
        str(PROJECT_ROOT / "app.py"),
        default_timeout=60,
    )
    app.session_state["navigation"] = "科研知识地图"
    app.session_state["sidebar_navigation"] = "科研知识地图"

    app.run(timeout=60)

    assert not app.exception, [str(item.value) for item in app.exception]
    assert any(
        title.value == "知识图谱"
        for title in app.title
    )
