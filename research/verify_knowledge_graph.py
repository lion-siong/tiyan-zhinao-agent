"""核验多层级科研知识图谱的真实结构与动态更新。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchmind.knowledge_map import build_knowledge_map
from researchmind.library_state import (
    graph_fingerprint,
    library_fingerprint,
    refresh_domain_analysis_now,
)
from researchmind.models import EvidenceRef, PaperRecord
from researchmind.storage import ResearchMindStore
from researchmind.workflow import ResearchMindWorkflow
from config import settings


def _build(store: ResearchMindStore, domain_id: str):
    domain = store.get_domain(domain_id)
    if domain is None:
        raise ValueError(f"领域不存在：{domain_id}")
    papers = store.list_papers(domain_id)
    return build_knowledge_map(
        domain_name=domain.name,
        papers=papers,
        analyses={
            paper.paper_id: store.get_analysis(paper.paper_id)
            for paper in papers
        },
        evidence_by_paper={
            paper.paper_id: store.list_evidence(paper.paper_id)
            for paper in papers
        },
        debate=store.latest_debate(domain_id),
        synthesis=store.get_latest_synthesis(domain_id),
    )


def _stats(store: ResearchMindStore, domain_id: str) -> dict:
    graph = _build(store, domain_id)
    return {
        "papers": len(graph.papers),
        "in_scope": len(graph.in_scope_papers),
        "pending": len(graph.out_of_scope_papers),
        "themes": graph.graph_statistics.theme_count,
        "directions": graph.graph_statistics.direction_count,
        "pending_directions": graph.graph_statistics.pending_direction_count,
        "variables": graph.graph_statistics.variable_count,
        "paper_nodes": graph.graph_statistics.paper_node_count,
        "hierarchy_relations": graph.graph_statistics.hierarchy_relation_count,
        "research_relations": graph.graph_statistics.research_relation_count,
        "typed_research_relations": (
            graph.graph_statistics.typed_research_relation_count
        ),
        "inconsistent_relations": (
            graph.graph_statistics.inconsistent_relation_count
        ),
        "paper_variable_relations": (
            graph.graph_statistics.paper_variable_relation_count
        ),
        "question_variable_relations": (
            graph.graph_statistics.question_variable_relation_count
        ),
        "consensus_nodes": graph.graph_statistics.consensus_node_count,
        "controversy_nodes": graph.graph_statistics.controversy_node_count,
        "research_gap_nodes": graph.graph_statistics.research_gap_node_count,
        "synthesis_nodes": graph.graph_statistics.synthesis_node_count,
        "node_type_counts": dict(Counter(node.node_type for node in graph.nodes)),
        "node_level_counts": dict(Counter(node.level for node in graph.nodes)),
        "edge_kind_counts": dict(Counter(edge.edge_kind for edge in graph.edges)),
        "relation_status_counts": dict(
            Counter(
                edge.relation_status
                for edge in graph.edges
                if edge.edge_kind == "research_relation"
            )
        ),
        "insight_counts": dict(
            Counter(item.category for item in graph.graph_insights)
        ),
        "duplicate_nodes": len(graph.nodes)
        - len({node.node_id for node in graph.nodes}),
        "duplicate_edges": len(graph.edges)
        - len({edge.edge_id for edge in graph.edges}),
        "relations_without_evidence": sum(
            not edge.evidence for edge in graph.edges
        ),
        "nodes_without_evidence": sum(
            not node.evidence
            for node in graph.nodes
            if node.node_id not in {"core:direction"}
            and node.level >= 2
        ),
        "library_fingerprint": library_fingerprint(store, domain_id)[:12],
        "graph_fingerprint": graph_fingerprint(store, domain_id)[:12],
    }


def _mutation_test(source_db: Path, domain_id: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="researchmind-graph-test-") as temp:
        test_db = Path(temp) / "researchmind.sqlite3"
        shutil.copy2(source_db, test_db)
        store = ResearchMindStore(test_db)
        domain = store.get_domain(domain_id)
        if domain is None:
            raise ValueError(f"领域不存在：{domain_id}")
        workflow = ResearchMindWorkflow(
            store=store,
            config=settings,
            knowledge_base_dir=Path(temp) / "knowledge-base",
            default_debate_topic="青少年体育锻炼干预与体质健康水平的关系",
        )
        before = _stats(store, domain_id)
        paper_id = "graph-mutation-test-paper"
        paper = PaperRecord(
            paper_id=paper_id,
            domain_id=domain_id,
            title="青少年有氧运动干预与心肺耐力自动化验证论文",
            authors=("Graph Fixture",),
            year="2026",
            journal="Automated Graph Mutation Test",
            doi="10.0000/graph.mutation.test",
            source_url="https://example.org/graph-fixture",
            abstract="自动化图谱测试夹具，研究青少年有氧运动干预与心肺耐力。",
            keywords=("青少年", "体育锻炼", "有氧运动", "心肺耐力"),
            pdf_path="fixture.pdf",
            source="自动化图谱测试夹具",
            verification_status="测试夹具",
            created_at=datetime.now(timezone.utc).isoformat(),
            parse_status="已解析",
            analysis_status="未分析",
        )
        evidence = EvidenceRef(
            chunk_id=f"{paper_id}-result",
            document_id=paper_id,
            document_title=paper.title,
            file_name="fixture.pdf",
            page_number=3,
            quote=(
                "青少年有氧运动干预显著改善心肺耐力和最大摄氧量"
                "（p < 0.05）。该句只用于自动化图谱动态测试。"
            ),
            source_type="全文",
            section_hint="结果",
        )
        store.upsert_paper(paper)
        store.replace_evidence(paper_id, (evidence,))
        refresh_domain_analysis_now(
            workflow,
            domain,
            workflow.default_debate_topic,
        )
        after_add = _stats(store, domain_id)
        store.soft_delete_papers(domain_id, (paper_id,))
        refresh_domain_analysis_now(
            workflow,
            domain,
            workflow.default_debate_topic,
        )
        after_delete = _stats(store, domain_id)
        return {
            "before": before,
            "after_add": after_add,
            "after_delete": after_delete,
            "add_changed_fingerprint": (
                before["graph_fingerprint"] != after_add["graph_fingerprint"]
            ),
            "delete_restored_baseline": (
                after_delete["papers"] == before["papers"]
                and after_delete["paper_nodes"] == before["paper_nodes"]
                and after_delete["directions"] == before["directions"]
                and after_delete["variables"] == before["variables"]
                and after_delete["synthesis_nodes"] == before["synthesis_nodes"]
                and after_delete["hierarchy_relations"]
                == before["hierarchy_relations"]
            ),
            "added_paper_node": after_add["paper_nodes"] == before["paper_nodes"] + 1,
        }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--mutation-test", action="store_true")
    args = parser.parse_args()
    db_path = Path(args.db).resolve()
    store = ResearchMindStore(db_path, read_only=True)
    payload = _stats(store, args.domain)
    if args.mutation_test:
        payload["mutation_test"] = _mutation_test(db_path, args.domain)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
