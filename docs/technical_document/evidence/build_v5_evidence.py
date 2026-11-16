"""Build the current, verifiable evidence summary from source artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchmind.knowledge_map import build_knowledge_map
from researchmind.storage import ResearchMindStore


DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "adolescent_fitness"
    / "researchmind.sqlite3"
)
DOMAIN_ID = "domain-510b94048a4c"
EVIDENCE_DIR = Path(__file__).resolve().parent
OUTPUT = EVIDENCE_DIR / "test-evidence-v5.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    store = ResearchMindStore(DB_PATH, read_only=True)
    papers = store.list_papers(DOMAIN_ID)
    all_papers = tuple(
        paper
        for domain in store.list_domains()
        for paper in store.list_papers(domain.domain_id)
    )
    analyses = {
        paper.paper_id: store.get_analysis(paper.paper_id) for paper in papers
    }
    evidence_by_paper = {
        paper.paper_id: store.list_evidence(paper.paper_id) for paper in papers
    }
    debate = store.latest_debate(DOMAIN_ID)
    graph = build_knowledge_map(
        domain_name=store.get_domain(DOMAIN_ID).name,
        papers=papers,
        analyses=analyses,
        evidence_by_paper=evidence_by_paper,
        debate=debate,
        synthesis=store.get_latest_synthesis(DOMAIN_ID),
    )
    pytest_xml = EVIDENCE_DIR / "tests" / "pytest-v5-results.xml"
    pytest_tree = ET.parse(pytest_xml)
    root = pytest_tree.getroot()
    suite = root.find("testsuite") if root.tag == "testsuites" else root
    if suite is None:
        raise RuntimeError("JUnit XML 中没有 testsuite。")
    comparison_path = (
        EVIDENCE_DIR / "controversy-comparison" / "summary.json"
    )
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    citation_path = (
        EVIDENCE_DIR
        / "controversy-comparison"
        / "citation-verification.json"
    )
    citation = json.loads(citation_path.read_text(encoding="utf-8"))
    export_manifest_path = EVIDENCE_DIR / "exports" / "export-manifest.json"
    export_manifest = json.loads(
        export_manifest_path.read_text(encoding="utf-8")
    )
    parse_counts = Counter(paper.parse_status for paper in papers)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "statistics_note": (
            "所有统计由当前 SQLite、JUnit XML、导出清单和已保存对照记录重新计算；"
            "旧文档中的39篇、3281条、133节点、1176边属于历史口径。"
        ),
        "active_database": {
            "domain_id": DOMAIN_ID,
            "papers": len(papers),
            "evidence_chunks": sum(len(items) for items in evidence_by_paper.values()),
            "all_database_papers": len(all_papers),
            "all_database_evidence_chunks": sum(
                len(store.list_evidence(paper.paper_id)) for paper in all_papers
            ),
            "papers_with_evidence": sum(
                bool(items) for items in evidence_by_paper.values()
            ),
            "parse_status_counts": dict(parse_counts),
        },
        "evidence_synthesis": {
            "engine_version": debate.engine_version if debate else "",
            "analysis_time": debate.analysis_time if debate else "",
            "question_syntheses": [
                {
                    "question_id": item.question_id,
                    "label": item.label,
                    "status": item.status,
                    "review_status": item.review_status,
                    "positive_paper_count": len(item.supporting_paper_ids),
                    "negative_paper_count": len(item.negative_paper_ids),
                    "null_paper_count": len(item.null_paper_ids),
                    "mixed_paper_count": len(item.mixed_paper_ids),
                }
                for item in (debate.question_syntheses if debate else ())
            ],
            "conflict_candidate_count": len(debate.conflicts) if debate else 0,
            "conflict_candidates": [
                {
                    "question_id": item.question_id,
                    "review_status": item.review_status,
                    "comparability_status": item.comparability_status,
                }
                for item in (debate.conflicts if debate else ())
            ],
        },
        "knowledge_graph": {
            "papers": len(graph.papers),
            "in_scope_papers": len(graph.in_scope_papers),
            "pending_papers": len(graph.out_of_scope_papers),
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "node_type_counts": dict(
                Counter(node.node_type for node in graph.nodes)
            ),
            "edge_kind_counts": dict(
                Counter(edge.edge_kind for edge in graph.edges)
            ),
            "disclaimer": (
                "该图为研究主题/方向/变量/论文关系图，不等同于论文引用网络。"
            ),
        },
        "pytest": {
            "junit_xml": str(pytest_xml.relative_to(PROJECT_ROOT)),
            "junit_sha256": sha256(pytest_xml),
            "tests": int(suite.attrib.get("tests", 0)),
            "failures": int(suite.attrib.get("failures", 0)),
            "errors": int(suite.attrib.get("errors", 0)),
            "skipped": int(suite.attrib.get("skipped", 0)),
            "time_seconds": float(suite.attrib.get("time", 0.0)),
            "scope": "当前源代码完整 pytest 集，不等于体育科研判断正确性测试。",
        },
        "controversy_comparison": {
            "historical_model": comparison.get("model", ""),
            "model_rerun_in_v5": False,
            "project_conflict_candidate_count": comparison.get(
                "v5_reanalysis", {}
            ).get("project_conflict_question_count"),
            "baseline_raw_conflict_count": comparison.get(
                "v5_reanalysis", {}
            ).get("baseline_raw_conflict_count"),
            "baseline_cross_paper_positive_vs_negative_count": comparison.get(
                "v5_reanalysis", {}
            ).get("baseline_cross_paper_positive_vs_negative_count"),
            "conflict_presence_agreement": comparison.get(
                "v5_reanalysis", {}
            ).get("conflict_presence_agreement"),
            "metric_interpretation": "一致性，不是准确率",
            "experiment_scale": "n=1工程对照",
            "manual_review_status": comparison.get(
                "v5_reanalysis", {}
            ).get("manual_review_status"),
            "citation_membership_verified": citation.get(
                "all_citations_found_in_input_context"
            ),
            "full_text_available_evidence_count": citation.get(
                "full_text_available_evidence_count"
            ),
            "claim_support_verified": citation.get(
                "claim_support_verified"
            ),
            "claim_support_pending_count": citation.get(
                "claim_support_pending_count"
            ),
        },
        "exports": {
            "manifest": str(export_manifest_path.relative_to(PROJECT_ROOT)),
            "manifest_sha256": sha256(export_manifest_path),
            "core_paper_count": export_manifest["core_domain"]["paper_count"],
            "core_evidence_count": export_manifest["core_domain"][
                "evidence_count"
            ],
            "all_database_paper_count": export_manifest["all_database"][
                "paper_count"
            ],
            "all_database_evidence_count": export_manifest["all_database"][
                "evidence_count"
            ],
            "knowledge_graph_node_count": export_manifest[
                "knowledge_graph_node_count"
            ],
            "knowledge_graph_edge_count": export_manifest[
                "knowledge_graph_edge_count"
            ],
        },
        "uncovered": [
            "OCR未实现",
            "500页以上长文档未实测",
            "541页赛事样例未运行",
            "非自主多Agent协商",
            "七题对照无独立专家金标准",
            "逐论断原文支持关系仍待人工复核",
        ],
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
