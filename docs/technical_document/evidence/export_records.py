"""Export verifiable paper, evidence and knowledge-graph records."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
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
OUTPUT_DIR = Path(__file__).resolve().parent / "exports"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"没有可导出的记录：{path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    store = ResearchMindStore(DB_PATH, read_only=True)
    domain = store.get_domain(DOMAIN_ID)
    if domain is None:
        raise RuntimeError("正式领域不存在。")
    all_domains = store.list_domains()
    all_papers = tuple(
        paper
        for item in all_domains
        for paper in store.list_papers(item.domain_id)
    )
    papers = tuple(
        paper for paper in all_papers if paper.domain_id == DOMAIN_ID
    )
    analyses = {
        paper.paper_id: store.get_analysis(paper.paper_id)
        for paper in all_papers
    }
    evidence_by_paper = {
        paper.paper_id: store.list_evidence(paper.paper_id)
        for paper in all_papers
    }

    paper_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    all_paper_rows: list[dict[str, object]] = []
    all_evidence_rows: list[dict[str, object]] = []
    for paper in all_papers:
        evidence = evidence_by_paper[paper.paper_id]
        analysis = analyses.get(paper.paper_id)
        paper_row = {
                "domain_id": paper.domain_id,
                "paper_id": paper.paper_id,
                "title": paper.title,
                "authors": "; ".join(paper.authors),
                "year": paper.year,
                "journal": paper.journal,
                "doi": paper.doi,
                "source": paper.source,
                "category": paper.category,
                "keywords": "; ".join(paper.keywords),
                "parse_status": paper.parse_status,
                "analysis_status": paper.analysis_status,
                "pdf_file_name": Path(paper.pdf_path).name,
                "evidence_count": len(evidence),
                "analysis_field_count": (
                    sum(
                        bool(analysis.field_evidence(field))
                        for field in (
                            "purpose",
                            "population",
                            "methods",
                            "interventions",
                            "comparators",
                            "measures",
                            "results",
                            "statistics",
                            "conclusions",
                            "limitations",
                        )
                    )
                    if analysis is not None
                    else 0
                ),
                "doi_resolution_url": (
                    f"https://doi.org/{paper.doi}" if paper.doi else ""
                ),
            }
        all_paper_rows.append(paper_row)
        if paper.domain_id == DOMAIN_ID:
            paper_rows.append(paper_row)
        for item in evidence:
            quote_bytes = item.quote.encode("utf-8")
            evidence_row = {
                    "domain_id": paper.domain_id,
                    "evidence_id": item.chunk_id,
                    "paper_id": paper.paper_id,
                    "paper_title": paper.title,
                    "pdf_file_name": item.file_name,
                    "page_number": item.page_number,
                    "source_type": item.source_type,
                    "section_hint": item.section_hint,
                    "quote_length": len(item.quote),
                    "quote_sha256": hashlib.sha256(quote_bytes).hexdigest(),
                    "quote_preview": item.quote[:160].replace("\r", " ").replace(
                        "\n",
                        " ",
                    ),
                }
            all_evidence_rows.append(evidence_row)
            if paper.domain_id == DOMAIN_ID:
                evidence_rows.append(evidence_row)

    papers_path = OUTPUT_DIR / f"papers-{len(paper_rows)}.csv"
    all_papers_path = (
        OUTPUT_DIR / f"papers-{len(all_paper_rows)}-all-database.csv"
    )
    evidence_path = (
        OUTPUT_DIR / f"evidence-{len(evidence_rows)}-core-domain.csv"
    )
    all_evidence_path = (
        OUTPUT_DIR / f"evidence-{len(all_evidence_rows)}-all-database.csv"
    )
    write_csv(papers_path, paper_rows)
    write_csv(all_papers_path, all_paper_rows)
    write_csv(evidence_path, evidence_rows)
    write_csv(all_evidence_path, all_evidence_rows)

    debate = store.latest_debate(DOMAIN_ID)
    graph = build_knowledge_map(
        domain_name=domain.name,
        papers=papers,
        analyses=analyses,
        evidence_by_paper=evidence_by_paper,
        debate=debate,
        synthesis=store.get_latest_synthesis(DOMAIN_ID),
    )
    node_rows = [
        {
            "node_id": node.node_id,
            "label": node.label,
            "node_type": node.node_type,
            "level": node.level,
            "paper_count": node.paper_count,
            "evidence_count": node.evidence_count,
            "source_paper_ids": "; ".join(sorted(node.paper_ids)),
        }
        for node in graph.nodes
    ]
    edge_rows = [
        {
            "edge_id": edge.edge_id,
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "edge_kind": edge.edge_kind,
            "relation": edge.relation,
            "relation_status": edge.relation_status,
            "evidence_count": edge.evidence_count,
            "source_paper_ids": "; ".join(sorted(edge.paper_ids)),
        }
        for edge in graph.edges
    ]
    nodes_path = OUTPUT_DIR / "knowledge-graph-nodes.csv"
    edges_path = OUTPUT_DIR / "knowledge-graph-edges.csv"
    write_csv(nodes_path, node_rows)
    write_csv(edges_path, edge_rows)

    stats = graph.graph_statistics
    knowledge_stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "domain_id": domain.domain_id,
        "domain_name": domain.name,
        "papers": len(graph.papers),
        "in_scope_papers": len(graph.in_scope_papers),
        "pending_papers": len(graph.out_of_scope_papers),
        "themes": stats.theme_count,
        "directions": stats.direction_count,
        "pending_directions": stats.pending_direction_count,
        "variables": stats.variable_count,
        "paper_nodes": stats.paper_node_count,
        "consensus_nodes": stats.consensus_node_count,
        "controversy_nodes": stats.controversy_node_count,
        "research_gap_nodes": stats.research_gap_node_count,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "duplicate_nodes": len(graph.nodes)
        - len({node.node_id for node in graph.nodes}),
        "duplicate_edges": len(graph.edges)
        - len({edge.edge_id for edge in graph.edges}),
        "relations_without_evidence": sum(
            not edge.evidence for edge in graph.edges
        ),
        "node_type_counts": dict(Counter(node.node_type for node in graph.nodes)),
        "node_level_counts": dict(Counter(node.level for node in graph.nodes)),
        "edge_kind_counts": dict(
            Counter(edge.edge_kind for edge in graph.edges)
        ),
        "relation_status_counts": dict(
            Counter(
                edge.relation_status
                for edge in graph.edges
                if edge.edge_kind == "research_relation"
            )
        ),
    }
    stats_path = OUTPUT_DIR / "knowledge-graph-statistics.json"
    stats_path.write_text(
        json.dumps(knowledge_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    synthesis_summary = {
        "generated_at": knowledge_stats["generated_at"],
        "domain_id": DOMAIN_ID,
        "engine_version": debate.engine_version if debate else "",
        "analysis_time": debate.analysis_time if debate else "",
        "paper_count": len(papers),
        "covered_paper_count": (
            len(debate.evidence_covered_paper_ids) if debate else 0
        ),
        "question_syntheses": [
            {
                "question_id": item.question_id,
                "label": item.label,
                "status": item.status,
                "review_status": item.review_status,
                "positive_paper_ids": list(item.supporting_paper_ids),
                "negative_paper_ids": list(item.negative_paper_ids),
                "null_paper_ids": list(item.null_paper_ids),
                "mixed_paper_ids": list(item.mixed_paper_ids),
                "inconclusive_paper_ids": list(item.inconclusive_paper_ids),
                "conclusion": item.conclusion,
            }
            for item in (debate.question_syntheses if debate else ())
        ],
        "conflicts": [
            {
                "conflict_id": item.conflict_id,
                "question_id": item.question_id,
                "review_status": item.review_status,
                "comparability_status": item.comparability_status,
                "significance_note": item.significance_note,
                "paper_ids_a": list(item.paper_ids_a),
                "paper_ids_b": list(item.paper_ids_b),
            }
            for item in (debate.conflicts if debate else ())
        ],
    }
    synthesis_path = OUTPUT_DIR / "evidence-synthesis-summary.json"
    synthesis_path.write_text(
        json.dumps(synthesis_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    source_counts = Counter(row["source"] for row in paper_rows)
    parse_counts = Counter(row["parse_status"] for row in paper_rows)
    category_counts = Counter(row["category"] for row in paper_rows)
    core_unique_evidence_ids = len(
        {row["evidence_id"] for row in evidence_rows}
    )
    core_positive_pages = sum(
        int(row["page_number"]) > 0 for row in evidence_rows
    )
    core_nonempty_hashes = sum(
        bool(str(row["quote_sha256"]).strip()) for row in evidence_rows
    )
    all_unique_evidence_ids = len(
        {row["evidence_id"] for row in all_evidence_rows}
    )
    all_positive_pages = sum(
        int(row["page_number"]) > 0 for row in all_evidence_rows
    )
    all_nonempty_hashes = sum(
        bool(str(row["quote_sha256"]).strip())
        for row in all_evidence_rows
    )
    manifest = {
        "generated_at": knowledge_stats["generated_at"],
        "database_sha256": sha256_file(DB_PATH),
        "domain_id": DOMAIN_ID,
        "core_domain": {
            "domain_id": DOMAIN_ID,
            "paper_count": len(paper_rows),
            "evidence_count": len(evidence_rows),
            "unique_evidence_ids": core_unique_evidence_ids,
            "positive_page_count": core_positive_pages,
            "nonempty_quote_hash_count": core_nonempty_hashes,
        },
        "all_database": {
            "domains": len(all_domains),
            "paper_count": len(all_paper_rows),
            "evidence_count": len(all_evidence_rows),
            "unique_evidence_ids": all_unique_evidence_ids,
            "positive_page_count": all_positive_pages,
            "nonempty_quote_hash_count": all_nonempty_hashes,
        },
        "knowledge_graph_node_count": len(node_rows),
        "knowledge_graph_edge_count": len(edge_rows),
        "source_counts": dict(source_counts),
        "parse_status_counts": dict(parse_counts),
        "category_counts": dict(category_counts),
        "files": {
            path.name: {
                "rows": (
                    len(all_paper_rows)
                    if path == all_papers_path
                    else len(paper_rows)
                    if path == papers_path
                    else len(all_evidence_rows)
                    if path == all_evidence_path
                    else len(evidence_rows)
                    if path == evidence_path
                    else len(node_rows)
                    if path == nodes_path
                    else len(edge_rows)
                    if path == edges_path
                    else 1
                    if path == synthesis_path
                    else 1
                ),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (
                papers_path,
                all_papers_path,
                evidence_path,
                all_evidence_path,
                nodes_path,
                edges_path,
                stats_path,
                synthesis_path,
            )
        },
    }
    manifest_path = OUTPUT_DIR / "export-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
