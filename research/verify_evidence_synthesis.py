"""实际运行证据综合升级并输出可核验统计。

用法：
    python research/verify_evidence_synthesis.py --db <sqlite> --domain <id>

该脚本会重读论文、重建结构化分析并保存新的争议记录。只应在目标数据库副本或
用户明确授权更新的正式库上运行。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchmind.agents import AcademicDebateAnalyzer, PaperReader
from researchmind.evidence_quality import evidence_quality_flags
from researchmind.storage import ResearchMindStore


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--topic", default="青少年体育锻炼与体质健康")
    parser.add_argument("--save-debate", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    store = ResearchMindStore(args.db)
    papers = store.list_papers(args.domain)
    reader = PaperReader(store)
    analyses = {
        paper.paper_id: reader.read_paper(paper.paper_id)
        for paper in papers
    }
    debate = AcademicDebateAnalyzer(store).analyze(
        args.domain,
        args.topic,
        save=args.save_debate,
    )
    full_text_ids = {
        paper.paper_id
        for paper in papers
        if store.list_evidence(paper.paper_id, limit=1)
    }
    abstract_analysis_ids = {
        analysis.paper_id
        for analysis in analyses.values()
        if analysis.paper_id not in full_text_ids
        and analysis.all_evidence
        and any(
            item.source_type == "摘要"
            for item in analysis.all_evidence
        )
    }
    excluded_evidence = sum(
        not evidence_quality_flags(
            item.quote,
            section_hint=item.section_hint,
            source_type=item.source_type,
        )[0]
        for paper in papers
        for item in store.list_evidence(paper.paper_id)
    )
    status_counts = Counter(
        synthesis.status for synthesis in debate.question_syntheses
    )
    certainty_counts = Counter(
        synthesis.evidence_certainty
        for synthesis in debate.question_syntheses
    )
    representative_evidence = tuple(
        evidence
        for synthesis in debate.question_syntheses
        for evidence in synthesis.representative_evidence
    )
    non_research_representatives = sum(
        not evidence_quality_flags(
            evidence.quote,
            section_hint=evidence.section_hint,
            source_type=evidence.source_type,
        )[0]
        for evidence in representative_evidence
    )
    payload = {
        "paper_count": len(papers),
        "full_text_count": len(full_text_ids),
        "abstract_analysis_count": len(abstract_analysis_ids),
        "structured_analysis_count": sum(
            bool(analysis.all_evidence) for analysis in analyses.values()
        ),
        "evidence_extraction_count": sum(
            len(analysis.all_evidence) for analysis in analyses.values()
        ),
        "filtered_non_research_chunks": excluded_evidence,
        "non_research_representatives": non_research_representatives,
        "preliminary_consensus_count": status_counts.get("初步共识", 0),
        "consistent_finding_count": status_counts.get("初步共识", 0)
        + status_counts.get("有一致研究发现", 0),
        "conflict_count": len(debate.conflicts),
        "insufficient_question_count": status_counts.get("证据不足", 0),
        "question_status_counts": dict(status_counts),
        "certainty_counts": dict(certainty_counts),
        "questions": [
            {
                "question_id": item.question_id,
                "label": item.label,
                "status": item.status,
                "certainty": item.evidence_certainty,
                "supporting": len(item.supporting_paper_ids),
                "opposing": len(item.opposing_paper_ids),
                "inconclusive": len(item.inconclusive_paper_ids),
                "conclusion": item.conclusion,
                "evidence": [
                    {
                        "paper_id": evidence.document_id,
                        "source_type": evidence.source_type,
                        "location": evidence.location_label,
                        "quote": evidence.quote[:320],
                    }
                    for evidence in item.representative_evidence
                ],
            }
            for item in debate.question_syntheses
        ],
        "conflicts": [
            {
                "question_id": item.question_id,
                "viewpoint_a": item.viewpoint_a,
                "paper_ids_a": item.paper_ids_a,
                "viewpoint_b": item.viewpoint_b,
                "paper_ids_b": item.paper_ids_b,
                "reasons": item.divergence_reasons,
            }
            for item in debate.conflicts
        ],
    }
    if args.summary_only:
        payload = {
            key: value
            for key, value in payload.items()
            if key not in {"questions", "conflicts"}
        }
        payload["question_card_brief"] = [
            {
                "label": item.label,
                "status": item.status,
                "certainty": item.evidence_certainty,
                "supporting": len(item.supporting_paper_ids),
                "opposing": len(item.opposing_paper_ids),
                "inconclusive": len(item.inconclusive_paper_ids),
                "first_evidence_source": (
                    item.representative_evidence[0].source_type
                    if item.representative_evidence
                    else ""
                ),
                "first_evidence_location": (
                    item.representative_evidence[0].location_label
                    if item.representative_evidence
                    else ""
                ),
                "first_evidence": (
                    item.representative_evidence[0].quote[:220]
                    if item.representative_evidence
                    else ""
                ),
            }
            for item in debate.question_syntheses
        ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
