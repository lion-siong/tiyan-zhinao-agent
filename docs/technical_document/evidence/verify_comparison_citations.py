"""Independently verify model citations against saved input contexts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "controversy-comparison"
SUMMARY = ROOT / "summary.json"
OUTPUT = ROOT / "citation-verification.json"


def normalize_id_list(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [
            item.strip()
            for item in value.replace(",", " ").replace(";", " ").split()
            if item.strip()
        ]
    return []


def main() -> int:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    checks = []
    citation_count = 0
    invalid_citation_count = 0
    paper_reference_count = 0
    invalid_paper_count = 0
    full_text_available_count = 0
    full_text_missing_count = 0
    for comparison in summary["comparisons"]:
        path = ROOT / (
            comparison["question_id"].replace(":", "-") + ".json"
        )
        saved = json.loads(path.read_text(encoding="utf-8"))
        provided_evidence = {
            item["evidence_id"] for item in saved["input_evidence"]
        }
        provided_papers = {
            item["paper_id"] for item in saved["input_evidence"]
        }
        full_text_ids = {
            item["evidence_id"]
            for item in saved["input_evidence"]
            if str(item.get("quote_text", "")).strip()
        }
        missing_full_text = sorted(provided_evidence - full_text_ids)
        baseline = comparison["baseline"]
        citations = normalize_id_list(baseline.get("citations", []))
        papers = normalize_id_list(baseline.get("papers_a", [])) + (
            normalize_id_list(baseline.get("papers_b", []))
        )
        invalid_citations = sorted(set(citations) - provided_evidence)
        invalid_papers = sorted(set(papers) - provided_papers)
        citation_count += len(citations)
        invalid_citation_count += len(invalid_citations)
        paper_reference_count += len(papers)
        invalid_paper_count += len(invalid_papers)
        checks.append(
            {
                "question": comparison["question"],
                "citations": citations,
                "papers": papers,
                "invalid_citations": invalid_citations,
                "invalid_papers": invalid_papers,
                "full_text_available_evidence": sorted(full_text_ids),
                "full_text_missing_evidence": missing_full_text,
                "claim_support_status": "待人工复核",
                "claim_support_note": (
                    "本脚本只验证引用编号属于保存的输入上下文；"
                    "编号存在不等于原文支持模型论断。"
                ),
            }
        )
        full_text_available_count += len(full_text_ids)
        full_text_missing_count += len(missing_full_text)
    result = {
        "verification_scope": (
            "引用编号归属与输入全文可用性；不包含论断支持关系或专家结论"
        ),
        "question_count": len(checks),
        "citation_instances": citation_count,
        "invalid_citation_instances": invalid_citation_count,
        "paper_reference_instances": paper_reference_count,
        "invalid_paper_instances": invalid_paper_count,
        "all_citations_found_in_input_context": invalid_citation_count == 0,
        "all_papers_found_in_input_context": invalid_paper_count == 0,
        "full_text_available_evidence_count": full_text_available_count,
        "full_text_missing_evidence_count": full_text_missing_count,
        "claim_support_verified_count": 0,
        "claim_support_pending_count": len(checks),
        "claim_support_verified": False,
        "manual_review_required": True,
        "checks": checks,
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if (
        invalid_citation_count == 0 and invalid_paper_count == 0
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
