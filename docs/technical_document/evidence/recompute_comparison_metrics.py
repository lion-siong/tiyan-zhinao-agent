"""Recompute comparison metrics from saved raw model responses."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "controversy-comparison" / "summary.json"


def normalize_id_list(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def main() -> int:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    for comparison in data["comparisons"]:
        baseline = comparison.get("baseline", {})
        papers_a = normalize_id_list(baseline.get("papers_a", []))
        papers_b = normalize_id_list(baseline.get("papers_b", []))
        disjoint = bool(
            papers_a and papers_b and set(papers_a).isdisjoint(papers_b)
        )
        non_significant_as_opposition = comparison.get(
            "baseline_uses_non_significant_as_opposition"
        )
        if non_significant_as_opposition is None:
            text = " ".join(
                (
                    str(baseline.get("viewpoint_b", "")),
                    str(baseline.get("explanation", "")),
                )
            ).casefold()
            non_significant_as_opposition = any(
                term in text
                for term in (
                    "no significant",
                    "not significant",
                    "did not significantly",
                    "无显著",
                    "未达统计显著",
                    "不显著",
                )
            )
            comparison[
                "baseline_uses_non_significant_as_opposition"
            ] = non_significant_as_opposition
        comparison["baseline_paper_sets_disjoint"] = disjoint
        comparison["baseline_same_paper_conflict"] = bool(
            comparison["baseline_has_conflict"] and not disjoint
        )
        comparison["baseline_structurally_traceable_conflict"] = bool(
            comparison["baseline_structurally_traceable_conflict"]
            and disjoint
            and not non_significant_as_opposition
        )
        comparison["traceable_agreement_with_project"] = bool(
            comparison["baseline_structurally_traceable_conflict"]
            == comparison["project_has_conflict"]
        )

    project_conflicts = {
        item["question_id"]
        for item in data["comparisons"]
        if item["project_has_conflict"]
    }
    raw_conflicts = {
        item["question_id"]
        for item in data["comparisons"]
        if item["baseline_has_conflict"]
    }
    traceable_conflicts = {
        item["question_id"]
        for item in data["comparisons"]
        if item["baseline_structurally_traceable_conflict"]
    }
    data["baseline_raw_conflict_questions"] = sorted(raw_conflicts)
    data["baseline_traceable_conflict_questions"] = sorted(
        traceable_conflicts
    )
    data["raw_conflict_overlap"] = sorted(
        project_conflicts & raw_conflicts
    )
    data["exact_conflict_overlap"] = sorted(
        project_conflicts & traceable_conflicts
    )
    data["same_paper_conflict_questions"] = sorted(
        item["question_id"]
        for item in data["comparisons"]
        if item["baseline_same_paper_conflict"]
    )
    data["agreement_count"] = sum(
        item["agreement_with_project"] for item in data["comparisons"]
    )
    data["traceable_agreement_count"] = sum(
        item["traceable_agreement_with_project"]
        for item in data["comparisons"]
    )
    data["metric_interpretation"] = (
        "一致性计数，不是准确率；实验为n=1工程对照，均待独立专家复核"
    )
    SUMMARY.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "raw_conflicts": data["baseline_raw_conflict_questions"],
                "traceable_conflicts": data[
                    "baseline_traceable_conflict_questions"
                ],
                "same_paper_conflicts": data[
                    "same_paper_conflict_questions"
                ],
                "agreement_count": data["agreement_count"],
                "traceable_agreement_count": data[
                    "traceable_agreement_count"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
