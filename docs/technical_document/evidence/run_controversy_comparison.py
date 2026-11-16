"""Compare a one-shot LLM controversy baseline with the project pipeline."""

from __future__ import annotations

import json
import hashlib
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from researchmind.storage import ResearchMindStore
from services.llm_service import LLMService


DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "adolescent_fitness"
    / "researchmind.sqlite3"
)
DOMAIN_ID = "domain-510b94048a4c"
OUTPUT = Path(__file__).resolve().parent / "controversy-comparison"


def extract_json(value: str) -> dict[str, Any]:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型未返回 JSON 对象。")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("模型返回的 JSON 顶层不是对象。")
    return parsed


def normalize_id_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [
            item.strip()
            for item in re.split(r"[,;，；\s]+", value)
            if item.strip()
        ]
    return []


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not settings.llm_configured:
        raise RuntimeError("当前未配置大模型，无法执行真实对照实验。")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    store = ResearchMindStore(DB_PATH, read_only=True)
    debate = store.latest_debate(DOMAIN_ID)
    if debate is None:
        raise RuntimeError("当前领域没有已保存的学术争议分析。")

    service = LLMService(settings)
    comparisons: list[dict[str, Any]] = []
    for synthesis in debate.question_syntheses:
        evidence_items = []
        context_lines = []
        evidence_by_id: dict[str, dict[str, str]] = {}
        for index, evidence in enumerate(
            synthesis.representative_evidence,
            start=1,
        ):
            evidence_id = f"E{index}"
            payload = {
                "evidence_id": evidence_id,
                "paper_id": evidence.document_id,
                "paper_title": evidence.document_title,
                "source_type": evidence.source_type,
                "location": evidence.location_label,
                "quote_sha256": hashlib.sha256(
                    evidence.quote.encode("utf-8")
                ).hexdigest(),
                "quote_preview": evidence.quote[:160],
                "quote_text": evidence.quote,
            }
            evidence_by_id[evidence_id] = payload
            evidence_items.append(payload)
            context_lines.append(
                f"[{evidence_id}] paper_id={evidence.document_id}; "
                f"title={evidence.document_title}; "
                f"source={evidence.source_type}; "
                f"location={evidence.location_label}\n"
                f"quote: {evidence.quote}"
            )

        prompt = (
            "你是体育科研文献分析基线。只依据给出的原始证据片段，"
            "判断该研究问题在相同结局和相同方向比较下是否存在真实学术争议。\n"
            "规则：\n"
            "1. 方向未判定、证据不足或研究问题不同，都不算争议。\n"
            "2. 未达统计显著不等于效应方向相反，不能作为观点B的反方。\n"
            "3. 声称存在争议时，观点 A 和观点 B 必须各自引用给出的 paper_id，"
            "并引用至少一个 E 编号证据。\n"
            "4. 不得使用未提供的论文、页码、DOI 或数据。\n"
            "5. 只输出 JSON，不要 Markdown。\n"
            "输出字段：has_conflict（布尔值）、conflict_question、viewpoint_a、"
            "papers_a、viewpoint_b、papers_b、citations、confidence、explanation。"
            "无争议时观点字段可为空数组或空字符串。\n\n"
            f"研究问题：{synthesis.label}\n"
            f"证据片段数：{len(evidence_items)}\n"
            + "\n\n".join(context_lines)
        )
        started = time.perf_counter()
        try:
            message = service.chat_messages(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是严格的科研争议判定器，只能使用用户提供的证据。"
                            "无法证明同一研究问题中存在相反结果时必须回答无争议。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            content = message.get("content")
            if isinstance(content, list):
                raw = "\n".join(
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, dict)
                ).strip()
            else:
                raw = str(content or "").strip()
            parsed = extract_json(raw)
            error = ""
        except Exception as exc:
            raw = ""
            parsed = {}
            error = f"{type(exc).__name__}: {exc}"
        elapsed = round(time.perf_counter() - started, 3)

        papers_a = normalize_id_list(parsed.get("papers_a"))
        papers_b = normalize_id_list(parsed.get("papers_b"))
        citations = normalize_id_list(parsed.get("citations"))
        provided_papers = {
            item["paper_id"] for item in evidence_items
        }
        provided_evidence = set(evidence_by_id)
        invalid_papers = sorted(
            (set(papers_a) | set(papers_b)) - provided_papers
        )
        invalid_citations = sorted(set(citations) - provided_evidence)
        raw_conflict_value = parsed.get("has_conflict")
        has_conflict = raw_conflict_value is True or (
            isinstance(raw_conflict_value, str)
            and raw_conflict_value.strip().lower() == "true"
        )
        paper_sets_disjoint = bool(
            papers_a and papers_b and set(papers_a).isdisjoint(papers_b)
        )
        opposition_text = " ".join(
            (
                str(parsed.get("viewpoint_b", "")),
                str(parsed.get("explanation", "")),
            )
        ).casefold()
        baseline_uses_non_significant_as_opposition = any(
            term in opposition_text
            for term in (
                "no significant",
                "not significant",
                "did not significantly",
                "无显著",
                "未达统计显著",
                "不显著",
            )
        )
        structurally_traceable = bool(
            has_conflict
            and papers_a
            and papers_b
            and paper_sets_disjoint
            and citations
            and not invalid_papers
            and not invalid_citations
            and not baseline_uses_non_significant_as_opposition
        )
        project_has_conflict = bool(
            synthesis.positive_paper_ids
            and synthesis.negative_paper_ids
        )
        comparisons.append(
            {
                "question_id": synthesis.question_id,
                "question": synthesis.label,
                "project_status": synthesis.status,
                "project_has_conflict": project_has_conflict,
                "project_certainty": synthesis.evidence_certainty,
                "project_supporting": len(synthesis.supporting_paper_ids),
                "project_opposing": len(synthesis.opposing_paper_ids),
                "project_inconclusive": len(
                    synthesis.inconclusive_paper_ids
                ),
                "provided_paper_count": len(provided_papers),
                "provided_evidence_count": len(evidence_items),
                "baseline_error": error,
                "baseline_elapsed_seconds": elapsed,
                "baseline_raw_response": raw,
                "baseline": parsed,
                "baseline_has_conflict": has_conflict,
                "baseline_structurally_traceable_conflict": (
                    structurally_traceable
                ),
                "baseline_paper_sets_disjoint": paper_sets_disjoint,
                "baseline_same_paper_conflict": (
                    has_conflict and not paper_sets_disjoint
                ),
                "baseline_uses_non_significant_as_opposition": (
                    baseline_uses_non_significant_as_opposition
                ),
                "invalid_paper_ids": invalid_papers,
                "invalid_evidence_ids": invalid_citations,
                "agreement_with_project": (
                    has_conflict == project_has_conflict
                ),
                "traceable_agreement_with_project": (
                    structurally_traceable == project_has_conflict
                ),
            }
        )
        (OUTPUT / f"{synthesis.question_id.replace(':', '-')}.json").write_text(
            json.dumps(
                {
                    "input_evidence": evidence_items,
                    "model_response": parsed,
                    "raw_response": raw,
                    "error": error,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    project_conflicts = {
        item["question_id"]
        for item in comparisons
        if item["project_has_conflict"]
    }
    baseline_conflicts = {
        item["question_id"]
        for item in comparisons
        if item["baseline_structurally_traceable_conflict"]
    }
    baseline_raw_conflicts = {
        item["question_id"]
        for item in comparisons
        if item["baseline_has_conflict"]
    }
    attempted = len(comparisons)
    successful = sum(not item["baseline_error"] for item in comparisons)
    citation_instances = sum(
        len(normalize_id_list(item["baseline"].get("citations", [])))
        for item in comparisons
    )
    invalid_citation_instances = sum(
        len(item["invalid_evidence_ids"]) for item in comparisons
    )
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": settings.openai_model,
        "domain_id": DOMAIN_ID,
        "experiment_design": (
            "同一组 7 个研究问题与相同 representative evidence；"
            "普通大模型单次提示判定，体研智脑使用同题结构化分析。"
        ),
        "research_question_count": len(comparisons),
        "successful_baseline_calls": successful,
        "failed_baseline_calls": attempted - successful,
        "project_conflict_questions": sorted(project_conflicts),
        "baseline_traceable_conflict_questions": sorted(baseline_conflicts),
        "baseline_raw_conflict_questions": sorted(baseline_raw_conflicts),
        "exact_conflict_overlap": sorted(
            project_conflicts & baseline_conflicts
        ),
        "raw_conflict_overlap": sorted(
            project_conflicts & baseline_raw_conflicts
        ),
        "same_paper_conflict_questions": sorted(
            item["question_id"]
            for item in comparisons
            if item["baseline_same_paper_conflict"]
        ),
        "agreement_count": sum(
            item["agreement_with_project"] for item in comparisons
        ),
        "traceable_agreement_count": sum(
            item["traceable_agreement_with_project"]
            for item in comparisons
        ),
        "citation_instances": citation_instances,
        "invalid_citation_instances": invalid_citation_instances,
        "comparisons": comparisons,
        "limitations": [
            "单次调用，不做统计显著性推断。",
            "体研智脑结果是规则分析结果，不是人工金标准。",
            "普通大模型使用的是系统选出的 representative evidence 子集，"
            "不代表它在全量原始 PDF 上的上限。",
        ],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if attempted == 7 else 1


if __name__ == "__main__":
    raise SystemExit(main())
