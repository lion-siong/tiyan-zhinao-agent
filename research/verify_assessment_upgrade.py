"""在正式论文库副本上验收五模块组卷和证据评分。"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchmind.agents import LearningAssessmentAgent
from researchmind.models import ASSESSMENT_MODULES
from researchmind.storage import ResearchMindStore


DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "adolescent_fitness"
    / "researchmind.sqlite3"
)


class _LocalOnlyLLM:
    """强制验收本地证据评分，避免网络和密钥状态影响结果。"""

    is_configured = False


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="assessment-upgrade-") as temp:
        target = Path(temp) / DATABASE_PATH.name
        shutil.copy2(DATABASE_PATH, target)
        store = ResearchMindStore(target)
        domains = tuple(
            domain
            for domain in store.list_domains()
            if store.paper_count(domain.domain_id) > 0
        )
        if not domains:
            raise RuntimeError("正式论文库中没有活动论文。")
        domain = max(
            domains,
            key=lambda item: store.paper_count(item.domain_id),
        )
        agent = LearningAssessmentAgent(
            store,
            llm_service=_LocalOnlyLLM(),
        )
        questions = agent.generate_question_set(
            domain.domain_id,
            assessment_modules=ASSESSMENT_MODULES,
            debate_topic="青少年体质健康与体育锻炼",
        )
        module_counts = {
            module: sum(
                question.assessment_module == module
                for question in questions
            )
            for module in ASSESSMENT_MODULES
        }
        comparison = next(
            (
                question
                for question in questions
                if question.question_type == "文献对比分析题"
            ),
            None,
        )
        evidence_score = (
            agent.score(comparison, comparison.standard_answer)
            if comparison is not None
            else None
        )
        superficial_score = (
            agent.score(
                comparison,
                "研究有意义并且结论很重要。" * 300,
            )
            if comparison is not None
            else None
        )
        report = {
            "database": str(DATABASE_PATH),
            "domain_id": domain.domain_id,
            "domain_name": domain.name,
            "active_papers": store.paper_count(domain.domain_id),
            "question_count": len(questions),
            "module_counts": module_counts,
            "missing_modules": list(
                agent.last_missing_assessment_modules
            ),
            "missing_question_types": list(
                agent.last_missing_question_types
            ),
            "unparsed_papers": len(agent.last_unparsed_titles),
            "field_gaps": agent.last_field_gap_counts,
            "all_explanations_present": all(
                question.explanation.strip()
                for question in questions
            ),
            "all_modules_persisted": all(
                (
                    persisted := store.get_question(question.question_id)
                )
                is not None
                and persisted.assessment_module
                == question.assessment_module
                for question in questions
            ),
            "subjective_question_count": sum(
                not question.is_objective for question in questions
            ),
            "all_subjective_rubrics_have_five_dimensions": all(
                len(question.rubric) == 5
                for question in questions
                if not question.is_objective
            ),
            "comparison_paper_count": (
                len(set(comparison.source_paper_ids))
                if comparison is not None
                else 0
            ),
            "comparison_evidence_count": (
                len(comparison.evidence)
                if comparison is not None
                else 0
            ),
            "comparison_rubric": (
                [item.label for item in comparison.rubric]
                if comparison is not None
                else []
            ),
            "evidence_answer_score": (
                evidence_score.score
                if evidence_score is not None
                else None
            ),
            "evidence_answer_max": (
                evidence_score.max_score
                if evidence_score is not None
                else None
            ),
            "superficial_long_answer_score": (
                superficial_score.score
                if superficial_score is not None
                else None
            ),
        }
        report["passed"] = all(
            (
                all(module_counts.values()),
                not report["missing_modules"],
                not report["missing_question_types"],
                report["all_explanations_present"],
                report["all_modules_persisted"],
                report["all_subjective_rubrics_have_five_dimensions"],
                report["comparison_paper_count"] >= 2,
                report["comparison_evidence_count"] >= 4,
                report["evidence_answer_score"]
                > report["superficial_long_answer_score"],
            )
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
