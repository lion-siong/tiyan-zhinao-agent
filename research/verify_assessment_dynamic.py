"""在论文库副本上验证新论文参与组卷及删除后的题目失效。"""

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
from researchmind.models import (
    ASSESSMENT_MODULES,
    EvidenceRef,
    PaperAnalysis,
    PaperRecord,
)
from researchmind.storage import ResearchMindStore


DOMAIN_ID = "domain-510b94048a4c"
TEST_PAPER_ID = "assessment-dynamic-test-paper"
QUESTION_TYPES = (
    "单项选择题",
    "多项选择题",
    "判断题",
    "简答题",
    "文献对比分析题",
)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    source = PROJECT_ROOT / "data/adolescent_fitness/researchmind.sqlite3"
    with tempfile.TemporaryDirectory(prefix="assessment-dynamic-") as temp:
        target = Path(temp) / "researchmind.sqlite3"
        shutil.copy2(source, target)
        store = ResearchMindStore(target)
        paper = PaperRecord(
            paper_id=TEST_PAPER_ID,
            domain_id=DOMAIN_ID,
            title="青少年体育锻炼干预与心肺耐力自动化验证研究",
            authors=("Assessment Fixture",),
            year="2099",
            journal="Automated Assessment Test",
            doi="10.0000/assessment.dynamic",
            source_url="https://example.org/assessment-fixture",
            abstract=(
                "本研究验证新增论文参与真实组卷。研究对象为青少年，"
                "采用随机对照运动干预并评价心肺耐力。"
            ),
            keywords=("青少年", "体育锻炼", "心肺耐力", "随机对照"),
            pdf_path="assessment-fixture.pdf",
            source="自动化组卷测试夹具",
            verification_status="测试夹具",
            created_at="2026-11-15T00:00:00+00:00",
            parse_status="已解析",
            analysis_status="已分析",
        )
        methods = EvidenceRef(
            chunk_id=f"{TEST_PAPER_ID}-methods",
            document_id=TEST_PAPER_ID,
            document_title=paper.title,
            file_name=paper.pdf_path,
            page_number=2,
            quote=(
                "采用随机对照研究，干预组接受十二周青少年体育锻炼干预，"
                "对照组维持常规体育活动。"
            ),
            source_type="全文",
            section_hint="方法",
        )
        results = EvidenceRef(
            chunk_id=f"{TEST_PAPER_ID}-results",
            document_id=TEST_PAPER_ID,
            document_title=paper.title,
            file_name=paper.pdf_path,
            page_number=5,
            quote=(
                "研究结果显示，干预组青少年的心肺耐力和最大摄氧量较对照组"
                "显著提高，结果具有统计学意义。"
            ),
            source_type="全文",
            section_hint="结果",
        )
        conclusions = EvidenceRef(
            chunk_id=f"{TEST_PAPER_ID}-conclusions",
            document_id=TEST_PAPER_ID,
            document_title=paper.title,
            file_name=paper.pdf_path,
            page_number=6,
            quote=(
                "研究结论认为，十二周体育锻炼干预可改善青少年心肺耐力，"
                "但长期效果仍需进一步随访。"
            ),
            source_type="全文",
            section_hint="结论",
        )
        store.upsert_paper(paper)
        store.replace_evidence(
            TEST_PAPER_ID,
            (methods, results, conclusions),
        )
        store.save_analysis(
            PaperAnalysis(
                paper_id=TEST_PAPER_ID,
                methods=(methods,),
                results=(results,),
                conclusions=(conclusions,),
                status="自动化组卷测试夹具，字段完整。",
            )
        )
        agent = LearningAssessmentAgent(store)
        module_questions = agent.generate_question_set(
            DOMAIN_ID,
            assessment_modules=ASSESSMENT_MODULES,
            debate_topic="青少年体育锻炼与体质健康",
        )
        module_counts = {
            module: sum(
                question.assessment_module == module
                for question in module_questions
            )
            for module in ASSESSMENT_MODULES
        }
        comparison = next(
            question
            for question in module_questions
            if question.question_type == "文献对比分析题"
        )
        five_dimension_labels = [
            criterion.label for criterion in comparison.rubric
        ]
        generated = agent.generate_question_set(
            DOMAIN_ID,
            question_types=QUESTION_TYPES,
            debate_topic="青少年体育锻炼与体质健康",
        )
        selected_new_paper = any(
            TEST_PAPER_ID in question.source_paper_ids
            for question in generated
        )
        explanations_complete = all(
            question.explanation.strip() for question in generated
        )

        store.soft_delete_papers(DOMAIN_ID, (TEST_PAPER_ID,))
        cleanup = store.invalidate_domain_derivations(
            DOMAIN_ID,
            (TEST_PAPER_ID,),
        )
        remaining_with_deleted_source = [
            question.question_id
            for question in store.list_questions(DOMAIN_ID)
            if TEST_PAPER_ID in question.source_paper_ids
            or any(
                item.document_id == TEST_PAPER_ID
                for item in question.evidence
            )
        ]
        regenerated = agent.generate_question_set(
            DOMAIN_ID,
            question_types=QUESTION_TYPES,
            debate_topic="青少年体育锻炼与体质健康",
        )
        regenerated_uses_deleted = any(
            TEST_PAPER_ID in question.source_paper_ids
            for question in regenerated
        )
        report = {
            "five_module_question_count": len(module_questions),
            "five_module_counts": module_counts,
            "comparison_paper_count": len(
                set(comparison.source_paper_ids)
            ),
            "comparison_evidence_count": len(comparison.evidence),
            "comparison_five_dimensions": five_dimension_labels,
            "generated_question_count": len(generated),
            "generated_types": [item.question_type for item in generated],
            "selected_new_paper": selected_new_paper,
            "explanations_complete": explanations_complete,
            "deleted_question_cleanup": cleanup["questions"],
            "remaining_with_deleted_source": remaining_with_deleted_source,
            "regenerated_uses_deleted_paper": regenerated_uses_deleted,
            "regenerated_question_count": len(regenerated),
        }
        report["passed"] = all(
            (
                all(module_counts.values()),
                len(module_questions) == 8,
                len(set(comparison.source_paper_ids)) >= 2,
                len(comparison.evidence) >= 4,
                five_dimension_labels
                == [
                    "核心观点准确性",
                    "文献证据引用能力",
                    "多篇论文对比能力",
                    "学术逻辑与推理能力",
                    "批判性思维能力",
                ],
                len(generated) == 5,
                selected_new_paper,
                explanations_complete,
                cleanup["questions"] >= 1,
                not remaining_with_deleted_source,
                not regenerated_uses_deleted,
                len(regenerated) == 5,
            )
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
