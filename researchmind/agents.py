"""ResearchMind 的六个核心 Agent 与可追溯分析逻辑。

本地模式只从真实 PDF 证据片段提取和比较，不使用模板补写论文结论。大模型
仅在主观题评分时作为可选增强，并要求输出评分细则。
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

from researchmind.collectors import LiteratureCollector
from researchmind.models import (
    AssessmentQuestion,
    CriterionScore,
    DebateAnalysis,
    EvidenceRef,
    KnowledgeEdge,
    KnowledgeNode,
    PaperAnalysis,
    PaperRecord,
    ResearchSynthesis,
    RubricCriterion,
    ScoreResult,
    StudyRecommendation,
    Viewpoint,
)
from researchmind.storage import BASELINE_MASTERY, ResearchMindStore
from services.llm_service import LLMService


FIELD_LABELS = {
    "purpose": "研究目的",
    "population": "研究对象",
    "methods": "研究方法",
    "results": "研究结果",
    "conclusions": "研究结论",
    "limitations": "研究局限",
}

FIELD_KEYWORDS = {
    "purpose": (
        "研究目的",
        "本文旨在",
        "本研究旨在",
        "旨在",
        "目的是",
        "本研究目的",
        "objective",
        "purpose",
        "this study aims",
    ),
    "population": (
        "研究对象",
        "调查对象",
        "参与者",
        "被试",
        "样本",
        "大学生",
        "高校学生",
        "participants",
        "sample",
        "college students",
        "university students",
    ),
    "methods": (
        "研究方法",
        "采用",
        "问卷",
        "量表",
        "实验",
        "随机对照",
        "横断面",
        "统计分析",
        "回归",
        "method",
        "questionnaire",
        "randomized controlled trial",
        "cross-sectional",
        "statistical analysis",
    ),
    "results": (
        "研究结果",
        "结果表明",
        "结果显示",
        "结果发现",
        "显著",
        "相关",
        "result",
        "results",
        "found that",
        "significant",
    ),
    "conclusions": (
        "研究结论",
        "本研究结论",
        "结论与建议",
        "结论",
        "conclusion",
        "conclusions",
        "in conclusion",
    ),
    "limitations": (
        "研究局限",
        "局限性",
        "局限",
        "不足",
        "未来研究",
        "limitation",
        "limitations",
        "future research",
    ),
}

POSITIVE_TERMS = (
    "正相关",
    "显著提高",
    "显著改善",
    "促进",
    "正向影响",
    "增加",
    "提升",
    "positive",
    "improve",
    "increase",
    "beneficial",
    "associated with higher",
)
NEGATIVE_TERMS = (
    "负相关",
    "显著降低",
    "减少",
    "反向影响",
    "negative",
    "decrease",
    "associated with lower",
    "barrier",
)
NULL_TERMS = (
    "无显著",
    "不显著",
    "没有显著",
    "no significant",
    "not significant",
)


class PaperReader:
    """Agent 2：从真实 PDF 证据中提取结构化论文信息。"""

    def __init__(self, store: ResearchMindStore) -> None:
        self.store = store

    def read_paper(self, paper_id: str) -> PaperAnalysis:
        paper = self.store.get_paper(paper_id)
        if paper is None:
            raise ValueError(f"论文不存在：{paper_id}")
        evidence = self.store.list_evidence(paper_id)
        if not evidence:
            analysis = PaperAnalysis(
                paper_id=paper_id,
                extraction_mode="本地规则提取",
                status="证据不足：该论文尚未导入可检索正文。",
            )
            self.store.save_analysis(analysis)
            return analysis

        fields = {
            field: _select_evidence(evidence, keywords, limit=2)
            for field, keywords in FIELD_KEYWORDS.items()
        }
        filled = sum(bool(value) for value in fields.values())
        analysis = PaperAnalysis(
            paper_id=paper_id,
            purpose=fields["purpose"],
            population=fields["population"],
            methods=fields["methods"],
            results=fields["results"],
            conclusions=fields["conclusions"],
            limitations=fields["limitations"],
            extraction_mode="本地规则提取，保留原文与页码",
            status=(
                f"已从原文定位 {filled}/6 个结构字段；"
                "缺失字段标记为证据不足。"
            ),
        )
        self.store.save_analysis(analysis)
        return analysis

    def read_domain(
        self,
        domain_id: str,
    ) -> tuple[PaperAnalysis, ...]:
        """读取领域内全部论文，便于增量更新。"""

        return tuple(
            self.read_paper(paper.paper_id)
            for paper in self.store.list_papers(domain_id)
        )


class AcademicDebateAnalyzer:
    """Agent 3：比较不同论文的真实观点、研究设计和结论边界。"""

    def __init__(self, store: ResearchMindStore) -> None:
        self.store = store

    def analyze(
        self,
        domain_id: str,
        topic: str,
        *,
        evidence_limit: int = 8,
        save: bool = True,
    ) -> DebateAnalysis:
        clean_topic = topic.strip()
        if not clean_topic:
            raise ValueError("争议主题不能为空。")
        evidence = self.store.list_domain_evidence(domain_id)
        selected = _rank_evidence_for_query(
            evidence,
            clean_topic,
            limit=evidence_limit,
        )
        if not selected:
            selected = _representative_evidence(evidence, evidence_limit)

        viewpoints = _build_viewpoints(selected)
        papers = {
            paper.paper_id: paper for paper in self.store.list_papers(domain_id)
        }
        analyses = {
            paper_id: self.store.get_analysis(paper_id)
            for paper_id in papers
        }
        stances = Counter(
            viewpoint.stance
            for viewpoint in viewpoints
            if viewpoint.stance in {"支持", "不支持", "无显著关系"}
        )
        contradictory = (
            stances.get("支持", 0) > 0
            and (
                stances.get("不支持", 0) > 0
                or stances.get("无显著关系", 0) > 0
            )
        )
        distinct_supporters = {
            paper_id
            for viewpoint in viewpoints
            for paper_id in viewpoint.supporting_papers
        }

        if (
            len(distinct_supporters) >= 2
            and len(stances) == 1
            and next(iter(stances.values()), 0) >= 2
        ):
            consensus_summary = (
                f"初步共识（待系统综述核验）：当前 {len(distinct_supporters)} "
                f"篇论文在“{clean_topic}”上呈现相同方向，仍需核对效应大小、"
                "样本代表性和研究设计。"
            )
        else:
            consensus_summary = (
                "证据不足，暂不能判定学术共识。当前结果只反映本次检索到的"
                "有限原文片段。"
            )

        if contradictory:
            disagreement_summary = (
                "发现可验证的方向差异，但相反结论可能来自样本、设计、测量或"
                "统计模型差异，不能仅凭片段判定谁对谁错。"
            )
        else:
            disagreement_summary = (
                "本次证据未发现可验证的相反结论；这不等于该领域不存在争议。"
            )

        possible_reasons = _method_differences(
            selected,
            analyses,
            papers,
        )
        limitations = _collect_limitations(analyses, papers, limit=6)
        if len(distinct_supporters) < 3:
            research_gaps = (
                "当前主题下可核验论文少于 3 篇，研究空白判断证据不足。",
            )
        else:
            research_gaps = (
                "需要进一步核验不同样本、测量工具和统计模型能否解释结论差异。",
                "当前分析基于原文片段，不能替代系统综述或元分析。",
            )
        evidence_strength = _evidence_strength(analyses, papers)
        debate = DebateAnalysis(
            topic=clean_topic,
            viewpoints=viewpoints,
            possible_reasons=possible_reasons,
            limitations=limitations,
            evidence_strength=evidence_strength,
            research_gaps=research_gaps,
            consensus_summary=consensus_summary,
            disagreement_summary=disagreement_summary,
        )
        if save:
            self.store.save_debate(domain_id, debate)
        return debate


class KnowledgeSynthesizer:
    """Agent 4：从论文库和争议结果构建领域知识体系。"""

    def __init__(self, store: ResearchMindStore) -> None:
        self.store = store

    def synthesize(
        self,
        domain_id: str,
        *,
        debate: DebateAnalysis | None = None,
    ) -> ResearchSynthesis:
        papers = self.store.list_papers(domain_id)
        if not papers:
            raise ValueError("当前领域没有论文，无法生成知识体系。")
        papers_by_id = {paper.paper_id: paper for paper in papers}
        analyses = {
            paper.paper_id: self.store.get_analysis(paper.paper_id)
            for paper in papers
        }
        debate = debate or self.store.latest_debate(domain_id)
        concepts = _extract_concepts(papers)
        theory_framework = _build_framework_chain(analyses, papers)
        consensus = (
            (debate.consensus_summary,)
            if debate
            else (
                "尚未执行学术分歧分析，学术共识判断证据不足。",
            )
        )
        disagreements = (
            (
                debate.disagreement_summary,
                *debate.research_gaps,
            )
            if debate
            else (
                "尚未执行学术分歧分析，不能确认或否认领域争议。",
            )
        )
        limitations = _collect_limitations(
            analyses,
            papers_by_id,
            limit=8,
        )
        gaps = _coverage_gaps(analyses, len(papers))
        innovations = tuple(
            f"候选创新方向：围绕“{gap.removesuffix('。')}”设计可核验研究。"
            for gap in gaps[:4]
        )
        nodes, edges = _build_knowledge_graph(
            papers,
            analyses,
            concepts,
            debate,
        )
        synthesis = ResearchSynthesis(
            domain_id=domain_id,
            concepts=concepts,
            theory_framework=theory_framework,
            consensus=consensus,
            disagreements=disagreements,
            limitations=limitations,
            research_gaps=gaps,
            innovation_directions=innovations,
            source_paper_ids=tuple(paper.paper_id for paper in papers),
            generated_at=_now(),
            nodes=nodes,
            edges=edges,
            mode="本地证据归纳（未把假设写成学术共识）",
        )
        self.store.save_synthesis(synthesis)
        return synthesis


class LearningAssessmentAgent:
    """Agent 5：依据论文证据自动组卷并按评分细则评分。"""

    def __init__(
        self,
        store: ResearchMindStore,
        llm_service: LLMService | None = None,
    ) -> None:
        self.store = store
        self.llm_service = llm_service or LLMService()

    def generate_question_set(
        self,
        domain_id: str,
        *,
        question_types: Iterable[str] | None = None,
        debate_topic: str = "体育锻炼行为与体质健康",
    ) -> tuple[AssessmentQuestion, ...]:
        """每种题型至少生成一题；证据不足的题型会跳过并保持可解释。"""

        papers = self.store.list_papers(domain_id)
        if not papers:
            raise ValueError("当前领域没有论文，无法生成测试题。")
        analyses = []
        for paper in papers:
            analysis = self.store.get_analysis(paper.paper_id)
            if analysis is None:
                analysis = PaperReader(self.store).read_paper(paper.paper_id)
            analyses.append((paper, analysis))
        evidence_pool = self.store.list_domain_evidence(domain_id)
        token = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        requested = tuple(question_types or _default_question_types())

        generators = {
            "单项选择题": self._single_choice,
            "多项选择题": self._multiple_choice,
            "判断题": self._true_false,
            "简答题": self._short_answer,
            "论述题": self._essay_question,
            "文献对比分析题": self._comparison_question,
        }
        questions: list[AssessmentQuestion] = []
        for question_type in requested:
            generator = generators.get(question_type)
            if generator is None:
                continue
            question = generator(
                domain_id=domain_id,
                papers=papers,
                analyses=analyses,
                evidence_pool=evidence_pool,
                token=token,
                debate_topic=debate_topic,
            )
            if question is not None:
                questions.append(question)
        if not questions:
            raise ValueError("当前文献证据不足，无法生成可追溯题目。")
        self.store.save_questions(tuple(questions))
        return tuple(questions)

    def score(
        self,
        question: AssessmentQuestion,
        user_answer: str,
    ) -> ScoreResult:
        if question.is_objective:
            return self._score_objective(question, user_answer)
        return self._score_subjective(question, user_answer)

    def _single_choice(
        self,
        **context: Any,
    ) -> AssessmentQuestion | None:
        selected = _select_single_choice_evidence(
            context["analyses"],
            ("purpose", "population", "methods", "results"),
        )
        if selected is None:
            return None
        paper, target_field, correct_evidence = selected
        distractors = _option_distractors(
            context["evidence_pool"],
            excluded_quote=correct_evidence.quote,
            excluded_document_ids={correct_evidence.document_id},
            limit=3,
        )
        if len(distractors) < 3:
            return None

        correct_text = chinese_evidence_option_text(
            correct_evidence,
            designated=True,
            ordinal=1,
        )
        option_texts = [
            correct_text,
            *(
                chinese_evidence_option_text(
                    item,
                    designated=False,
                    ordinal=index,
                )
                for index, item in enumerate(distractors[:3], start=1)
            ),
        ]
        options = tuple(
            _deterministic_unique(
                option_texts,
                token=context["token"],
            )[:4]
        )
        if len(options) < 4 or correct_text not in options:
            return None
        correct_index = options.index(correct_text)
        source_evidence = (correct_evidence, *distractors[:3])
        return AssessmentQuestion(
            question_id=_question_id(
                context["domain_id"],
                "single",
                context["token"],
                correct_evidence.chunk_id,
            ),
            domain_id=context["domain_id"],
            question_type="单项选择题",
            prompt=(
                f"阅读论文《{paper.title}》。下列哪一项最直接对应其"
                f"“{FIELD_LABELS[target_field]}”相关原文证据？"
            ),
            options=options,
            standard_answer=_letter(correct_index),
            rubric=(
                RubricCriterion(
                    label="选项判断",
                    description="选择与指定论文、字段和页码直接对应的证据标签。",
                    points=2.0,
                ),
            ),
            knowledge_tags=(FIELD_LABELS[target_field], "原文定位"),
            difficulty="基础",
            source_paper_ids=(paper.paper_id,),
            evidence=source_evidence,
            created_at=_now(),
        )

    def _multiple_choice(
        self,
        **context: Any,
    ) -> AssessmentQuestion | None:
        selected = _select_multiple_choice_evidence(
            context["analyses"],
        )
        if selected is None:
            return None
        paper, correct_evidence = selected
        distractors = _option_distractors(
            context["evidence_pool"],
            excluded_quote=" ".join(
                evidence.quote for evidence in correct_evidence
            ),
            excluded_document_ids={correct_evidence[0].document_id},
            limit=3,
        )
        if len(distractors) < 2:
            return None

        correct_texts = [
            chinese_evidence_option_text(
                evidence,
                designated=True,
                ordinal=index,
            )
            for index, evidence in enumerate(correct_evidence, start=1)
        ]
        distractor_texts = [
            chinese_evidence_option_text(
                evidence,
                designated=False,
                ordinal=index,
            )
            for index, evidence in enumerate(distractors[:2], start=1)
        ]
        options = tuple(
            _deterministic_unique(
                [*correct_texts, *distractor_texts],
                token=context["token"],
            )[:4]
        )
        answer_indexes = [
            index
            for index, option in enumerate(options)
            if option in correct_texts
        ]
        if len(options) < 4 or len(answer_indexes) < 2:
            return None
        return AssessmentQuestion(
            question_id=_question_id(
                context["domain_id"],
                "multiple",
                context["token"],
                *(item.chunk_id for item in correct_evidence),
            ),
            domain_id=context["domain_id"],
            question_type="多项选择题",
            prompt=(
                f"阅读论文《{paper.title}》。下列哪些中文证据标签可作为其"
                "研究结果或研究结论的原文证据？（多选）"
            ),
            options=options,
            standard_answer=",".join(_letter(i) for i in answer_indexes),
            rubric=(
                RubricCriterion(
                    label="多选完整性",
                    description="全部选中指定论文的正确证据标签，且未选择其他论文。",
                    points=3.0,
                ),
            ),
            knowledge_tags=("研究结果", "研究结论", "证据辨识"),
            difficulty="进阶",
            source_paper_ids=(paper.paper_id,),
            evidence=(*correct_evidence, *distractors[:2]),
            created_at=_now(),
        )

    def _true_false(
        self,
        **context: Any,
    ) -> AssessmentQuestion | None:
        papers: tuple[PaperRecord, ...] = context["papers"]
        evidence_pool: tuple[EvidenceRef, ...] = context["evidence_pool"]
        paper = papers[_stable_index(context["token"], len(papers))]
        paper_evidence = tuple(
            item
            for item in evidence_pool
            if item.document_id == paper.paper_id
        )
        if not paper_evidence:
            return None
        evidence = paper_evidence[
            _stable_index(context["token"] + "evidence", len(paper_evidence))
        ]
        make_false = _stable_index(context["token"], 2) == 1
        other_paper = next(
            (item for item in papers if item.paper_id != paper.paper_id),
            None,
        )
        if make_false and other_paper is None:
            make_false = False
        cited_title = other_paper.title if make_false and other_paper else paper.title
        is_true = not make_false
        return AssessmentQuestion(
            question_id=_question_id(
                context["domain_id"],
                "true_false",
                context["token"],
                evidence.chunk_id,
                str(make_false),
            ),
            domain_id=context["domain_id"],
            question_type="判断题",
            prompt=(
                f"原文片段“{_clip(evidence.quote, 220)}”来自"
                f"论文《{cited_title}》第 {evidence.page_number} 页。"
                "请判断该说法是否正确。"
            ),
            options=("正确", "错误"),
            standard_answer="正确" if is_true else "错误",
            rubric=(
                RubricCriterion(
                    label="来源核验",
                    description="判断原文、论文题名和页码是否一致。",
                    points=2.0,
                ),
            ),
            knowledge_tags=("来源核验", "学术规范"),
            difficulty="基础",
            source_paper_ids=(paper.paper_id,),
            evidence=(evidence,),
            created_at=_now(),
        )

    def _short_answer(
        self,
        **context: Any,
    ) -> AssessmentQuestion | None:
        paper, analysis = _select_analysis_field(
            context["analyses"],
            ("methods", "results", "conclusions"),
        )
        field = _field_with_evidence(
            analysis,
            ("methods", "results", "conclusions"),
        )
        if field is None:
            return None
        evidence = analysis.field_evidence(field)
        return AssessmentQuestion(
            question_id=_question_id(
                context["domain_id"],
                "short",
                context["token"],
                paper.paper_id,
                field,
            ),
            domain_id=context["domain_id"],
            question_type="简答题",
            prompt=(
                f"请依据《{paper.title}》原文，简述其"
                f"{FIELD_LABELS[field]}。回答应尽量使用论文中的核心概念。"
            ),
            options=(),
            standard_answer=_clip(evidence[0].quote, 500),
            rubric=(
                RubricCriterion(
                    label="核心内容",
                    description="准确复述该论文对应字段的核心信息。",
                    points=5.0,
                    expected_keywords=_keywords_from_evidence(evidence),
                ),
                RubricCriterion(
                    label="来源边界",
                    description="不加入论文原文和答题要求之外的新结论。",
                    points=2.0,
                    expected_keywords=("论文", "原文", "证据"),
                ),
                RubricCriterion(
                    label="概念准确",
                    description="关键研究概念使用准确，无明显概念替换。",
                    points=3.0,
                    expected_keywords=_keywords_from_evidence(evidence),
                ),
            ),
            knowledge_tags=(FIELD_LABELS[field], "文献理解"),
            difficulty="进阶",
            source_paper_ids=(paper.paper_id,),
            evidence=evidence,
            created_at=_now(),
            scoring_mode="大模型评分优先，失败时按本地关键词细则",
        )

    def _essay_question(
        self,
        **context: Any,
    ) -> AssessmentQuestion | None:
        candidates = [
            (paper, analysis)
            for paper, analysis in context["analyses"]
            if analysis.methods and (analysis.results or analysis.conclusions)
        ]
        if len(candidates) < 2:
            return None
        first, second = candidates[:2]
        evidence = (
            *first[1].methods[:1],
            *first[1].conclusions[:1],
            *second[1].methods[:1],
            *second[1].conclusions[:1],
        )
        if len(evidence) < 3:
            return None
        return AssessmentQuestion(
            question_id=_question_id(
                context["domain_id"],
                "essay",
                context["token"],
                *(item.chunk_id for item in evidence),
            ),
            domain_id=context["domain_id"],
            question_type="论述题",
            prompt=(
                f"比较《{first[0].title}》与《{second[0].title}》在研究方法、"
                "主要结果或结论上的异同，并说明为什么不能仅凭这两篇论文"
                "宣称存在普遍学术共识。"
            ),
            options=(),
            standard_answer=_build_comparison_standard_answer(
                (
                    (first[0].title, first[1], first[1].methods[0]),
                    (
                        second[0].title,
                        second[1],
                        second[1].methods[0],
                    ),
                )
            ),
            rubric=(
                RubricCriterion(
                    label="证据引用",
                    description="至少引用两项与题目直接相关的原文证据。",
                    points=4.0,
                    expected_keywords=_keywords_from_evidence(evidence),
                ),
                RubricCriterion(
                    label="比较分析",
                    description="分别比较研究方法、结果或结论，而不是只复述摘要。",
                    points=4.0,
                    expected_keywords=("方法", "结果", "结论", "比较"),
                ),
                RubricCriterion(
                    label="批判性边界",
                    description="指出样本、设计、测量或统计差异及共识边界。",
                    points=4.0,
                    expected_keywords=(
                        "样本",
                        "设计",
                        "测量",
                        "统计",
                        "不能",
                        "共识",
                    ),
                ),
            ),
            knowledge_tags=("文献对比", "批判性思维", "学术共识边界"),
            difficulty="挑战",
            source_paper_ids=(first[0].paper_id, second[0].paper_id),
            evidence=evidence,
            created_at=_now(),
            scoring_mode="大模型按细则评分优先，失败时本地细则评分",
        )

    def _comparison_question(
        self,
        **context: Any,
    ) -> AssessmentQuestion | None:
        papers: tuple[PaperRecord, ...] = context["papers"]
        if len(papers) < 2:
            return None
        evidence_pool: tuple[EvidenceRef, ...] = context["evidence_pool"]
        target = _rank_evidence_for_query(
            evidence_pool,
            context["debate_topic"],
            limit=4,
        )
        distinct = {}
        for item in target:
            distinct.setdefault(item.document_id, item)
        selected = tuple(distinct.values())[:3]
        if len(selected) < 2:
            selected = _representative_evidence(evidence_pool, limit=3)
        if len(selected) < 2:
            return None
        return AssessmentQuestion(
            question_id=_question_id(
                context["domain_id"],
                "comparison",
                context["token"],
                *(item.chunk_id for item in selected),
            ),
            domain_id=context["domain_id"],
            question_type="文献对比分析题",
            prompt=(
                f"围绕“{context['debate_topic']}”，比较下列论文片段的研究设计、"
                "结果方向与证据强度。请说明这些片段能否构成学术共识，并给出理由。"
            ),
            options=(),
            standard_answer=_build_comparison_standard_answer(
                tuple(
                    (
                        item.document_title,
                        next(
                            (
                                analysis
                                for paper, analysis in context["analyses"]
                                if paper.paper_id == item.document_id
                            ),
                            None,
                        ),
                        item,
                    )
                    for item in selected
                )
            ),
            rubric=(
                RubricCriterion(
                    label="观点与来源对应",
                    description="每个观点都说明对应论文和页码。",
                    points=4.0,
                    expected_keywords=tuple(
                        str(item.page_number) for item in selected
                    ),
                ),
                RubricCriterion(
                    label="设计与结果比较",
                    description="比较研究设计、样本或结果方向。",
                    points=3.0,
                    expected_keywords=("研究设计", "样本", "结果", "方向"),
                ),
                RubricCriterion(
                    label="争议边界",
                    description="说明证据不足时不能把相关性写成因果关系。",
                    points=3.0,
                    expected_keywords=("相关", "因果", "证据不足", "共识"),
                ),
            ),
            knowledge_tags=("对比分析", "学术争议", "证据强度"),
            difficulty="挑战",
            source_paper_ids=tuple(
                dict.fromkeys(item.document_id for item in selected)
            ),
            evidence=selected,
            created_at=_now(),
            scoring_mode="大模型按细则评分优先，失败时本地细则评分",
        )

    @staticmethod
    def _score_objective(
        question: AssessmentQuestion,
        user_answer: str,
    ) -> ScoreResult:
        answer = user_answer.strip()
        standard = question.standard_answer.strip()
        if question.question_type == "多项选择题":
            user_set = _parse_letters(answer)
            standard_set = _parse_letters(standard)
            correct = bool(user_set) and user_set == standard_set
        elif question.question_type == "单项选择题":
            correct = _parse_letters(answer) == _parse_letters(standard)
        else:
            correct = answer == standard
        score = question.max_score if correct else 0.0
        feedback = (
            "回答正确。标准答案与原文证据一致。"
            if correct
            else f"回答不正确。标准答案为 {standard}。"
        )
        return ScoreResult(
            score=score,
            max_score=question.max_score,
            feedback=feedback,
            criteria=(
                CriterionScore(
                    label=question.rubric[0].label,
                    score=score,
                    max_score=question.max_score,
                    reason=feedback,
                ),
            ),
            mode="客观题自动评分",
            correct=correct,
            user_answer=user_answer,
        )

    def _score_subjective(
        self,
        question: AssessmentQuestion,
        user_answer: str,
    ) -> ScoreResult:
        clean_answer = user_answer.strip()
        if not clean_answer:
            return _empty_subjective_score(question, clean_answer)
        if self.llm_service.is_configured:
            try:
                return self._score_with_llm(question, clean_answer)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                pass
        return _local_rubric_score(question, clean_answer)

    def _score_with_llm(
        self,
        question: AssessmentQuestion,
        user_answer: str,
    ) -> ScoreResult:
        rubric_text = "\n".join(
            f"- {item.label}：最多 {item.points} 分；{item.description}"
            + (
                f"；关键内容可包含：{'、'.join(item.expected_keywords)}"
                if item.expected_keywords
                else ""
            )
            for item in question.rubric
        )
        evidence_text = "\n".join(
            f"- {item.citation}：{item.quote}"
            for item in question.evidence
        )
        prompt = (
            "请严格按评分细则批改下面的 ResearchMind 题目。"
            "评分依据必须是内容与证据对应关系，不得因为语言流畅额外加分。"
            "学生使用同义转述或自己的比较句完成细则要求时，应按内容给分。"
            "比较类题目中，学生逐篇说明研究设计、样本、结果方向、测量或统计方法，"
            "并明确写出论文之间的相同点或不同点，就应视为完成比较，不要求照抄"
            "标准答案措辞。"
            "只输出 JSON，不要 Markdown。格式："
            '{"criteria":[{"label":"细则名称","score":0,"reason":"依据"}],'
            '"feedback":"总体反馈"}。\n\n'
            f"题目：{question.prompt}\n"
            f"标准答案：{question.standard_answer}\n"
            f"评分细则：\n{rubric_text}\n"
            f"来源证据：\n{evidence_text}\n"
            f"学生答案：{user_answer}"
        )
        response = self.llm_service.answer(
            prompt,
            allow_simulation=False,
            system_prompt=(
                "你是严格的体育科研课程评分器。不得编造证据，不得因表达流畅"
                "而忽略知识错误。所有得分必须能对应评分细则。"
            ),
        )
        if not response.success:
            raise ValueError(response.message)
        payload = _parse_json_object(response.answer)
        raw_criteria = payload.get("criteria")
        if not isinstance(raw_criteria, list):
            raise ValueError("大模型评分缺少 criteria。")
        by_label = {
            str(item.get("label", "")).strip(): item
            for item in raw_criteria
            if isinstance(item, dict)
        }
        criteria: list[CriterionScore] = []
        for rubric in question.rubric:
            item = by_label.get(rubric.label)
            if item is None:
                raise ValueError(f"大模型评分缺少细则：{rubric.label}")
            score = max(0.0, min(float(item.get("score", 0)), rubric.points))
            criteria.append(
                CriterionScore(
                    label=rubric.label,
                    score=score,
                    max_score=rubric.points,
                    reason=str(item.get("reason", "未提供评分理由。"))[:500],
                )
            )
        total = sum(item.score for item in criteria)
        return ScoreResult(
            score=total,
            max_score=question.max_score,
            feedback=str(payload.get("feedback", "已按评分细则完成批改。")),
            criteria=tuple(criteria),
            mode="大模型按评分细则批改",
            user_answer=user_answer,
        )


class PersonalTutorAgent:
    """Agent 6：批改记录、薄弱点识别、复习推荐和下一轮学习。"""

    def __init__(
        self,
        store: ResearchMindStore,
        assessment: LearningAssessmentAgent | None = None,
    ) -> None:
        self.store = store
        self.assessment = assessment or LearningAssessmentAgent(store)

    def grade_and_record(
        self,
        question: AssessmentQuestion,
        user_answer: str,
    ) -> ScoreResult:
        result = self.assessment.score(question, user_answer)
        self.store.record_attempt(
            domain_id=question.domain_id,
            question=question,
            user_answer=user_answer,
            score=result.score,
            feedback=result.feedback,
            mode=result.mode,
        )
        self.update_mastery(question.domain_id)
        return result

    def update_mastery(self, domain_id: str) -> dict[str, float]:
        """从全部历史答题重算掌握度，避免重复叠加同一条记录。"""

        scores = dict(BASELINE_MASTERY)
        counts = {dimension: 0 for dimension in BASELINE_MASTERY}
        attempts = tuple(reversed(self.store.list_attempts(domain_id)))
        for attempt in attempts:
            percentage = (
                100.0 * attempt.score / attempt.max_score
                if attempt.max_score > 0
                else 0.0
            )
            dimensions = _question_dimensions(attempt.question_type)
            for dimension in dimensions:
                old = scores[dimension]
                scores[dimension] = round(old * 0.72 + percentage * 0.28, 2)
                counts[dimension] += 1
        for dimension, score in scores.items():
            self.store.set_mastery(
                domain_id,
                dimension,
                score,
                counts[dimension],
            )
        return scores

    def recommend(
        self,
        domain_id: str,
        *,
        next_question_ids: tuple[str, ...] = (),
    ) -> StudyRecommendation:
        mastery = self.store.ensure_mastery(domain_id)
        weak_dimensions = tuple(
            item.dimension for item in mastery if item.score < 60.0
        )
        wrong_attempts = self.store.list_wrong_attempts(domain_id)
        tag_counts = Counter(
            tag for attempt in wrong_attempts for tag in attempt.knowledge_tags
        )
        weak_tags = tuple(tag for tag, _ in tag_counts.most_common(6))
        papers = self.store.list_papers(domain_id)
        recommended: list[str] = []
        for paper in papers:
            haystack = f"{paper.title} {' '.join(paper.keywords)}"
            if any(tag in haystack for tag in weak_tags):
                recommended.append(paper.paper_id)
        if not recommended:
            wrong_question_ids = {
                attempt.question_id for attempt in wrong_attempts
            }
            for question in self.store.list_questions(domain_id):
                if question.question_id in wrong_question_ids:
                    recommended.extend(question.source_paper_ids)
        recommended_unique = tuple(dict.fromkeys(recommended))[:5]

        review_plan: list[str] = []
        if weak_dimensions:
            review_plan.append(
                "优先复习：" + "、".join(weak_dimensions) + "。"
            )
        if weak_tags:
            review_plan.append(
                "围绕错题知识点重读原文：" + "、".join(weak_tags) + "。"
            )
        review_plan.extend(
            (
                "每道主观题先写出观点，再补充论文题名、页码和原文证据。",
                "区分相关性与因果性；样本和设计不一致时不要写成统一共识。",
                "完成下一轮测评后，回看错题本中的评分细则缺分原因。",
            )
        )
        if weak_dimensions:
            summary = (
                f"当前最需要提升的是{'、'.join(weak_dimensions)}。"
                "建议先重读推荐论文中的对应证据，再完成针对性题目。"
            )
        else:
            summary = (
                "当前五个能力维度均达到 60 分以上。"
                "建议用文献对比题和论述题继续训练证据边界与批判性分析。"
            )
        return StudyRecommendation(
            weak_dimensions=weak_dimensions,
            weak_tags=weak_tags,
            recommended_paper_ids=recommended_unique,
            review_plan=tuple(review_plan),
            next_question_ids=next_question_ids,
            summary=summary,
            generated_at=_now(),
        )


def _select_evidence(
    evidence: tuple[EvidenceRef, ...],
    keywords: tuple[str, ...],
    *,
    limit: int,
) -> tuple[EvidenceRef, ...]:
    scored: list[tuple[int, EvidenceRef]] = []
    for item in evidence:
        lowered = item.quote.lower()
        score = sum(
            lowered.count(keyword.lower()) * max(1, len(keyword))
            for keyword in keywords
            if keyword.lower() in lowered
        )
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda pair: (-pair[0], pair[1].page_number))
    selected: list[EvidenceRef] = []
    seen: set[str] = set()
    for _, item in scored:
        if item.quote in seen:
            continue
        seen.add(item.quote)
        selected.append(item)
        if len(selected) >= limit:
            break
    return tuple(selected)


def _rank_evidence_for_query(
    evidence: tuple[EvidenceRef, ...],
    query: str,
    *,
    limit: int,
) -> tuple[EvidenceRef, ...]:
    terms = _query_terms(query)
    scored: list[tuple[int, EvidenceRef]] = []
    for item in evidence:
        text = item.quote.lower()
        score = sum(3 for term in terms if term in text)
        if score:
            scored.append((score, item))
    scored.sort(
        key=lambda pair: (
            -pair[0],
            pair[1].document_title,
            pair[1].page_number,
        )
    )
    per_document: Counter[str] = Counter()
    output: list[EvidenceRef] = []
    for _, item in scored:
        if per_document[item.document_id] >= 2:
            continue
        output.append(item)
        per_document[item.document_id] += 1
        if len(output) >= limit:
            break
    return tuple(output)


def _representative_evidence(
    evidence: tuple[EvidenceRef, ...],
    limit: int,
) -> tuple[EvidenceRef, ...]:
    per_document: Counter[str] = Counter()
    output: list[EvidenceRef] = []
    for item in evidence:
        if per_document[item.document_id] >= 1:
            continue
        output.append(item)
        per_document[item.document_id] += 1
        if len(output) >= limit:
            break
    return tuple(output)


def _build_comparison_standard_answer(
    entries: tuple[tuple[str, PaperAnalysis | None, EvidenceRef], ...],
) -> str:
    """用真实原文证据组成包含逐篇分析和跨论文比较的参考答案。"""

    rows: list[dict[str, Any]] = []
    lines = [
        "比较框架（只依据下列论文原文和页码；原文未提供的字段写“证据不足”）："
    ]
    for index, (title, analysis, anchor) in enumerate(entries, start=1):
        design = _infer_study_design(title, analysis)
        methods = _comparison_field(analysis, "methods")
        population = _sample_summary(analysis)
        results = _comparison_field(analysis, "results")
        conclusions = _comparison_field(analysis, "conclusions")
        statistics = _statistical_clues(analysis)
        direction = _result_direction(analysis)
        rows.append(
            {
                "title": title,
                "design": design,
                "direction": direction,
                "statistics": statistics,
                "population": population,
            }
        )

        lines.append(
            f"{index}. 《{title}》"
            f"（证据定位起点：第 {anchor.page_number} 页）"
        )
        lines.append(f"   - 研究设计判断：{design}。")
        lines.append(
            "   - 研究对象/样本："
            + _format_comparison_field(population)
        )
        lines.append(
            "   - 研究方法："
            + _format_comparison_field(methods)
        )
        lines.append(
            "   - 研究结果："
            + _format_comparison_field(results)
        )
        lines.append(
            "   - 研究结论："
            + _format_comparison_field(conclusions)
        )
        lines.append(
            "   - 测量指标与统计方法线索："
            + ("、".join(statistics) if statistics else "证据不足")
            + "。"
        )
        lines.append(f"   - 结果方向：{direction}。")

    lines.append("跨论文比较与异同：")
    for left_index, left in enumerate(rows):
        for right_index in range(left_index + 1, len(rows)):
            right = rows[right_index]
            relation = (
                "研究设计类别相同"
                if left["design"] == right["design"]
                else "研究设计类别不同"
            )
            merge_boundary = (
                "样本、测量和分析模型仍需逐项核对，不能仅凭设计类别相同"
                "就合并结果"
                if left["design"] == right["design"]
                else "不能把不同设计、样本或测量下得到的结论直接当作同一结果的"
                "重复验证"
            )
            lines.append(
                f"- 方法异同：论文{left_index + 1}为“{left['design']}”，"
                f"论文{right_index + 1}为“{right['design']}”；"
                f"{relation}；{merge_boundary}。"
            )
            lines.append(
                f"- 结果方向异同：论文{left_index + 1}为“{left['direction']}”，"
                f"论文{right_index + 1}为“{right['direction']}”；"
                "方向相同也不等于因果关系已经确立，方向不同则还需核对"
                "样本、测量工具、研究时点和统计模型。"
            )
            left_stats = "、".join(left["statistics"]) or "证据不足"
            right_stats = "、".join(right["statistics"]) or "证据不足"
            lines.append(
                f"- 统计分析比较：论文{left_index + 1}可定位线索为"
                f"“{left_stats}”，论文{right_index + 1}为"
                f"“{right_stats}”；若统计模型或测量指标不一致，效应不能直接合并。"
            )

    lines.extend(
        (
            "- 证据不足与边界：上述比较只覆盖已定位的原文片段；样本代表性、"
            "测量信效度和统计假设检验条件仍需阅读全文核验。",
            "- 相关与因果：原文中的“相关、关联、影响”表述不能自动升级为因果结论；"
            "只有与研究设计相符时才能讨论因果解释。",
            "- 学术共识边界：当前有限论文和片段不足以宣称形成普遍学术共识，"
            "只能表述为待系统综述或元分析核验的初步证据。",
        )
    )
    return "\n".join(lines)


def _comparison_field(
    analysis: PaperAnalysis | None,
    field_name: str,
) -> tuple[str, int] | None:
    """返回字段的第一条可定位证据，缺失时保持证据不足。"""

    if analysis is None:
        return None
    evidence = analysis.field_evidence(field_name)
    if not evidence:
        return None
    if field_name == "conclusions":
        evidence = tuple(
            sorted(
                evidence,
                key=lambda item: _conclusion_marker_score(item.quote),
                reverse=True,
            )
        )
    item = evidence[0]
    if field_name == "conclusions":
        marker_position = _conclusion_marker_position(item.quote)
        if marker_position >= 0:
            return (
                _clip(item.quote[marker_position : marker_position + 320], 280),
                item.page_number,
            )
    return _clip(item.quote, 280), item.page_number


def _conclusion_marker_score(quote: str) -> int:
    """优先选择带明确结论标题的片段，避免把引言中的“表明”当结论。"""

    lowered = quote.lower()
    markers = (
        ("结论与建议", 8),
        ("研究结论", 7),
        ("6.1. 结论", 7),
        ("conclusions", 6),
        ("conclusion", 5),
        ("结论", 4),
        ("in conclusion", 4),
    )
    return sum(weight for marker, weight in markers if marker in lowered)


def _conclusion_marker_position(quote: str) -> int:
    """返回最明确结论标题的位置，找不到时返回 -1。"""

    lowered = quote.lower()
    for marker in (
        "结论与建议",
        "研究结论",
        "6.1. 结论",
        "conclusions",
        "conclusion",
        "in conclusion",
        "结论",
    ):
        position = lowered.find(marker)
        if position >= 0:
            return position
    return -1


def _sample_summary(
    analysis: PaperAnalysis | None,
) -> tuple[str, int] | None:
    """从真实证据中选择更具体的样本描述，找不到时标证据不足。"""

    if analysis is None:
        return None
    markers = (
        "调查对象",
        "研究对象",
        "共发放",
        "累计发放",
        "有效问卷",
        "study was conducted with",
        "participants were",
        "sample included",
        "a total of",
    )
    for evidence in analysis.all_evidence:
        lowered = evidence.quote.lower()
        for marker in markers:
            position = lowered.find(marker)
            if position < 0:
                continue
            excerpt = evidence.quote[position : position + 300]
            return _clip(excerpt, 280), evidence.page_number
    return None


def _format_comparison_field(
    value: tuple[str, int] | None,
) -> str:
    if value is None:
        return "证据不足。"
    quote, page_number = value
    return f"{quote}（第 {page_number} 页）"


def _infer_study_design(
    title: str,
    analysis: PaperAnalysis | None,
) -> str:
    """只依据题名和已定位原文给出保守的设计类别。"""

    evidence_text = ""
    if analysis is not None:
        evidence_text = " ".join(
            item.quote
            for field_name in ("methods", "results", "conclusions")
            for item in analysis.field_evidence(field_name)
        )
    text = f"{title} {evidence_text}".lower()
    rules = (
        (("meta-analysis", "meta analysis", "元分析"), "元分析"),
        (("systematic review", "系统综述"), "系统综述"),
        (
            ("randomized controlled trial", "随机对照试验"),
            "随机对照试验",
        ),
        (
            ("app-supported", "versus conventional", "组间比较"),
            "教学干预与组间比较",
        ),
        (
            ("cross-sectional", "横断面"),
            "横断面研究",
        ),
        (
            ("问卷调查", "questionnaire survey", "影响因素研究"),
            "问卷调查与影响因素分析",
        ),
        (
            ("cohort", "队列研究"),
            "队列研究",
        ),
    )
    for keywords, label in rules:
        if any(keyword in text for keyword in keywords):
            return label
    return "原文可定位片段不足以归类研究设计（证据不足）"


def _statistical_clues(
    analysis: PaperAnalysis | None,
) -> tuple[str, ...]:
    """提取原文中出现的统计或测量术语，保留可核验线索。"""

    if analysis is None:
        return ()
    text = " ".join(
        item.quote
        for field_name in ("methods", "results", "conclusions")
        for item in analysis.field_evidence(field_name)
    ).lower()
    candidates = (
        "paired-samples t tests",
        "独立样本 t 检验",
        "t 检验",
        "线性回归",
        "linear regression",
        "卡方检验",
        "chi-square",
        "方差分析",
        "anova",
        "holm–bonferroni",
        "holm-bonferroni",
        "问卷",
        "量表",
    )
    return tuple(
        dict.fromkeys(
            keyword
            for keyword in candidates
            if keyword.lower() in text
        )
    )


def _result_direction(analysis: PaperAnalysis | None) -> str:
    """按原文关键词描述结果方向，不据此声称因果关系。"""

    if analysis is None:
        return "证据不足，无法判断"
    text = " ".join(
        item.quote
        for field_name in ("results", "conclusions")
        for item in analysis.field_evidence(field_name)
    )
    lowered = text.lower()
    positive = any(term in lowered for term in POSITIVE_TERMS)
    negative = any(term in lowered for term in NEGATIVE_TERMS)
    no_significant = any(term in lowered for term in NULL_TERMS)
    if sum((positive, negative, no_significant)) >= 2:
        return (
            "原文同时包含正向、负向或未达显著的混合结果"
            "（不能概括为单一方向）"
        )
    if no_significant:
        return "原文报告无显著关系"
    if negative:
        return "原文出现负向关联或降低方向"
    if positive:
        return "原文出现正向关联或改善方向"
    return "原文片段不足以判断方向（证据不足）"


def _build_viewpoints(
    evidence: tuple[EvidenceRef, ...],
) -> tuple[Viewpoint, ...]:
    viewpoints: list[Viewpoint] = []
    seen_quotes: set[str] = set()
    for item in evidence:
        normalized_quote = _clip(item.quote, 320)
        if normalized_quote in seen_quotes:
            continue
        seen_quotes.add(normalized_quote)
        stance = _classify_stance(item.quote)
        viewpoints.append(
            Viewpoint(
                claim=f"原文证据：{normalized_quote}",
                stance=stance,
                supporting_papers=(item.document_id,),
                evidence=(item,),
            )
        )
    return tuple(viewpoints)


def _classify_stance(quote: str) -> str:
    lowered = quote.lower()
    if any(term in lowered for term in NULL_TERMS):
        return "无显著关系"
    if any(term in lowered for term in NEGATIVE_TERMS):
        return "不支持"
    if any(term in lowered for term in POSITIVE_TERMS):
        return "支持"
    return "待评估"


def _method_differences(
    evidence: tuple[EvidenceRef, ...],
    analyses: dict[str, PaperAnalysis | None],
    papers: dict[str, PaperRecord],
) -> tuple[str, ...]:
    differences: list[str] = []
    for paper_id in dict.fromkeys(item.document_id for item in evidence):
        paper = papers.get(paper_id)
        analysis = analyses.get(paper_id)
        if paper is None or analysis is None:
            continue
        if analysis.methods:
            method = _clip(analysis.methods[0].quote, 180)
            differences.append(
                f"{paper.title}的方法证据：{method}"
                f"（第 {analysis.methods[0].page_number} 页）"
            )
        else:
            differences.append(
                f"{paper.title}：未提取到明确方法证据，设计差异待评估。"
            )
    if len(differences) < 2:
        differences.append(
            "当前可比较的方法证据不足，无法判断差异是否解释结论变化。"
        )
    return tuple(differences)


def _collect_limitations(
    analyses: dict[str, PaperAnalysis | None],
    papers: dict[str, PaperRecord],
    *,
    limit: int,
) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for paper_id, analysis in analyses.items():
        if analysis is None or not analysis.limitations:
            continue
        paper = papers[paper_id]
        for item in analysis.limitations:
            quote = _clip(item.quote, 220)
            if quote in seen:
                continue
            seen.add(quote)
            output.append(
                f"{paper.title}，第 {item.page_number} 页：{quote}"
            )
            if len(output) >= limit:
                return tuple(output)
    if not output:
        return ("当前论文库未提取到明确的研究局限，需阅读全文人工核验。",)
    return tuple(output)


def _evidence_strength(
    analyses: dict[str, PaperAnalysis | None],
    papers: dict[str, PaperRecord],
) -> str:
    method_text = " ".join(
        item.quote.lower()
        for analysis in analyses.values()
        if analysis is not None
        for item in analysis.methods
    )
    if "随机对照" in method_text or "randomized controlled trial" in method_text:
        return "待评估：当前库包含随机对照研究证据，但未进行偏倚风险评价。"
    if "横断面" in method_text or "cross-sectional" in method_text:
        return "待评估：当前证据以横断面研究为主，不能直接推断因果关系。"
    return "待评估：尚未完成系统化的研究设计与偏倚风险评价。"


def _extract_concepts(papers: tuple[PaperRecord, ...]) -> tuple[str, ...]:
    counts: Counter[str] = Counter()
    for paper in papers:
        counts.update(
            keyword.strip()
            for keyword in paper.keywords
            if keyword.strip() and keyword.strip() not in {"待补充", "无"}
        )
    return tuple(
        concept for concept, count in counts.most_common(12) if count >= 1
    )


def _build_framework_chain(
    analyses: dict[str, PaperAnalysis | None],
    papers: tuple[PaperRecord, ...],
) -> tuple[str, ...]:
    total = len(papers)
    labels = (
        ("研究对象与场景", "population"),
        ("研究设计与方法", "methods"),
        ("研究结果", "results"),
        ("研究结论", "conclusions"),
        ("研究局限", "limitations"),
    )
    chain: list[str] = []
    for label, field_name in labels:
        supported = sum(
            bool(analysis and analysis.field_evidence(field_name))
            for analysis in analyses.values()
        )
        chain.append(
            f"{label}：当前 {supported}/{total} 篇论文有可定位证据。"
        )
    return tuple(chain)


def _coverage_gaps(
    analyses: dict[str, PaperAnalysis | None],
    total: int,
) -> tuple[str, ...]:
    gaps: list[str] = []
    for field_name, label in FIELD_LABELS.items():
        supported = sum(
            bool(analysis and analysis.field_evidence(field_name))
            for analysis in analyses.values()
        )
        if supported < total:
            gaps.append(
                f"当前论文库仅 {supported}/{total} 篇能定位到明确{label}证据，"
                "该结构字段存在证据覆盖缺口。"
            )
    if not gaps:
        gaps.append(
            "当前库六个结构字段均有证据覆盖，但样本代表性和因果推断边界仍需核验。"
        )
    return tuple(gaps)


def _build_knowledge_graph(
    papers: tuple[PaperRecord, ...],
    analyses: dict[str, PaperAnalysis | None],
    concepts: tuple[str, ...],
    debate: DebateAnalysis | None,
) -> tuple[tuple[KnowledgeNode, ...], tuple[KnowledgeEdge, ...]]:
    nodes: list[KnowledgeNode] = []
    edges: list[KnowledgeEdge] = []
    paper_nodes: dict[str, str] = {}
    for paper in papers:
        node_id = f"paper:{paper.paper_id}"
        paper_nodes[paper.paper_id] = node_id
        evidence_count = len(
            analyses[paper.paper_id].all_evidence
            if analyses.get(paper.paper_id)
            else ()
        )
        nodes.append(
            KnowledgeNode(
                node_id=node_id,
                label=paper.title,
                node_type="论文",
                description=paper.citation,
                evidence_count=evidence_count,
            )
        )

    for concept in concepts:
        linked_papers = [
            paper
            for paper in papers
            if concept.lower() in (
                f"{paper.title} {' '.join(paper.keywords)}".lower()
            )
        ]
        if not linked_papers:
            continue
        concept_id = "concept:" + _safe_node_key(concept)
        nodes.append(
            KnowledgeNode(
                node_id=concept_id,
                label=concept,
                node_type="概念",
                description=(
                    f"由 {len(linked_papers)} 篇论文的真实题名或关键词支持。"
                ),
                evidence_count=len(linked_papers),
            )
        )
        for paper in linked_papers:
            edges.append(
                KnowledgeEdge(
                    source_id=paper_nodes[paper.paper_id],
                    target_id=concept_id,
                    relation="论文涉及概念",
                    evidence_count=1,
                )
            )

    if debate and debate.viewpoints:
        debate_id = "debate:" + _safe_node_key(debate.topic)
        nodes.append(
            KnowledgeNode(
                node_id=debate_id,
                label=debate.topic,
                node_type="学术议题",
                description=debate.disagreement_summary,
                evidence_count=sum(
                    len(viewpoint.evidence) for viewpoint in debate.viewpoints
                ),
            )
        )
        for viewpoint in debate.viewpoints:
            for paper_id in viewpoint.supporting_papers:
                if paper_id in paper_nodes:
                    edges.append(
                        KnowledgeEdge(
                            source_id=paper_nodes[paper_id],
                            target_id=debate_id,
                            relation=f"提供{viewpoint.stance}观点证据",
                            evidence_count=len(viewpoint.evidence),
                        )
                    )
    return tuple(nodes), tuple(edges)


def _select_analysis_field(
    analyses: list[tuple[PaperRecord, PaperAnalysis]],
    fields: tuple[str, ...],
) -> tuple[PaperRecord, PaperAnalysis]:
    for paper, analysis in analyses:
        if any(analysis.field_evidence(field) for field in fields):
            return paper, analysis
    paper, analysis = analyses[0]
    return paper, analysis


def _field_with_evidence(
    analysis: PaperAnalysis,
    fields: tuple[str, ...],
) -> str | None:
    return next(
        (field for field in fields if analysis.field_evidence(field)),
        None,
    )


def _select_single_choice_evidence(
    analyses: list[tuple[PaperRecord, PaperAnalysis]],
    fields: tuple[str, ...],
) -> tuple[PaperRecord, str, EvidenceRef] | None:
    """优先选择带中文原文的证据，找不到时仍使用可核验原文。"""

    candidates: list[tuple[int, int, PaperRecord, str, EvidenceRef]] = []
    order = 0
    for paper, analysis in analyses:
        for field in fields:
            for evidence in analysis.field_evidence(field):
                candidates.append(
                    (
                        _chinese_character_count(evidence.quote),
                        -order,
                        paper,
                        field,
                        evidence,
                    )
                )
                order += 1
    if not candidates:
        return None
    _, _, paper, field, evidence = max(candidates, key=lambda item: item[:2])
    return paper, field, evidence


def _select_multiple_choice_evidence(
    analyses: list[tuple[PaperRecord, PaperAnalysis]],
) -> tuple[PaperRecord, tuple[EvidenceRef, ...]] | None:
    """为一篇论文选择两条可区分的结果或结论证据。"""

    candidates: list[
        tuple[int, int, PaperRecord, tuple[EvidenceRef, ...]]
    ] = []
    for order, (paper, analysis) in enumerate(analyses):
        unique: dict[str, EvidenceRef] = {}
        for field in ("results", "conclusions"):
            for evidence in analysis.field_evidence(field):
                unique.setdefault(evidence.chunk_id, evidence)
        selected = sorted(
            unique.values(),
            key=lambda evidence: -_chinese_character_count(evidence.quote),
        )[:2]
        if len(selected) < 2:
            continue
        candidates.append(
            (
                sum(_chinese_character_count(item.quote) for item in selected),
                -order,
                paper,
                tuple(selected),
            )
        )
    if not candidates:
        return None
    _, _, paper, evidence = max(candidates, key=lambda item: item[:2])
    return paper, evidence


def _chinese_character_count(value: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", value or ""))


def chinese_only_excerpt(value: str, max_chars: int = 120) -> str:
    """移除网址、英文和 Markdown 符号，只保留可展示的中文片段。"""

    cleaned = re.sub(
        r"(?:https?://|www\.|doi\s*:?\s*)\S+|10\.\d{4,9}/\S+",
        " ",
        value or "",
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[\w.+-]+@[\w.-]+\.\w+", " ", cleaned)
    cleaned = re.sub(
        r"[^\u4e00-\u9fff0-9，。；：、！？（）《》“”‘’·%\-—\s]",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ，。；：、-—_")
    if _chinese_character_count(cleaned) < 6:
        return ""
    return _clip(cleaned, max_chars)


def _infer_evidence_field(value: str) -> str | None:
    normalized = (value or "").lower()
    for field, keywords in FIELD_KEYWORDS.items():
        if any(keyword.lower() in normalized for keyword in keywords):
            return field
    return None


def _chinese_number(value: int) -> str:
    numerals = "零一二三四五六七八九十"
    if 0 <= value <= 10:
        return numerals[value]
    return str(value)


def chinese_evidence_option_text(
    evidence: EvidenceRef,
    *,
    designated: bool,
    ordinal: int,
) -> str:
    """把原文证据转成不含网址和英文的中文选项标签。"""

    source = "本题指定论文" if designated else "其他论文"
    excerpt = chinese_only_excerpt(evidence.quote)
    if excerpt:
        return f"{source}第 {evidence.page_number} 页：{excerpt}"
    field = _infer_evidence_field(evidence.quote)
    clue = f"{FIELD_LABELS[field]}线索" if field else "可定位原文证据"
    return (
        f"{source}第 {evidence.page_number} 页："
        f"含{clue}（证据{_chinese_number(ordinal)}）"
    )


def _option_distractors(
    evidence: tuple[EvidenceRef, ...],
    *,
    excluded_quote: str,
    excluded_document_ids: set[str] | None = None,
    limit: int,
) -> tuple[EvidenceRef, ...]:
    excluded = set(excluded_quote.split())
    excluded_documents = excluded_document_ids or set()
    output: list[EvidenceRef] = []
    seen_quotes: set[str] = set()
    for item in evidence:
        if item.document_id in excluded_documents:
            continue
        normalized = item.quote.strip()
        if not normalized or normalized in seen_quotes:
            continue
        overlap = len(excluded.intersection(normalized.split()))
        if overlap > max(8, len(excluded) // 2):
            continue
        seen_quotes.add(normalized)
        output.append(item)
        if len(output) >= limit:
            break
    return tuple(output)


def _deterministic_unique(
    values: list[str] | tuple[str, ...],
    *,
    token: str,
) -> list[str]:
    unique = list(dict.fromkeys(value.strip() for value in values if value.strip()))
    return sorted(unique, key=lambda value: _stable_hash(token + value))


def _keywords_from_evidence(
    evidence: tuple[EvidenceRef, ...],
) -> tuple[str, ...]:
    text = " ".join(item.quote for item in evidence)
    candidates = (
        "体育锻炼",
        "体质健康",
        "大学生",
        "问卷",
        "随机对照",
        "横断面",
        "相关",
        "影响",
        "方法",
        "结果",
        "结论",
        "样本",
        "测量",
        "physical activity",
        "physical fitness",
        "college students",
        "randomized controlled trial",
        "cross-sectional",
        "questionnaire",
        "result",
        "conclusion",
    )
    found = tuple(
        keyword for keyword in candidates if keyword.lower() in text.lower()
    )
    return found or ("论文", "原文", "证据")


def _parse_letters(value: str) -> frozenset[str]:
    return frozenset(
        letter.upper()
        for letter in re.findall(r"[A-H]", value.upper())
    )


def _letter(index: int) -> str:
    return chr(ord("A") + index)


def _clip(value: str, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "…"


def _query_terms(query: str) -> tuple[str, ...]:
    terms = [
        term.lower()
        for term in re.findall(r"[A-Za-z]{3,}|[\u4e00-\u9fff]{2,}", query)
        if term.strip()
    ]
    if not terms:
        return tuple(query.lower())
    return tuple(dict.fromkeys(terms))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _stable_hash(value: str) -> str:
    import hashlib

    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _stable_index(seed: str, length: int) -> int:
    if length <= 0:
        return 0
    return int(_stable_hash(seed), 16) % length


def _question_id(
    domain_id: str,
    question_type: str,
    token: str,
    *parts: str,
) -> str:
    import hashlib

    digest = hashlib.sha1(
        "|".join((domain_id, question_type, token, *parts)).encode("utf-8")
    ).hexdigest()[:20]
    return f"q-{digest}"


def _safe_node_key(value: str) -> str:
    import hashlib

    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def _default_question_types() -> tuple[str, ...]:
    return (
        "单项选择题",
        "多项选择题",
        "判断题",
        "简答题",
        "论述题",
        "文献对比分析题",
    )


def _question_dimensions(question_type: str) -> tuple[str, ...]:
    if question_type in {"单项选择题", "判断题"}:
        return ("概念记忆", "文献理解")
    if question_type == "多项选择题":
        return ("文献理解", "概念记忆")
    if question_type == "简答题":
        return ("文献理解", "学术应用")
    if question_type == "论述题":
        return ("批判性思维", "学术应用")
    if question_type == "文献对比分析题":
        return ("对比分析", "批判性思维", "学术应用")
    return ("文献理解",)


def _local_rubric_score(
    question: AssessmentQuestion,
    user_answer: str,
) -> ScoreResult:
    normalized = user_answer.lower()
    criteria: list[CriterionScore] = []
    for rubric in question.rubric:
        keywords = tuple(
            keyword.lower()
            for keyword in rubric.expected_keywords
            if keyword.strip()
        )
        matched = tuple(
            keyword for keyword in keywords if keyword in normalized
        )
        if not keywords:
            ratio = min(1.0, len(user_answer) / 120)
            reason = "当前细则没有关键词，仅按内容覆盖长度给出保守分。"
        else:
            ratio = len(matched) / len(keywords)
            reason = (
                f"命中细则关键词 {len(matched)}/{len(keywords)}："
                f"{'、'.join(matched) if matched else '未命中'}。"
            )
        criteria.append(
            CriterionScore(
                label=rubric.label,
                score=round(rubric.points * ratio, 2),
                max_score=rubric.points,
                reason=reason,
            )
        )
    total = sum(item.score for item in criteria)
    return ScoreResult(
        score=total,
        max_score=question.max_score,
        feedback=(
            "本地评分只检查评分细则中的内容要点，不评价语言流畅度。"
            "该结果用于学习反馈，不等同于教师评分。"
        ),
        criteria=tuple(criteria),
        mode="本地评分细则（非大模型）",
        user_answer=user_answer,
    )


def _empty_subjective_score(
    question: AssessmentQuestion,
    user_answer: str = "",
) -> ScoreResult:
    return ScoreResult(
        score=0.0,
        max_score=question.max_score,
        feedback="答案为空，未获得评分细则中的任何内容分。",
        criteria=tuple(
            CriterionScore(
                label=item.label,
                score=0.0,
                max_score=item.points,
                reason="未作答。",
            )
            for item in question.rubric
        ),
        mode="输入校验",
        user_answer=user_answer,
    )


def _parse_json_object(value: str) -> dict[str, Any]:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("大模型评分未返回 JSON 对象。")
    data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("大模型评分 JSON 顶层必须是对象。")
    return data


__all__ = [
    "AcademicDebateAnalyzer",
    "KnowledgeSynthesizer",
    "LearningAssessmentAgent",
    "LiteratureCollector",
    "PaperReader",
    "PersonalTutorAgent",
]
