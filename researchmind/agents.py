"""ResearchMind 的六个核心 Agent 与可追溯分析逻辑。

本地模式只从真实 PDF 证据片段提取和比较，不使用模板补写论文结论。大模型
仅在主观题评分时作为可选增强，并要求输出评分细则。
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Iterable

from researchmind.collectors import LiteratureCollector
from researchmind.evidence_quality import (
    ANALYSIS_ENGINE_VERSION,
    AUTO_GRADE_DISCLAIMER,
    RESEARCH_QUESTIONS,
    ResearchQuestionDefinition,
    classify_result_direction,
    contains_any,
    evidence_quality_flags,
    extract_sample_size,
    infer_study_design,
    preliminary_certainty,
)
from researchmind.models import (
    ASSESSMENT_MODULES,
    AssessmentQuestion,
    CriterionScore,
    DebateAnalysis,
    EvidenceRef,
    KnowledgeEdge,
    KnowledgeNode,
    PaperAnalysis,
    PaperRecord,
    QuestionSynthesis,
    ResearchConflict,
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
    "interventions": "干预措施",
    "comparators": "对照条件",
    "measures": "测量指标",
    "results": "研究结果",
    "statistics": "统计结果",
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
    "interventions": (
        "干预组",
        "实验组",
        "接受",
        "训练方案",
        "运动方案",
        "intervention group",
        "experimental group",
        "training program",
        "exercise program",
        "received",
    ),
    "comparators": (
        "对照组",
        "控制组",
        "常规护理",
        "等待名单",
        "control group",
        "comparison group",
        "usual care",
        "waitlist",
        "compared with",
    ),
    "measures": (
        "测量指标",
        "结局指标",
        "评价指标",
        "测试指标",
        "bmi",
        "体脂率",
        "最大摄氧量",
        "outcome measure",
        "measured",
        "questionnaire",
        "body mass index",
        "vo2",
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
        "improved",
        "increased",
        "decreased",
        "reduced",
        "effect size",
        "p =",
        "p<",
        "p <",
    ),
    "statistics": (
        "p =",
        "p<",
        "p <",
        "95% ci",
        "confidence interval",
        "effect size",
        "mean difference",
        "standard deviation",
        "odds ratio",
        "anova",
        "regression",
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
        "future studies",
        "further research",
        "未来研究",
        "limitation",
        "limitations",
        "future research",
    ),
}

ABSTRACT_FIELD_MARKERS = {
    "purpose": (
        "objective",
        "aim",
        "purpose",
        "本研究",
        "目的",
        "旨在",
        "探讨",
        "评估",
    ),
    "population": (
        "participants",
        "sample",
        "children",
        "adolescents",
        "students",
        "青少年",
        "儿童",
        "学生",
        "样本",
        "研究对象",
    ),
    "methods": (
        "method",
        "design",
        "systematic review",
        "meta-analysis",
        "randomized",
        "randomised",
        "questionnaire",
        "survey",
        "方法",
        "系统综述",
        "元分析",
        "随机对照",
        "问卷调查",
        "研究框架",
    ),
    "interventions": (
        "intervention",
        "exercise",
        "training",
        "physical activity",
        "体育教学",
        "运动",
        "干预",
        "训练",
    ),
    "comparators": (
        "control group",
        "comparison group",
        "usual care",
        "对照组",
        "比较组",
        "常规护理",
    ),
    "measures": (
        "outcome",
        "measure",
        "questionnaire",
        "bmi",
        "fitness",
        "指标",
        "测量",
        "量表",
        "问卷",
        "体质",
    ),
    "results": (
        "result",
        "finding",
        "we found",
        "significant",
        "improved",
        "increased",
        "decreased",
        "reduced",
        "结果",
        "发现",
        "改善",
        "提高",
        "降低",
        "有效",
        "正向影响",
    ),
    "statistics": (
        "p =",
        "p<",
        "p <",
        "95% ci",
        "effect size",
        "mean difference",
        "odds ratio",
    ),
    "conclusions": (
        "conclusion",
        "we conclude",
        "findings suggest",
        "结论",
        "表明",
        "建议",
    ),
    "limitations": (
        "limitation",
        "future research",
        "future studies",
        "局限",
        "不足",
        "未来研究",
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


class AssessmentGenerationError(RuntimeError):
    """带用户提示和修复建议的组卷异常。"""

    def __init__(
        self,
        code: str,
        message: str,
        suggestion: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.user_message = message
        self.suggestion = suggestion


MODULE_QUESTION_TYPES = {
    "基础知识理解": (
        "单项选择题",
        "多项选择题",
        "判断题",
        "简答题",
    ),
    "文献深度理解": ("深度理解题",),
    "文献对比分析": ("文献对比分析题",),
    "学术思辨训练": ("学术思辨题",),
    "个性化错题与补学": ("个性化补学题",),
}


class PaperReader:
    """Agent 2：从真实 PDF 证据中提取结构化论文信息。"""

    def __init__(self, store: ResearchMindStore) -> None:
        self.store = store

    def read_paper(
        self,
        paper_id: str,
        *,
        save: bool = True,
    ) -> PaperAnalysis:
        paper = self.store.get_paper(paper_id)
        if paper is None:
            raise ValueError(f"论文不存在：{paper_id}")
        raw_evidence = self.store.list_evidence(paper_id)
        evidence = tuple(
            item
            for item in raw_evidence
            if evidence_quality_flags(
                item.quote,
                section_hint=item.section_hint,
                source_type=item.source_type,
            )[0]
        )
        abstract_evidence = _paper_abstract_evidence(paper)
        if not evidence and abstract_evidence is None:
            analysis = PaperAnalysis(
                paper_id=paper_id,
                extraction_mode="本地规则提取",
                status="证据不足：该论文尚未导入可检索正文。",
            )
            if save:
                self.store.save_analysis(analysis)
            return analysis

        fields = {}
        for field, keywords in FIELD_KEYWORDS.items():
            selected = _select_evidence(evidence, keywords, limit=2)
            if not selected and abstract_evidence is not None:
                selected = _select_evidence(
                    (abstract_evidence,),
                    keywords,
                    limit=2,
                )
            if not selected and abstract_evidence is not None:
                selected = _select_abstract_field_evidence(
                    abstract_evidence,
                    field,
                )
            fields[field] = selected
        if not evidence and not any(
            fields[name]
            for name in ("purpose", "methods", "results", "conclusions")
        ):
            # 只有题录和宽泛主题词、没有可核验研究结构的摘要不算已解析。
            fields = {field: () for field in FIELD_KEYWORDS}
        filled = sum(bool(value) for value in fields.values())
        abstract_field_count = sum(
            any(item.source_type == "摘要" for item in values)
            for values in fields.values()
        )
        if filled == 0:
            extraction_mode = "仅题录/摘要可用，未提取到可核验结构字段"
        elif evidence and abstract_field_count:
            extraction_mode = "本地规则提取；全文优先，缺失字段由摘要补充"
        elif evidence:
            extraction_mode = "本地规则提取，保留可定位全文证据"
        else:
            extraction_mode = "仅基于摘要分析，未读取可检索全文"
        analysis = PaperAnalysis(
            paper_id=paper_id,
            purpose=fields["purpose"],
            population=fields["population"],
            methods=fields["methods"],
            interventions=fields["interventions"],
            comparators=fields["comparators"],
            measures=fields["measures"],
            results=fields["results"],
            statistics=fields["statistics"],
            conclusions=fields["conclusions"],
            limitations=fields["limitations"],
            extraction_mode=extraction_mode,
            status=(
                "证据不足：仅题录摘要未提供可核验的方法或结果。"
                if filled == 0
                else (
                    f"已定位 {filled}/{len(FIELD_KEYWORDS)} 个结构字段；"
                    + (
                        f"其中 {abstract_field_count} 个字段仅来自摘要。"
                        if abstract_field_count
                        else ""
                    )
                    + "缺失字段标记为证据不足。"
                )
            ),
        )
        if save:
            self.store.save_analysis(analysis)
        return analysis

    def read_domain(
        self,
        domain_id: str,
    ) -> tuple[PaperAnalysis, ...]:
        """读取全部活动论文；无全文时明确使用可核验摘要。"""

        analyses: list[PaperAnalysis] = []
        for paper in self.store.list_papers(domain_id):
            analysis = self.read_paper(paper.paper_id, save=False)
            has_full_text = bool(self.store.list_evidence(paper.paper_id, limit=1))
            has_abstract_evidence = any(
                item.source_type == "摘要"
                for values in (
                    analysis.purpose,
                    analysis.population,
                    analysis.methods,
                    analysis.interventions,
                    analysis.comparators,
                    analysis.measures,
                    analysis.results,
                    analysis.statistics,
                    analysis.conclusions,
                    analysis.limitations,
                )
                for item in values
            ) and not analysis.status.startswith("证据不足")
            if has_full_text or has_abstract_evidence:
                self.store.save_analysis(analysis)
                analyses.append(analysis)
        return tuple(analyses)


class AcademicDebateAnalyzer:
    """Agent 3：按具体研究问题综合真实论文证据与结果差异。"""

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
        papers = {
            paper.paper_id: paper for paper in self.store.list_papers(domain_id)
        }
        analyses = {
            paper_id: (
                self.store.get_analysis(paper_id)
                or PaperReader(self.store).read_paper(paper_id)
            )
            for paper_id in papers
        }
        evidence_by_paper = {
            paper_id: _question_evidence_pool(
                self.store.list_evidence(paper_id),
                paper,
            )
            for paper_id, paper in papers.items()
        }
        covered_paper_ids = tuple(
            paper_id
            for paper_id, items in evidence_by_paper.items()
            if items
        )
        fulltext_paper_ids = tuple(
            paper_id
            for paper_id, items in evidence_by_paper.items()
            if any(item.source_type == "全文" for item in items)
        )
        abstract_paper_ids = tuple(
            paper_id
            for paper_id, items in evidence_by_paper.items()
            if items and paper_id not in fulltext_paper_ids
        )
        metadata_only_count = len(papers) - len(covered_paper_ids)
        analysis_time = _now()
        question_syntheses: list[QuestionSynthesis] = []
        conflicts: list[ResearchConflict] = []
        for question in RESEARCH_QUESTIONS:
            synthesis, conflict = _synthesize_research_question(
                question=question,
                papers=papers,
                analyses=analyses,
                evidence_by_paper=evidence_by_paper,
                analysis_time=analysis_time,
            )
            question_syntheses.append(synthesis)
            if conflict is not None:
                conflicts.append(conflict)

        viewpoints = _viewpoints_from_question_syntheses(
            question_syntheses,
            analyses,
            papers,
        )
        coverage_note = (
            f"本次分析考虑 {len(papers)} 篇活动论文；"
            f"{len(fulltext_paper_ids)} 篇有可定位全文，"
            f"{len(abstract_paper_ids)} 篇仅使用摘要级证据，"
            f"{metadata_only_count} 篇为仅题录或暂无可检索正文。"
        )
        consensus_questions = tuple(
            item
            for item in question_syntheses
            if item.status in {"初步共识", "有一致研究发现"}
        )
        insufficient_questions = tuple(
            item for item in question_syntheses if item.status == "证据不足"
        )
        if consensus_questions:
            consensus_summary = (
                f"在 {len(question_syntheses)} 个预设研究问题中，"
                f"{len(consensus_questions)} 个形成同方向研究发现，"
                f"其中 {len(conflicts)} 个问题同时存在可核验结果差异。"
                f"所有确定性均为{AUTO_GRADE_DISCLAIMER}。"
            )
        else:
            consensus_summary = (
                "证据不足：当前论文未达到形成主题级研究发现综合的最低条件；"
                f"{len(insufficient_questions)} 个研究问题已逐项标记。"
            )
        if conflicts:
            disagreement_summary = (
                f"在 {len(conflicts)} 个具体研究问题中发现同题结果方向差异；"
                "差异描述只比较可定位结果，并把样本、设计、剂量和测量差异列为"
                "可能解释，不据此判定某一方正确。"
            )
        else:
            disagreement_summary = (
                "本次没有在同一个研究问题内发现可核验的相反结果；"
                "这不等于整个领域不存在争议。"
            )

        possible_reasons = tuple(
            dict.fromkeys(
                reason
                for conflict in conflicts
                for reason in conflict.divergence_reasons
            )
        ) or (
            "当前没有同题结果冲突，方法差异不会被拼凑成学术争议。",
        )
        limitations = _collect_limitations(analyses, papers, limit=6)
        research_gaps = tuple(
            f"{item.label}：{item.conclusion}"
            for item in insufficient_questions[:4]
        ) or (
            "需要进一步核对同一结局、同一运动剂量和同一研究设计下的结果可重复性。",
        )
        if conflicts:
            research_gaps = (
                *research_gaps,
                "优先开展同一结局定义、同一干预剂量和明确随访周期的对照研究。",
            )
        if metadata_only_count:
            research_gaps = (
                *research_gaps,
                f"有 {metadata_only_count} 篇论文没有可检索摘要或全文，"
                "未参与结果方向比较。",
            )
        certainty_counts = Counter(
            item.evidence_certainty for item in question_syntheses
        )
        graded_labels = {"高", "中等", "低", "极低"}
        has_graded = any(label in graded_labels for label in certainty_counts)
        strength_prefix = (
            "自动初筛："
            if has_graded
            else "待评估：自动确定性初筛，"
        )
        evidence_strength = (
            strength_prefix
            + "、".join(
                f"{label} {count} 个问题"
                for label, count in certainty_counts.items()
            )
            + f"。{AUTO_GRADE_DISCLAIMER}。"
        )
        debate = DebateAnalysis(
            topic=clean_topic,
            viewpoints=viewpoints,
            possible_reasons=possible_reasons,
            limitations=limitations,
            evidence_strength=evidence_strength,
            research_gaps=research_gaps,
            consensus_summary=consensus_summary,
            disagreement_summary=disagreement_summary,
            analysis_mode="本地规则证据归纳",
            considered_paper_ids=tuple(papers),
            evidence_covered_paper_ids=covered_paper_ids,
            coverage_note=coverage_note,
            question_syntheses=tuple(question_syntheses),
            conflicts=tuple(conflicts),
            analysis_time=analysis_time,
            engine_version=ANALYSIS_ENGINE_VERSION,
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
        self.last_missing_question_types: tuple[str, ...] = ()
        self.last_missing_assessment_modules: tuple[str, ...] = ()
        self.last_field_gap_counts: dict[str, int] = {}
        self.last_unparsed_titles: tuple[str, ...] = ()
        self.last_parse_errors: tuple[str, ...] = ()

    def generate_question_set(
        self,
        domain_id: str,
        *,
        question_types: Iterable[str] | None = None,
        assessment_modules: Iterable[str] | None = None,
        debate_topic: str = "体育锻炼行为与体质健康",
    ) -> tuple[AssessmentQuestion, ...]:
        """每种题型至少生成一题；证据不足的题型会跳过并保持可解释。"""

        try:
            papers = self.store.list_papers(domain_id)
        except Exception as error:
            raise AssessmentGenerationError(
                "database_read_failed",
                "无法读取当前论文库，组卷未开始。",
                "请检查 SQLite 数据库文件、磁盘权限和数据库连接后重试。",
            ) from error
        if not papers:
            raise AssessmentGenerationError(
                "empty_library",
                "当前领域没有活动论文，无法生成有来源的试题。",
                "请先从 Zotero 或文献采集页面导入论文。",
            )
        analyses = []
        unparsed_titles: list[str] = []
        parse_errors: list[str] = []
        for paper in papers:
            try:
                analysis = self.store.get_analysis(paper.paper_id)
                if analysis is None:
                    analysis = PaperReader(self.store).read_paper(paper.paper_id)
            except Exception as error:
                parse_errors.append(
                    f"{paper.title}：{type(error).__name__}: {error}"
                )
                unparsed_titles.append(paper.title)
                continue
            if not analysis.all_evidence:
                unparsed_titles.append(paper.title)
            analyses.append((paper, analysis))
        self.last_unparsed_titles = tuple(unparsed_titles)
        self.last_parse_errors = tuple(parse_errors)
        self.last_field_gap_counts = {
            field: sum(
                not analysis.field_evidence(field)
                for _, analysis in analyses
            )
            for field in (
                "methods",
                "interventions",
                "comparators",
                "measures",
                "results",
                "statistics",
                "conclusions",
            )
        }
        if len(unparsed_titles) == len(papers):
            raise AssessmentGenerationError(
                "unparsed_library",
                "当前论文库没有任何论文完成可检索解析，无法组卷。",
                "请先补充全文或摘要，并在“论文智能阅读”中确认至少一篇论文产生了原文证据。"
                + (
                    " 最近解析错误：" + "；".join(parse_errors[:2])
                    if parse_errors
                    else ""
                ),
            )
        try:
            page_evidence = self.store.list_domain_evidence(domain_id)
        except Exception as error:
            raise AssessmentGenerationError(
                "evidence_retrieval_failed",
                "论文库已读取，但原文证据检索失败。",
                "请检查数据库完整性；如果使用了 RAG/向量索引，请同时检查索引状态后重试。",
            ) from error
        evidence_pool = _assessment_evidence_pool(page_evidence, analyses)
        if not evidence_pool:
            raise AssessmentGenerationError(
                "evidence_insufficient",
                "论文库中没有可用于出题的方法、结果、结论或摘要证据。",
                "请补充全文或有效摘要，并重新执行论文结构化阅读。",
            )
        token = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        selected_modules = tuple(assessment_modules or ())
        if selected_modules:
            invalid_modules = tuple(
                module
                for module in selected_modules
                if module not in MODULE_QUESTION_TYPES
            )
            if invalid_modules:
                raise AssessmentGenerationError(
                    "invalid_assessment_module",
                    "包含不支持的测评模块："
                    + "、".join(invalid_modules),
                    "请从基础知识理解、文献深度理解、文献对比分析、"
                    "学术思辨训练和个性化错题与补学中选择。",
                )
            tasks = tuple(
                (question_type, module)
                for module in selected_modules
                for question_type in MODULE_QUESTION_TYPES[module]
            )
        else:
            requested_types = tuple(
                question_types or _default_question_types()
            )
            tasks = tuple(
                (
                    question_type,
                    _module_for_question_type(question_type),
                )
                for question_type in requested_types
            )
        if not tasks:
            raise AssessmentGenerationError(
                "no_question_types",
                "尚未选择测评模块或题型，无法生成试卷。",
                "请至少选择一个测评模块后重试。",
            )

        generators = {
            "单项选择题": self._single_choice,
            "多项选择题": self._multiple_choice,
            "判断题": self._true_false,
            "简答题": self._short_answer,
            "论述题": self._essay_question,
            "文献对比分析题": self._comparison_question,
            "深度理解题": self._deep_understanding,
            "学术思辨题": self._academic_critical_thinking,
            "个性化补学题": self._personalized_review,
        }
        wrong_attempts = self.store.list_wrong_attempts(domain_id)
        tag_counts = Counter(
            tag
            for attempt in wrong_attempts
            for tag in attempt.knowledge_tags
        )
        focus_tags = tuple(
            tag for tag, _ in tag_counts.most_common(8)
        )
        questions: list[AssessmentQuestion] = []
        generated_types: set[str] = set()
        generated_modules: set[str] = set()
        for question_type, assessment_module in tasks:
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
                assessment_module=assessment_module,
                focus_tags=focus_tags,
                wrong_attempts=wrong_attempts,
            )
            if question is not None:
                question = replace(
                    question,
                    assessment_module=assessment_module,
                )
                questions.append(question)
                generated_types.add(question_type)
                generated_modules.add(assessment_module)
        if not questions:
            raise AssessmentGenerationError(
                "generation_insufficient",
                "已找到论文证据，但不足以生成所选题型的可追溯题目。",
                "请减少题型数量、补充更多全文，或先完成论文结构化阅读。",
            )
        missing_types = tuple(
            question_type
            for question_type, _ in tasks
            if question_type in generators and question_type not in generated_types
        )
        if missing_types:
            # 保留已生成题目，同时把具体缺失题型交由页面明确提示。
            self.last_missing_question_types = missing_types
        else:
            self.last_missing_question_types = ()
        requested_modules = tuple(
            dict.fromkeys(module for _, module in tasks)
        )
        self.last_missing_assessment_modules = tuple(
            module
            for module in requested_modules
            if module not in generated_modules
        )
        try:
            self.store.save_questions(tuple(questions))
        except Exception as error:
            raise AssessmentGenerationError(
                "database_write_failed",
                "试题已经生成，但保存到本地数据库失败。",
                "请检查数据库磁盘空间和写入权限；当前试卷不会被视为已持久化。",
            ) from error
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
            source_paper_ids=tuple(
                dict.fromkeys(
                    item.document_id for item in source_evidence
                )
            ),
            evidence=source_evidence,
            created_at=_now(),
            explanation=(
                f"正确选项来自《{paper.title}》{correct_evidence.location_label}的"
                f"{FIELD_LABELS[target_field]}证据：{_clip(correct_evidence.quote, 260)}"
                " 其他选项来自其他论文或不同证据片段，因此不能作为本题指定"
                "论文对应字段的答案。"
            ),
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
            source_paper_ids=tuple(
                dict.fromkeys(
                    item.document_id for item in (*correct_evidence, *distractors[:2])
                )
            ),
            evidence=(*correct_evidence, *distractors[:2]),
            created_at=_now(),
            explanation=(
                "正确选项依据同一论文的可定位结果或结论证据："
                + "；".join(
                    f"{_clip(item.quote, 180)}（{item.location_label}）"
                    for item in correct_evidence
                )
                + "。其他选项来自不同论文，不能计入本题正确答案。"
            ),
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
                f"论文《{cited_title}》{evidence.location_label}。"
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
            source_paper_ids=tuple(
                dict.fromkeys(
                    (
                        paper.paper_id,
                        *(
                            (other_paper.paper_id,)
                            if make_false and other_paper is not None
                            else ()
                        ),
                    )
                )
            ),
            evidence=(evidence,),
            created_at=_now(),
            explanation=(
                f"该片段实际来自《{paper.title}》{evidence.location_label}。"
                + (
                    "题干预先给出的论文题名与原文来源不一致，因此判断为错误。"
                    if make_false
                    else "题干预先给出的论文题名与原文来源一致，因此判断为正确。"
                )
            ),
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
            rubric=_five_dimension_subjective_rubric(
                evidence,
                comparison_required=False,
            ),
            knowledge_tags=(FIELD_LABELS[field], "文献理解"),
            difficulty="进阶",
            source_paper_ids=(paper.paper_id,),
            evidence=evidence,
            created_at=_now(),
            scoring_mode="大模型评分优先，失败时按本地关键词细则",
            explanation=(
                f"参考答案直接摘录《{paper.title}》的"
                f"{FIELD_LABELS[field]}证据：{_clip(evidence[0].quote, 420)}"
                f"（{evidence[0].location_label}）。未在原文出现的推断不计入核心答案。"
            ),
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
            rubric=_five_dimension_subjective_rubric(
                evidence,
                comparison_required=True,
            ),
            knowledge_tags=("文献对比", "批判性思维", "学术共识边界"),
            difficulty="挑战",
            source_paper_ids=(first[0].paper_id, second[0].paper_id),
            evidence=evidence,
            created_at=_now(),
            scoring_mode="大模型按细则评分优先，失败时本地细则评分",
            explanation=(
                "本题要求比较两篇真实论文。评分时可依据下列原文逐项核对："
                + "；".join(
                    f"《{title}》{evidence_item.location_label}："
                    f"{_clip(evidence_item.quote, 160)}"
                    for title, evidence_item in (
                        (first[0].title, first[1].methods[0]),
                        (second[0].title, second[1].methods[0]),
                    )
                )
                + "。结论必须区分两篇论文各自的证据边界。"
            ),
        )

    def _comparison_question(
        self,
        **context: Any,
    ) -> AssessmentQuestion | None:
        candidates = [
            (paper, analysis)
            for paper, analysis in context["analyses"]
            if analysis.methods
            and (analysis.results or analysis.conclusions)
        ]
        if len(candidates) < 2:
            return None
        candidates.sort(
            key=lambda item: (
                -len(item[1].methods),
                -len(item[1].results),
                -len(item[1].conclusions),
                item[0].title,
            )
        )
        first, second = candidates[:2]
        first_anchor = (
            (first[1].results or first[1].conclusions or first[1].methods)
            [:1]
        )
        second_anchor = (
            (second[1].results or second[1].conclusions or second[1].methods)
            [:1]
        )
        selected = _unique_question_evidence(
            (
                *first[1].population[:1],
                *first[1].methods[:1],
                *first[1].interventions[:1],
                *first[1].measures[:1],
                *first[1].results[:1],
                *first[1].conclusions[:1],
                *first[1].limitations[:1],
                *second[1].population[:1],
                *second[1].methods[:1],
                *second[1].interventions[:1],
                *second[1].measures[:1],
                *second[1].results[:1],
                *second[1].conclusions[:1],
                *second[1].limitations[:1],
            )
        )
        if not first_anchor or not second_anchor or len(selected) < 4:
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
                f"围绕“{context['debate_topic']}”，比较《{first[0].title}》与"
                f"《{second[0].title}》的研究对象、方法、样本、干预措施、"
                "测量指标、结果方向、证据质量和结论边界。请说明两篇论文"
                "是否形成共识或存在分歧，并分析可能原因。"
            ),
            options=(),
            standard_answer=_build_comparison_standard_answer(
                (
                    (first[0].title, first[1], first_anchor[0]),
                    (second[0].title, second[1], second_anchor[0]),
                )
            ),
            rubric=_five_dimension_subjective_rubric(
                selected,
                comparison_required=True,
            ),
            knowledge_tags=("对比分析", "学术争议", "证据强度"),
            difficulty="挑战",
            source_paper_ids=tuple(
                dict.fromkeys(item.document_id for item in selected)
            ),
            evidence=selected,
            created_at=_now(),
            scoring_mode="大模型按细则评分优先，失败时本地细则评分",
            explanation=(
                "本题只比较以下可定位片段："
                + "；".join(
                    f"《{item.document_title}》{item.location_label}："
                    f"{_clip(item.quote, 180)}"
                    for item in selected
                )
                + "。只有这些片段方向一致时才能讨论初步共识；片段不足或"
                "方向不同时必须说明证据边界。"
            ),
        )

    def _deep_understanding(
        self,
        **context: Any,
    ) -> AssessmentQuestion | None:
        candidates = [
            (paper, analysis)
            for paper, analysis in context["analyses"]
            if analysis.methods
            and (
                analysis.population
                or analysis.results
                or analysis.conclusions
            )
        ]
        if not candidates:
            return None
        paper, analysis = candidates[
            _stable_index(context["token"] + "deep", len(candidates))
        ]
        evidence = _unique_question_evidence(
            (
                *analysis.population[:1],
                *analysis.methods[:1],
                *analysis.results[:1],
                *analysis.conclusions[:1],
                *analysis.limitations[:1],
            )
        )
        if len(evidence) < 2:
            return None
        return AssessmentQuestion(
            question_id=_question_id(
                context["domain_id"],
                "deep",
                context["token"],
                *(item.chunk_id for item in evidence),
            ),
            domain_id=context["domain_id"],
            question_type="深度理解题",
            prompt=(
                f"阅读《{paper.title}》。作者为什么采用该研究方法？"
                "研究对象和样本是否支持外推？研究结果能够支持哪些结论，"
                "又有哪些表述超出了原始证据？请逐项回答并引用原文。"
            ),
            options=(),
            standard_answer=_build_deep_standard_answer(
                paper,
                analysis,
            ),
            rubric=_five_dimension_subjective_rubric(
                evidence,
                comparison_required=False,
            ),
            knowledge_tags=(
                "研究设计合理性",
                "样本代表性",
                "证据边界",
                "文献理解",
            ),
            difficulty="进阶",
            source_paper_ids=(paper.paper_id,),
            evidence=evidence,
            created_at=_now(),
            scoring_mode="大模型按证据与五维细则评分优先，失败时本地证据规则",
            explanation=(
                f"本题依据《{paper.title}》的研究对象、方法、结果和结论原文。"
                "答案解析只复述可定位证据；原文未提供的设计动机、样本代表性"
                "和长期效果均标记为“原文未说明”。也可核对限制段落中的作者局限。"
            ),
        )

    def _academic_critical_thinking(
        self,
        **context: Any,
    ) -> AssessmentQuestion | None:
        candidates = [
            (paper, analysis)
            for paper, analysis in context["analyses"]
            if analysis.methods
            and (analysis.results or analysis.conclusions)
        ]
        if len(candidates) < 2:
            return None
        first, second = candidates[:2]
        evidence = _unique_question_evidence(
            (
                *first[1].methods[:1],
                *first[1].results[:1],
                *first[1].conclusions[:1],
                *first[1].limitations[:1],
                *second[1].methods[:1],
                *second[1].results[:1],
                *second[1].conclusions[:1],
                *second[1].limitations[:1],
            )
        )
        if len(evidence) < 4:
            return None
        return AssessmentQuestion(
            question_id=_question_id(
                context["domain_id"],
                "critical",
                context["token"],
                *(item.chunk_id for item in evidence),
            ),
            domain_id=context["domain_id"],
            question_type="学术思辨题",
            prompt=(
                f"比较《{first[0].title}》与《{second[0].title}》。"
                "为什么两篇论文可能得出不同结论？哪篇证据在当前原文范围内"
                "更有说服力？请分析样本偏差、混杂因素、研究设计、测量与统计"
                "局限，并说明如果由你继续研究应如何改进。"
            ),
            options=(),
            standard_answer=_build_critical_standard_answer(first, second),
            rubric=_five_dimension_subjective_rubric(
                evidence,
                comparison_required=True,
            ),
            knowledge_tags=(
                "学术思辨",
                "偏倚风险",
                "混杂因素",
                "研究改进",
            ),
            difficulty="挑战",
            source_paper_ids=(first[0].paper_id, second[0].paper_id),
            evidence=evidence,
            created_at=_now(),
            scoring_mode="大模型按证据与五维细则评分优先，失败时本地证据规则",
            explanation=(
                "本题不预设哪篇论文必然正确。评分应依据两篇论文可定位的方法、"
                "样本、结果、结论和局限证据；若原文未报告偏倚风险或混杂因素，"
                "只能指出证据边界，不能补写未报告的数据。"
            ),
        )

    def _personalized_review(
        self,
        **context: Any,
    ) -> AssessmentQuestion | None:
        focus_tags = tuple(context.get("focus_tags", ()))
        if not focus_tags:
            mastery = self.store.ensure_mastery(context["domain_id"])
            weakest = min(mastery, key=lambda item: item.score)
            focus_tags = (weakest.dimension,)
        candidates = [
            (paper, analysis)
            for paper, analysis in context["analyses"]
            if analysis.results or analysis.conclusions
        ]
        if not candidates:
            return None
        focus_text = "、".join(focus_tags)

        def score_candidate(item):
            paper, analysis = item
            text = " ".join(
                (
                    paper.title,
                    " ".join(paper.keywords),
                    *(evidence.quote for evidence in analysis.all_evidence),
                )
            ).casefold()
            return sum(tag.casefold() in text for tag in focus_tags)

        candidates.sort(key=score_candidate, reverse=True)
        paper, analysis = candidates[0]
        evidence = _unique_question_evidence(
            (
                *analysis.results[:1],
                *analysis.conclusions[:1],
                *analysis.methods[:1],
            )
        )
        if not evidence:
            return None
        return AssessmentQuestion(
            question_id=_question_id(
                context["domain_id"],
                "personalized",
                context["token"],
                *(item.chunk_id for item in evidence),
            ),
            domain_id=context["domain_id"],
            question_type="个性化补学题",
            prompt=(
                f"你的近期薄弱点为“{focus_text}”。请依据《{paper.title}》"
                "原文，概括该论文与研究薄弱点最相关的方法、结果和结论，"
                "并指出一条不能由现有原文支持的推断。"
            ),
            options=(),
            standard_answer=(
                f"薄弱点：{focus_text}。"
                f"可核验原文：{_clip(evidence[0].quote, 520)}"
                f"（{evidence[0].location_label}）。"
                "超出原文样本、设计或测量范围的推断不能作为事实。"
            ),
            rubric=_five_dimension_subjective_rubric(
                evidence,
                comparison_required=False,
            ),
            knowledge_tags=("个性化补学", *focus_tags),
            difficulty="进阶",
            source_paper_ids=(paper.paper_id,),
            evidence=evidence,
            created_at=_now(),
            scoring_mode="大模型按证据与五维细则评分优先，失败时本地证据规则",
            explanation=(
                f"个性化补学依据近期错题标签“{focus_text}”和当前论文证据生成。"
                "若尚无错题，则使用当前掌握度最低的能力维度作为基线诊断。"
            ),
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
            except (
                ValueError,
                KeyError,
                TypeError,
                TimeoutError,
                RuntimeError,
                json.JSONDecodeError,
            ) as error:
                fallback = _local_rubric_score(question, clean_answer)
                return ScoreResult(
                    score=fallback.score,
                    max_score=fallback.max_score,
                    feedback=(
                        f"{fallback.feedback} "
                        f"模型评分未完成，已降级为本地评分细则。"
                        f"原因：{type(error).__name__}: {error}"
                    ),
                    criteria=fallback.criteria,
                    mode="本地关键词细则（模型调用失败，已明确降级）",
                    correct=fallback.correct,
                    user_answer=fallback.user_answer,
                )
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
            "请区分原文明确事实、基于原文的合理推断和缺少证据支持的学生观点；"
            "不得把推断或常识写成原文事实，也不得接受题目证据之外的页码、DOI"
            "或研究结论。若学生只堆砌关键词、复述题目或增加篇幅，但没有形成"
            "证据与结论之间的推理，不应在逻辑或批判性维度给高分。"
            "只输出 JSON，不要 Markdown。格式："
            '{"criteria":[{"label":"细则名称","score":0,"reason":"依据"}],'
            '"evidence_used":["实际引用的原文证据"],'
            '"unsupported_claims":["缺少原文支持的主张"],'
            '"fact_inference_boundary":"原文事实、合理推断与无证据观点的区分说明",'
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
        evidence_used = payload.get("evidence_used", [])
        unsupported_claims = payload.get("unsupported_claims", [])
        if not isinstance(evidence_used, list):
            raise ValueError("大模型评分的 evidence_used 必须是数组。")
        if not isinstance(unsupported_claims, list):
            raise ValueError("大模型评分的 unsupported_claims 必须是数组。")
        boundary = str(payload.get("fact_inference_boundary", "")).strip()
        feedback_parts = [
            str(payload.get("feedback", "已按评分细则完成批改。")).strip()
        ]
        if evidence_used:
            feedback_parts.append(
                "引用证据：" + "；".join(
                    str(item).strip()[:300]
                    for item in evidence_used[:5]
                    if str(item).strip()
                )
            )
        if unsupported_claims:
            feedback_parts.append(
                "缺少原文支持的主张：" + "；".join(
                    str(item).strip()[:300]
                    for item in unsupported_claims[:5]
                    if str(item).strip()
                )
            )
        if boundary:
            feedback_parts.append("事实与推断边界：" + boundary[:600])
        return ScoreResult(
            score=total,
            max_score=question.max_score,
            feedback="\n\n".join(part for part in feedback_parts if part),
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
        """按已记录题目计算练习表现；未作答维度不设置能力初值。"""

        totals = {dimension: 0.0 for dimension in BASELINE_MASTERY}
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
                totals[dimension] += percentage
                counts[dimension] += 1
        scores = {
            dimension: (
                round(totals[dimension] / counts[dimension], 2)
                if counts[dimension]
                else 0.0
            )
            for dimension in BASELINE_MASTERY
        }
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
            item.dimension
            for item in mastery
            if item.attempts_count > 0 and item.score < 60.0
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
            if any(item.attempts_count > 0 for item in mastery):
                summary = (
                    "当前已有记录的练习表现均不低于 60 分，"
                    "但该数值尚未经过能力量表校准，不宣称已具备稳定科研能力。"
                )
            else:
                summary = (
                    "当前五个维度均为“未测”，不能推断薄弱项或能力等级。"
                    "请先完成一轮有来源依据的测评。"
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
        is_research, _, _ = evidence_quality_flags(
            item.quote,
            section_hint=item.section_hint,
            source_type=item.source_type,
        )
        if not is_research:
            continue
        if _looks_like_front_matter(item):
            continue
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


def _paper_abstract_evidence(paper: PaperRecord) -> EvidenceRef | None:
    """把真实题录摘要转成显式来源类型为“摘要”的证据。"""

    abstract = re.sub(r"\s+", " ", paper.abstract or "").strip()
    if len(abstract) < 40:
        return None
    is_research, section, _ = evidence_quality_flags(
        abstract,
        source_type="摘要",
    )
    if not is_research:
        return None
    return EvidenceRef(
        chunk_id=f"abstract:{paper.paper_id}",
        document_id=paper.paper_id,
        document_title=paper.title,
        file_name="摘要（论文元数据）",
        page_number=1,
        quote=abstract,
        source_type="摘要",
        section_hint=section,
    )


def _select_abstract_field_evidence(
    abstract: EvidenceRef,
    field: str,
) -> tuple[EvidenceRef, ...]:
    """摘要缺少标准关键词时，用明确的摘要结构词做保守字段映射。"""

    lowered = abstract.quote.casefold()
    markers = ABSTRACT_FIELD_MARKERS.get(field, ())
    if any(marker.casefold() in lowered for marker in markers):
        return (abstract,)
    return ()


def _question_evidence_pool(
    evidence: tuple[EvidenceRef, ...],
    paper: PaperRecord,
) -> tuple[EvidenceRef, ...]:
    """只保留研究文本，并附带明确标注为摘要的真实题录摘要。"""

    output: list[EvidenceRef] = []
    seen: set[str] = set()
    for item in evidence:
        is_research, _, _ = evidence_quality_flags(
            item.quote,
            section_hint=item.section_hint,
            source_type=item.source_type,
        )
        if not is_research or item.chunk_id in seen:
            continue
        seen.add(item.chunk_id)
        output.append(item)
    abstract = _paper_abstract_evidence(paper)
    if abstract is not None and abstract.chunk_id not in seen:
        output.append(abstract)
    return tuple(output)


def _synthesize_research_question(
    *,
    question: ResearchQuestionDefinition,
    papers: dict[str, PaperRecord],
    analyses: dict[str, PaperAnalysis | None],
    evidence_by_paper: dict[str, tuple[EvidenceRef, ...]],
    analysis_time: str,
) -> tuple[QuestionSynthesis, ResearchConflict | None]:
    """在一个具体研究问题内做论文级结果分组和保守综合。"""

    direction_by_paper: dict[
        str,
        tuple[str, EvidenceRef, tuple[str, ...]],
    ] = {}
    relevant_paper_ids: list[str] = []
    measure_terms: set[str] = set()

    for paper_id, paper in papers.items():
        analysis = analyses.get(paper_id)
        evidence = evidence_by_paper.get(paper_id, ())
        metadata_text = " ".join(
            (
                paper.title,
                paper.abstract,
                " ".join(paper.keywords),
            )
        )
        analysis_text = " ".join(
            item.quote
            for item in (
                analysis.all_evidence if analysis is not None else ()
            )
        )
        evidence_text = " ".join(item.quote for item in evidence)
        combined = f"{metadata_text} {analysis_text} {evidence_text}"
        if not contains_any(combined, question.exercise_terms):
            continue
        if not contains_any(combined, question.focus_terms):
            continue
        relevant_paper_ids.append(paper_id)
        measure_terms.update(
            term for term in question.focus_terms if term.casefold() in combined.casefold()
        )

        candidates = _question_direction_candidates(
            question,
            analysis,
            evidence,
        )
        if not candidates:
            continue
        direction, selected, effect_terms = candidates[0]
        direction_by_paper[paper_id] = (
            direction,
            selected,
            effect_terms,
        )

    positive_ids = tuple(
        paper_id
        for paper_id, (direction, _, _) in direction_by_paper.items()
        if direction == "正向结果"
    )
    negative_ids = tuple(
        paper_id
        for paper_id, (direction, _, _) in direction_by_paper.items()
        if direction == "负向结果"
    )
    null_ids = tuple(
        paper_id
        for paper_id, (direction, _, _) in direction_by_paper.items()
        if direction == "无显著关系"
    )
    mixed_ids = tuple(
        paper_id
        for paper_id, (direction, _, _) in direction_by_paper.items()
        if direction == "混合结果"
    )
    directional_ids = (
        set(positive_ids)
        | set(negative_ids)
        | set(null_ids)
        | set(mixed_ids)
    )
    inconclusive_ids = tuple(
        paper_id
        for paper_id in relevant_paper_ids
        if paper_id not in set(positive_ids) | set(negative_ids)
    )
    # “未达统计显著”表示当前样本不足以支持差异结论，不能解释为方向相反。
    # 因而自动争议候选只由正向结果与明确负向结果同时出现构成。
    has_conflict = bool(positive_ids and negative_ids)

    designs = tuple(
        _study_design_for_question(
            paper_id,
            papers,
            analyses,
            evidence_by_paper,
        )
        for paper_id in relevant_paper_ids
    )
    sample_sizes = tuple(
        _sample_size_for_question(
            paper_id,
            papers,
            analyses,
            evidence_by_paper,
        )
        for paper_id in relevant_paper_ids
    )
    full_text_count = sum(
        any(
            item.source_type == "全文"
            for item in evidence_by_paper.get(paper_id, ())
        )
        for paper_id in relevant_paper_ids
    )
    abstract_only_count = sum(
        bool(evidence_by_paper.get(paper_id))
        and not any(
            item.source_type == "全文"
            for item in evidence_by_paper.get(paper_id, ())
        )
        for paper_id in relevant_paper_ids
    )

    if has_conflict:
        status = "存在争议（待审核候选）"
    elif len(positive_ids) >= 2:
        status = "有一致研究发现"
    elif len(negative_ids) >= 2:
        status = "有一致研究发现"
    elif len(null_ids) >= 2:
        status = "一致未观察到显著差异"
    else:
        status = "证据不足"

    certainty, certainty_reason = preliminary_certainty(
        status=status,
        designs=designs,
        sample_sizes=sample_sizes,
        full_text_paper_count=full_text_count,
        abstract_only_paper_count=abstract_only_count,
    )
    if (
        status == "有一致研究发现"
        and positive_ids
        and certainty in {"高", "中等"}
    ):
        status = "初步共识"

    representative_ids = _representative_direction_ids(
        positive_ids,
        negative_ids,
        null_ids,
        mixed_ids,
        inconclusive_ids,
        direction_by_paper,
        limit=6,
    )
    representative_items: list[EvidenceRef] = []
    for paper_id in representative_ids:
        item = _best_representative_evidence(
            paper_id,
            question,
            analyses.get(paper_id),
            direction_by_paper,
            evidence_by_paper,
        )
        if item is not None:
            representative_items.append(item)
    representative_evidence = tuple(representative_items)
    if has_conflict:
        conclusion = (
            f"{len(positive_ids)} 篇论文报告正向改善，"
            f"{len(negative_ids)} 篇报告明确负向结果；"
            f"另有 {len(null_ids)} 篇报告未达统计显著，"
            "未计入反方。该方向差异仅为待审核候选，"
            "需核对结局定义、干预条件、对照、测量时点和研究可比性。"
        )
    elif status in {"初步共识", "有一致研究发现"} and positive_ids:
        conclusion = (
            f"{len(positive_ids)} 篇相关论文报告与问题假设一致的结果；"
            f"另有 {len(inconclusive_ids)} 篇结果方向无法保守判定。"
        )
    elif status == "一致未观察到显著差异":
        conclusion = (
            f"{len(null_ids)} 篇相关论文报告未观察到统计显著差异；"
            "该表述不代表效应一定为零，也不能改写为负向结论。"
        )
    elif status in {"初步共识", "有一致研究发现"}:
        conclusion = (
            f"{len(negative_ids)} 篇相关论文报告明确负向结果；"
            "应表述为“未观察到一致改善”，不能改写成正向共识。"
        )
    elif relevant_paper_ids:
        conclusion = (
            f"共定位到 {len(relevant_paper_ids)} 篇相关论文，但可核验方向结果"
            f"仅为 {len(directional_ids)} 篇，尚不足以形成稳定综合。"
        )
        if null_ids:
            conclusion += (
                f"其中 {len(null_ids)} 篇报告未达统计显著，"
                "该结果未计入反方。"
            )
    else:
        conclusion = "当前论文库没有同时匹配该运动暴露和结局指标的论文。"

    limitations: list[str] = []
    if abstract_only_count:
        limitations.append(
            f"{abstract_only_count} 篇仅能基于摘要分析，未读取全文。"
        )
    if not full_text_count:
        limitations.append("该研究问题当前没有可用的全文证据。")
    if has_conflict:
        limitations.append(
            "该争议为自动待审核候选；未达统计显著的论文不属于反方，"
            "且结局定义、剂量、周期、对照与测量时点尚未完成可比性复核。"
        )
    if null_ids:
        limitations.append(
            f"{len(null_ids)} 篇论文报告未达统计显著；"
            "未达显著不等于效应方向相反。"
        )
    limitations.append(f"{AUTO_GRADE_DISCLAIMER}。")

    synthesis = QuestionSynthesis(
        question_id=question.question_id,
        label=question.label,
        question=question.question,
        conclusion=conclusion,
        status=status,
        evidence_certainty=certainty,
        certainty_reason=certainty_reason,
        supporting_paper_ids=positive_ids,
        opposing_paper_ids=negative_ids,
        inconclusive_paper_ids=inconclusive_ids,
        representative_evidence=representative_evidence,
        outcome_measures=tuple(sorted(measure_terms))[:8],
        limitations=tuple(limitations),
        analysis_time=analysis_time,
        negative_paper_ids=negative_ids,
        null_paper_ids=null_ids,
        mixed_paper_ids=mixed_ids,
        review_status=(
            "待体育科研教师复核"
            if has_conflict
            else "未形成自动争议候选"
        ),
    )
    if not has_conflict:
        return synthesis, None

    conflict = _build_question_conflict(
        question=question,
        synthesis=synthesis,
        papers=papers,
        analyses=analyses,
        direction_by_paper=direction_by_paper,
        positive_ids=positive_ids,
        negative_ids=negative_ids,
        evidence_by_paper=evidence_by_paper,
        analysis_time=analysis_time,
    )
    return synthesis, conflict


def _question_direction_candidates(
    question: ResearchQuestionDefinition,
    analysis: PaperAnalysis | None,
    evidence: tuple[EvidenceRef, ...],
) -> tuple[tuple[str, EvidenceRef, tuple[str, ...]], ...]:
    """从结果、结论和摘要中选取每篇论文最可信的问题相关结果。"""

    candidates: dict[str, EvidenceRef] = {}
    limitation_ids: set[str] = set()
    if analysis is not None:
        for item in (*analysis.results, *analysis.conclusions):
            candidates.setdefault(item.chunk_id, item)
        limitation_ids = {
            item.chunk_id for item in analysis.limitations
        }
    for item in evidence:
        if item.source_type == "摘要":
            candidates.setdefault(item.chunk_id, item)
    scored: list[tuple[int, str, EvidenceRef, tuple[str, ...]]] = []
    for item in candidates.values():
        is_research, section, _ = evidence_quality_flags(
            item.quote,
            section_hint=item.section_hint,
            source_type=item.source_type,
        )
        if not is_research:
            continue
        if item.chunk_id in limitation_ids:
            continue
        if _looks_like_limitation_text(item):
            continue
        if not contains_any(item.quote, question.focus_terms):
            continue
        if _looks_like_front_matter(item):
            continue
        direction, terms = classify_result_direction(item.quote, question)
        if direction == "待评估":
            continue
        score = 0
        score += 5 if direction != "混合结果" else 1
        score += 3 if item.source_type == "全文" else 1
        score += 3 if section == "结果" else 2 if section == "结论" else 0
        score += min(4, len(terms))
        scored.append((score, item.chunk_id, item, terms))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(
        (classify_result_direction(item.quote, question)[0], item, terms)
        for _, _, item, terms in scored[:2]
    )


def _looks_like_front_matter(item: EvidenceRef) -> bool:
    """拒绝首页题名、作者单位与摘要混排片段，避免标题词被当结果。"""

    if item.source_type != "全文" or item.page_number != 1:
        return False
    lowered = item.quote.casefold()
    markers = (
        "abstract",
        "keywords",
        "doi",
        "open access",
        "university",
        "college",
        "department",
        "corresponding author",
        "research article",
        "systematic review",
    )
    return sum(marker in lowered for marker in markers) >= 3


def _looks_like_limitation_text(item: EvidenceRef) -> bool:
    """局限或未来研究段落不能作为论文结果方向证据。"""

    if item.source_type != "全文":
        return False
    lowered = item.quote.casefold()
    return any(
        marker in lowered
        for marker in (
            "limitation",
            "future studies",
            "future research",
            "further research",
            "研究局限",
            "未来研究",
        )
    )


def _representative_direction_ids(
    positive_ids: tuple[str, ...],
    negative_ids: tuple[str, ...],
    null_ids: tuple[str, ...],
    mixed_ids: tuple[str, ...],
    inconclusive_ids: tuple[str, ...],
    direction_by_paper: dict[str, tuple[str, EvidenceRef, tuple[str, ...]]],
    *,
    limit: int,
) -> tuple[str, ...]:
    ordered = tuple(
        dict.fromkeys(
            (
                *positive_ids[:3],
                *negative_ids[:3],
                *null_ids[:3],
                *mixed_ids[:2],
                *inconclusive_ids[:2],
            )
        )
    )
    return ordered[:limit]


def _best_representative_evidence(
    paper_id: str,
    question: ResearchQuestionDefinition,
    analysis: PaperAnalysis | None,
    direction_by_paper: dict[str, tuple[str, EvidenceRef, tuple[str, ...]]],
    evidence_by_paper: dict[str, tuple[EvidenceRef, ...]],
) -> EvidenceRef | None:
    selected = direction_by_paper.get(paper_id)
    if selected is not None:
        return selected[1]
    limitation_ids = {
        item.chunk_id for item in (analysis.limitations if analysis else ())
    }
    structured = (
        (*analysis.results, *analysis.conclusions)
        if analysis is not None
        else ()
    )
    candidates = tuple(
        {
            item.chunk_id: item
            for item in (*structured, *evidence_by_paper.get(paper_id, ()))
            if item.source_type == "摘要" or item in structured
        }.values()
    )
    for item in candidates:
        if item.chunk_id in limitation_ids:
            continue
        if _looks_like_limitation_text(item):
            continue
        if not contains_any(item.quote, question.focus_terms):
            continue
        if _looks_like_front_matter(item):
            continue
        return item
    return None


def _build_question_conflict(
    *,
    question: ResearchQuestionDefinition,
    synthesis: QuestionSynthesis,
    papers: dict[str, PaperRecord],
    analyses: dict[str, PaperAnalysis | None],
    direction_by_paper: dict[str, tuple[str, EvidenceRef, tuple[str, ...]]],
    positive_ids: tuple[str, ...],
    negative_ids: tuple[str, ...],
    evidence_by_paper: dict[str, tuple[EvidenceRef, ...]],
    analysis_time: str,
) -> ResearchConflict:
    opposing_ids = negative_ids
    evidence_a = tuple(
        direction_by_paper[paper_id][1]
        for paper_id in positive_ids
        if paper_id in direction_by_paper
    )
    evidence_b = tuple(
        direction_by_paper[paper_id][1]
        for paper_id in opposing_ids
        if paper_id in direction_by_paper
    )
    reasons = _question_divergence_reasons(
        positive_ids,
        opposing_ids,
        papers,
        analyses,
        evidence_by_paper,
    )
    return ResearchConflict(
        conflict_id=f"conflict:{question.question_id.split(':', 1)[-1]}",
        question_id=question.question_id,
        question=question.question,
        viewpoint_a=(
            f"有 {len(positive_ids)} 篇论文报告运动暴露与改善方向一致。"
        ),
        paper_ids_a=positive_ids,
        evidence_a=evidence_a,
        viewpoint_b=(
            f"有 {len(negative_ids)} 篇论文报告明确负向结果。"
        ),
        paper_ids_b=opposing_ids,
        evidence_b=evidence_b,
        divergence_reasons=reasons,
        current_conclusion=(
            "现有证据在同一研究问题内出现正向与明确负向结果的待审核候选；"
            "未达统计显著的论文不计入反方，且不能把不同干预方式、"
            "结局定义或测量时点的结果直接合并。目前不足以判定哪一方正确。"
        ),
        unresolved_questions=(
            "统一运动剂量、干预周期、对照条件和结局测量后，结果是否仍然不同？",
            "差异是否由年龄、性别、基线体质或样本代表性造成？",
            "效应量和置信区间是否支持具有实践意义的改善？",
        ),
        evidence_certainty=synthesis.evidence_certainty,
        analysis_time=analysis_time,
        review_status="待体育科研教师复核",
        comparability_status="未自动判定，需核对结局、剂量、对照和测量时点",
        significance_note=(
            "候选仅比较正向与明确负向结果；未达统计显著不视为方向相反"
        ),
    )


def _question_divergence_reasons(
    positive_ids: tuple[str, ...],
    opposing_ids: tuple[str, ...],
    papers: dict[str, PaperRecord],
    analyses: dict[str, PaperAnalysis | None],
    evidence_by_paper: dict[str, tuple[EvidenceRef, ...]],
) -> tuple[str, ...]:
    left_designs = {
        _study_design_for_question(
            paper_id,
            papers,
            analyses,
            evidence_by_paper,
        )
        for paper_id in positive_ids
    }
    right_designs = {
        _study_design_for_question(
            paper_id,
            papers,
            analyses,
            evidence_by_paper,
        )
        for paper_id in opposing_ids
    }
    reasons: list[str] = []
    if left_designs != right_designs:
        reasons.append(
            "两组研究设计不同："
            + "、".join(sorted(left_designs))
            + " 对 "
            + "、".join(sorted(right_designs))
            + "。"
        )
    else:
        reasons.append(
            "研究设计类别相近，差异更可能来自干预剂量、对照条件、结局定义或测量时点。"
        )
    left_sizes: list[int] = []
    for paper_id in positive_ids:
        size = _sample_size_for_question(
            paper_id,
            papers,
            analyses,
            evidence_by_paper,
        )
        if size is not None:
            left_sizes.append(size)
    right_sizes: list[int] = []
    for paper_id in opposing_ids:
        size = _sample_size_for_question(
            paper_id,
            papers,
            analyses,
            evidence_by_paper,
        )
        if size is not None:
            right_sizes.append(size)
    if left_sizes or right_sizes:
        reasons.append(
            "可提取样本量不同：A 组 "
            + ("、".join(str(item) for item in left_sizes) or "证据不足")
            + "；B 组 "
            + ("、".join(str(item) for item in right_sizes) or "证据不足")
            + "。"
        )
    else:
        reasons.append("两组均未从可定位原文提取到明确样本量。")
    left_abstract = sum(
        not any(item.source_type == "全文" for item in evidence_by_paper.get(paper_id, ()))
        for paper_id in positive_ids
    )
    right_abstract = sum(
        not any(item.source_type == "全文" for item in evidence_by_paper.get(paper_id, ()))
        for paper_id in opposing_ids
    )
    if left_abstract or right_abstract:
        reasons.append(
            f"证据来源不完全对等：A 组 {left_abstract} 篇、"
            f"B 组 {right_abstract} 篇仅使用摘要。"
        )
    reasons.append(
        "上述差异是可能解释，不是已经证实的因果机制；需要回到原文核验。"
    )
    return tuple(reasons)


def _study_design_for_question(
    paper_id: str,
    papers: dict[str, PaperRecord],
    analyses: dict[str, PaperAnalysis | None],
    evidence_by_paper: dict[str, tuple[EvidenceRef, ...]],
) -> str:
    paper = papers[paper_id]
    analysis = analyses.get(paper_id)
    texts = [
        item.quote
        for item in (
            analysis.all_evidence if analysis is not None else ()
        )
    ]
    texts.extend(item.quote for item in evidence_by_paper.get(paper_id, ())[:3])
    return infer_study_design(paper.title, *texts)


def _sample_size_for_question(
    paper_id: str,
    papers: dict[str, PaperRecord],
    analyses: dict[str, PaperAnalysis | None],
    evidence_by_paper: dict[str, tuple[EvidenceRef, ...]],
) -> int | None:
    paper = papers[paper_id]
    analysis = analyses.get(paper_id)
    texts = [paper.abstract]
    if analysis is not None:
        texts.extend(
            item.quote
            for item in (*analysis.population, *analysis.methods)
        )
    texts.extend(item.quote for item in evidence_by_paper.get(paper_id, ())[:3])
    return extract_sample_size(*texts)


def _viewpoints_from_question_syntheses(
    syntheses: tuple[QuestionSynthesis, ...],
    analyses: dict[str, PaperAnalysis | None],
    papers: dict[str, PaperRecord],
) -> tuple[Viewpoint, ...]:
    del analyses, papers
    viewpoints: list[Viewpoint] = []
    for synthesis in syntheses:
        if synthesis.supporting_paper_ids:
            evidence = tuple(
                item
                for item in synthesis.representative_evidence
                if item.document_id in synthesis.supporting_paper_ids
            )
            if evidence:
                viewpoints.append(
                    Viewpoint(
                        claim=(
                            f"{synthesis.label}：{synthesis.conclusion}"
                        ),
                        stance="支持",
                        supporting_papers=synthesis.supporting_paper_ids,
                        evidence=evidence,
                        claim_origin="论文级结果证据综合",
                        stance_origin=(
                            f"本地结果规则推断；{AUTO_GRADE_DISCLAIMER}"
                        ),
                    )
                )
        negative_paper_ids = tuple(
            paper_id
            for paper_id in synthesis.negative_paper_ids
            if paper_id in {
                item.document_id for item in synthesis.representative_evidence
            }
        )
        negative_evidence = tuple(
            item
            for item in synthesis.representative_evidence
            if item.document_id in negative_paper_ids
        )
        if negative_paper_ids and negative_evidence:
            viewpoints.append(
                Viewpoint(
                    claim=(
                        f"{synthesis.label}：自动候选包含明确负向结果，"
                        "但尚待人工核对研究可比性。"
                    ),
                    stance="负向结果",
                    supporting_papers=negative_paper_ids,
                    evidence=negative_evidence,
                    claim_origin="论文级结果证据综合",
                    stance_origin=(
                        f"本地结果规则推断；{AUTO_GRADE_DISCLAIMER}"
                    ),
                )
            )
        null_paper_ids = tuple(
            paper_id
            for paper_id in synthesis.null_paper_ids
            if paper_id in {
                item.document_id for item in synthesis.representative_evidence
            }
        )
        null_evidence = tuple(
            item
            for item in synthesis.representative_evidence
            if item.document_id in null_paper_ids
        )
        if null_paper_ids and null_evidence:
            viewpoints.append(
                Viewpoint(
                    claim=(
                        f"{synthesis.label}：部分论文报告未达统计显著；"
                        "该结果不等同于效应方向相反。"
                    ),
                    stance="未达统计显著",
                    supporting_papers=null_paper_ids,
                    evidence=null_evidence,
                    claim_origin="论文级结果证据综合",
                    stance_origin=(
                        f"本地结果规则推断；{AUTO_GRADE_DISCLAIMER}"
                    ),
                )
            )
    return tuple(viewpoints)


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


def _rank_evidence_for_coverage(
    evidence: tuple[EvidenceRef, ...],
    query: str,
    *,
    per_paper_limit: int,
) -> tuple[EvidenceRef, ...]:
    """为每篇有正文的论文保留可比较证据，避免固定条数漏掉新论文。"""

    terms = _query_terms(query)
    by_document: dict[str, list[tuple[int, EvidenceRef]]] = {}
    seen_quotes_by_document: dict[str, set[str]] = {}
    for item in evidence:
        seen_quotes = seen_quotes_by_document.setdefault(
            item.document_id,
            set(),
        )
        if item.quote in seen_quotes:
            continue
        seen_quotes.add(item.quote)
        text = item.quote.lower()
        score = sum(3 for term in terms if term in text)
        by_document.setdefault(item.document_id, []).append((score, item))

    output: list[EvidenceRef] = []
    for document_id in sorted(by_document):
        ranked = sorted(
            by_document[document_id],
            key=lambda pair: (-pair[0], pair[1].page_number),
        )
        output.extend(
            item for _, item in ranked[:per_paper_limit]
        )
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


def _unique_question_evidence(
    evidence: Iterable[EvidenceRef],
) -> tuple[EvidenceRef, ...]:
    unique: dict[str, EvidenceRef] = {}
    for item in evidence:
        unique.setdefault(item.chunk_id, item)
    return tuple(unique.values())


def _five_dimension_subjective_rubric(
    evidence: tuple[EvidenceRef, ...],
    *,
    comparison_required: bool,
) -> tuple[RubricCriterion, ...]:
    """使用五项统一的科研主观题评分维度。"""

    evidence_keywords = _keywords_from_evidence(evidence)
    comparison_keywords = (
        ("比较", "不同", "相同", "样本", "测量")
        if comparison_required
        else ("研究设计", "样本", "证据边界")
    )
    return (
        RubricCriterion(
            label="核心观点准确性",
            description="核心结论与指定论文原文一致，不混入原文未支持的判断。",
            points=4.0,
            expected_keywords=evidence_keywords,
        ),
        RubricCriterion(
            label="文献证据引用能力",
            description="引用论文、原文或页码，而不是只写个人印象。",
            points=3.0,
            expected_keywords=(
                *evidence_keywords[:8],
                "原文",
                "论文",
            ),
        ),
        RubricCriterion(
            label="多篇论文对比能力",
            description=(
                "比较两篇及以上的设计、样本、干预、测量或结果。"
                if comparison_required
                else "单篇题需说明该篇证据与结论边界，不能把单篇结果外推为共识。"
            ),
            points=2.0,
            expected_keywords=comparison_keywords,
        ),
        RubricCriterion(
            label="学术逻辑与推理能力",
            description="论证步骤前后一致，区分观察结果、解释和推断。",
            points=3.0,
            expected_keywords=("因为", "因此", "但是", "研究设计", "测量"),
        ),
        RubricCriterion(
            label="批判性思维能力",
            description="指出局限、偏倚、混杂或因果推断边界。",
            points=2.0,
            expected_keywords=(
                "局限",
                "偏倚",
                "混杂",
                "证据不足",
                "不能",
                "因果",
            ),
        ),
    )


def _build_deep_standard_answer(
    paper: PaperRecord,
    analysis: PaperAnalysis,
) -> str:
    population = _comparison_field(analysis, "population")
    methods = _comparison_field(analysis, "methods")
    results = _comparison_field(analysis, "results")
    conclusions = _comparison_field(analysis, "conclusions")
    limitations = _comparison_field(analysis, "limitations")
    design = _infer_study_design(paper.title, analysis)
    return "\n".join(
        (
            f"论文：《{paper.title}》",
            f"- 研究设计：{design}。",
            "- 研究对象与样本：" + _format_comparison_field(population),
            "- 研究方法：" + _format_comparison_field(methods),
            "- 研究结果：" + _format_comparison_field(results),
            "- 作者结论：" + _format_comparison_field(conclusions),
            "- 原文局限：" + _format_comparison_field(limitations),
            "- 证据边界：结果只能支持与原文样本、设计、测量和比较条件"
            "相对应的结论；作者未报告的外推范围不能补写。",
        )
    )


def _build_critical_standard_answer(
    first: tuple[PaperRecord, PaperAnalysis],
    second: tuple[PaperRecord, PaperAnalysis],
) -> str:
    lines = ["两篇论文的证据化思辨框架："]
    for index, (paper, analysis) in enumerate((first, second), start=1):
        design = _infer_study_design(paper.title, analysis)
        lines.extend(
            (
                f"{index}. 《{paper.title}》",
                f"   - 设计：{design}。",
                "   - 样本：" + _format_comparison_field(
                    _sample_summary(analysis)
                ),
                "   - 方法：" + _format_comparison_field(
                    _comparison_field(analysis, "methods")
                ),
                "   - 结果：" + _format_comparison_field(
                    _comparison_field(analysis, "results")
                ),
                "   - 结论：" + _format_comparison_field(
                    _comparison_field(analysis, "conclusions")
                ),
                "   - 局限：" + _format_comparison_field(
                    _comparison_field(analysis, "limitations")
                ),
            )
        )
    lines.extend(
        (
            "- 结论差异的可能来源：样本构成、研究设计、干预剂量、测量工具、"
            "统计模型和随访时间；这些只能作为可能解释。",
            "- 证据强弱判断：不能仅凭篇幅或期刊名称判断，需要核对设计、偏倚、"
            "效应量、置信区间与重复验证。",
            "- 因果边界：观察性关联不能自动升级为因果结论。",
            "- 后续研究：统一结局定义、明确运动剂量、报告效应量与置信区间，"
            "并针对当前局限设计更长随访或更代表性的样本。",
        )
    )
    return "\n".join(lines)


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
                claim_origin="论文原文摘录",
                stance_origin="本地关键词规则推断",
            )
        )
    return tuple(viewpoints)


def _classify_stance(quote: str) -> str:
    lowered = quote.lower()
    if any(term in lowered for term in NULL_TERMS):
        return "未达统计显著"
    if any(term in lowered for term in NEGATIVE_TERMS):
        return "负向结果"
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


def _assessment_evidence_pool(
    page_evidence: tuple[EvidenceRef, ...],
    analyses: Iterable[tuple[PaperRecord, PaperAnalysis]],
) -> tuple[EvidenceRef, ...]:
    """合并全文与摘要分析证据，供所有题型共享并保持来源可追溯。"""

    unique: dict[str, EvidenceRef] = {}
    for item in page_evidence:
        if item.source_type != "元数据":
            unique.setdefault(item.chunk_id, item)
    for _, analysis in analyses:
        for item in analysis.all_evidence:
            if item.source_type != "元数据":
                unique.setdefault(item.chunk_id, item)
    return tuple(unique.values())


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
        "深度理解题",
        "学术思辨题",
        "个性化补学题",
    )


def _module_for_question_type(question_type: str) -> str:
    if question_type in {"单项选择题", "多项选择题", "判断题", "简答题"}:
        return "基础知识理解"
    if question_type == "深度理解题":
        return "文献深度理解"
    if question_type == "文献对比分析题":
        return "文献对比分析"
    if question_type in {"论述题", "学术思辨题"}:
        return "学术思辨训练"
    return "个性化错题与补学"


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
    if question_type == "深度理解题":
        return ("文献理解", "批判性思维")
    if question_type == "学术思辨题":
        return ("批判性思维", "学术应用", "对比分析")
    if question_type == "个性化补学题":
        return ("概念记忆", "文献理解", "对比分析", "批判性思维", "学术应用")
    return ("文献理解",)


def _local_rubric_score(
    question: AssessmentQuestion,
    user_answer: str,
) -> ScoreResult:
    normalized = user_answer.lower()
    evidence_text = " ".join(item.quote for item in question.evidence)
    evidence_overlap = _text_evidence_overlap(
        normalized,
        evidence_text,
    )
    citation_strength = _citation_strength(
        normalized,
        question.evidence,
    )
    comparison_strength = _marker_strength(
        normalized,
        (
            "相比",
            "不同",
            "相同",
            "两篇",
            "前者",
            "后者",
            "whereas",
            "compared",
        ),
    )
    logic_strength = _marker_strength(
        normalized,
        ("因为", "因此", "所以", "但是", "然而", "首先", "其次", "综上"),
    )
    critical_strength = _marker_strength(
        normalized,
        (
            "局限",
            "偏倚",
            "混杂",
            "样本",
            "外推",
            "证据不足",
            "不能",
            "因果",
        ),
    )
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
        keyword_ratio = len(matched) / len(keywords) if keywords else 0.0
        if rubric.label == "核心观点准确性":
            ratio = max(keyword_ratio, evidence_overlap)
        elif rubric.label == "文献证据引用能力":
            ratio = max(keyword_ratio, citation_strength)
        elif rubric.label == "多篇论文对比能力":
            ratio = max(keyword_ratio, comparison_strength, evidence_overlap * 0.5)
        elif rubric.label == "学术逻辑与推理能力":
            ratio = max(keyword_ratio, logic_strength)
        elif rubric.label in {"批判性思维", "批判性思维能力"}:
            ratio = max(keyword_ratio, critical_strength)
        else:
            ratio = max(keyword_ratio, evidence_overlap * 0.6)
        ratio = max(0.0, min(1.0, ratio))
        reason = (
            f"证据文本重合度 {evidence_overlap:.0%}；"
            f"引用定位强度 {citation_strength:.0%}；"
            f"关键词命中 {len(matched)}/{len(keywords) if keywords else 0}。"
            "评分不使用回答篇幅或语言流畅度。"
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
            "本地评分综合原文文本重合、引用定位、比较、逻辑和批判性标记，"
            "不按回答篇幅或语言流畅度加分。该结果用于学习反馈，不等同于"
            "教师或正式同行评议。"
        ),
        criteria=tuple(criteria),
        mode="本地评分细则（非大模型）",
        user_answer=user_answer,
    )


def _normalized_evidence_units(value: str) -> set[str]:
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", value.casefold())
    units = {
        normalized[index : index + 2]
        for index in range(max(0, len(normalized) - 1))
    }
    units.update(
        token
        for token in re.findall(r"[a-z]{4,}", value.casefold())
        if token
    )
    return {unit for unit in units if unit.strip()}


def _text_evidence_overlap(answer: str, evidence: str) -> float:
    answer_units = _normalized_evidence_units(answer)
    evidence_units = _normalized_evidence_units(evidence)
    if not answer_units or not evidence_units:
        return 0.0
    matched = answer_units & evidence_units
    return min(1.0, len(matched) / max(8, min(len(answer_units), 40)))


def _citation_strength(
    answer: str,
    evidence: tuple[EvidenceRef, ...],
) -> float:
    markers = 0
    if re.search(r"\b10\.\d{4,9}/\S+", answer):
        markers += 1
    if re.search(r"第\s*\d+\s*页", answer):
        markers += 1
    source_titles = " ".join(item.document_title for item in evidence).casefold()
    if any(
        token in answer
        for token in re.findall(r"[\u4e00-\u9fff]{4,}|[a-z]{5,}", source_titles)
    ):
        markers += 1
    if any(item.file_name.casefold() in answer for item in evidence):
        markers += 1
    return min(1.0, markers / 2)


def _marker_strength(answer: str, markers: Iterable[str]) -> float:
    matched = sum(marker in answer for marker in markers)
    return min(1.0, matched / 3)


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
    "AssessmentGenerationError",
    "KnowledgeSynthesizer",
    "LearningAssessmentAgent",
    "LiteratureCollector",
    "PaperReader",
    "PersonalTutorAgent",
]
