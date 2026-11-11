"""ResearchMind 的顺序编排层。

这一层把六个 Agent、真实论文清单、SQLite 论文库和学习记录串成完整工作流。
当前使用清晰可调试的顺序调用；未来接入 LangGraph 时，可以把每个阶段映射为
图节点，而不需要重写 Agent 和存储层。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from config import Settings, settings as default_settings
from researchmind.agents import (
    AcademicDebateAnalyzer,
    KnowledgeSynthesizer,
    LearningAssessmentAgent,
    PaperReader,
    PersonalTutorAgent,
)
from researchmind.models import (
    AssessmentQuestion,
    DebateAnalysis,
    EvidenceRef,
    PaperAnalysis,
    PaperRecord,
    ResearchDomain,
    ResearchSynthesis,
    ScoreResult,
    StudyRecommendation,
)
from researchmind.storage import ResearchMindStore
from services.llm_service import LLMService
from tools.knowledge_base import (
    BundledKnowledgeBase,
    load_bundled_knowledge_base,
)


DEFAULT_DOMAIN_NAME = "大学生体育锻炼行为与体质健康"
DEFAULT_DOMAIN_KEYWORDS = (
    "体育锻炼",
    "体质健康",
    "大学生",
    "身体活动",
    "运动干预",
)
DEFAULT_DEBATE_TOPIC = "大学生体育锻炼行为与体质健康水平的关系"


class ResearchMindWorkflowError(RuntimeError):
    """工作流无法完成某一阶段时抛出的可展示错误。"""


@dataclass(frozen=True)
class LibrarySyncResult:
    """真实论文库同步到 ResearchMind SQLite 的结果。"""

    domain: ResearchDomain
    document_count: int
    papers_written: int
    papers_removed: int
    evidence_written: int
    total_papers: int
    total_evidence: int
    used_cached_sync: bool
    message: str


@dataclass(frozen=True)
class PipelineResult:
    """阅读、争议、知识体系和自动组卷的完整结果。"""

    domain: ResearchDomain
    analyses: tuple[PaperAnalysis, ...]
    debate: DebateAnalysis
    synthesis: ResearchSynthesis
    questions: tuple[AssessmentQuestion, ...]
    summary: str


@dataclass(frozen=True)
class LearningCycleResult:
    """一次答题、批改、错题分析和下一轮组卷结果。"""

    domain_id: str
    scores: tuple[tuple[AssessmentQuestion, ScoreResult], ...]
    mastery: dict[str, float]
    recommendation: StudyRecommendation
    next_questions: tuple[AssessmentQuestion, ...]
    summary: str


class ResearchMindWorkflow:
    """ResearchMind 的可调试顺序工作流。"""

    def __init__(
        self,
        *,
        store: ResearchMindStore | None = None,
        config: Settings | None = None,
        knowledge_base_dir: Path | str | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
        self.config = config or default_settings
        self.store = store or ResearchMindStore()
        self.knowledge_base_dir = Path(
            knowledge_base_dir or self.config.knowledge_base_dir
        ).resolve()
        self.llm_service = llm_service or LLMService(self.config)
        self.reader = PaperReader(self.store)
        self.debate_analyzer = AcademicDebateAnalyzer(self.store)
        self.synthesizer = KnowledgeSynthesizer(self.store)
        self.assessment = LearningAssessmentAgent(
            self.store,
            self.llm_service,
        )
        self.tutor = PersonalTutorAgent(self.store, self.assessment)

    def ensure_default_domain(self) -> ResearchDomain:
        """创建默认领域；已有同名领域时直接复用。"""

        for domain in self.store.list_domains():
            if domain.name == DEFAULT_DOMAIN_NAME:
                return domain
        domain = ResearchDomain.create(
            name=DEFAULT_DOMAIN_NAME,
            description=(
                "围绕大学生体育锻炼行为、体质健康、心理社会影响因素与"
                "运动干预证据建立的可追溯研究领域。"
            ),
            keywords=DEFAULT_DOMAIN_KEYWORDS,
            created_at=_now(),
        )
        self.store.upsert_domain(domain)
        return domain

    def create_domain(
        self,
        *,
        name: str,
        description: str,
        keywords: Iterable[str],
    ) -> ResearchDomain:
        """创建用户自定义研究领域，名称重复时返回已有领域。"""

        clean_name = name.strip()
        for domain in self.store.list_domains():
            if domain.name == clean_name:
                return domain
        domain = ResearchDomain.create(
            name=clean_name,
            description=description,
            keywords=tuple(keywords),
            created_at=_now(),
        )
        self.store.upsert_domain(domain)
        return domain

    def sync_bundled_library(
        self,
        *,
        domain: ResearchDomain | None = None,
        force: bool = False,
    ) -> LibrarySyncResult:
        """把 manifest.json 与 evidence_chunks.json 同步到 SQLite。

        新 PDF 通过现有 ``tools.library_ingest`` 导入后，会出现在 manifest 和
        证据索引中；再次调用本方法即可自动进入 ResearchMind，不需要手工建表
        或复制证据。
        """

        target_domain = domain or self.ensure_default_domain()
        try:
            bundle = load_bundled_knowledge_base(self.knowledge_base_dir)
        except Exception as exc:
            raise ResearchMindWorkflowError(
                f"无法读取项目论文库：{exc}"
            ) from exc

        signature = _library_signature(self.knowledge_base_dir, bundle)
        metadata_key = f"library_signature:{target_domain.domain_id}"
        cached = self.store.get_metadata(metadata_key)
        if (
            not force
            and cached == signature
            and self.store.paper_count(target_domain.domain_id)
            == len(bundle.documents)
            and self.store.evidence_count(target_domain.domain_id)
            == len(bundle.chunks)
        ):
            return LibrarySyncResult(
                domain=target_domain,
                document_count=len(bundle.documents),
                papers_written=0,
                papers_removed=0,
                evidence_written=0,
                total_papers=self.store.paper_count(target_domain.domain_id),
                total_evidence=self.store.evidence_count(
                    target_domain.domain_id
                ),
                used_cached_sync=True,
                message="论文库未变化，已复用 SQLite 中的真实元数据和证据。",
            )

        documents_by_file = {
            document.file_name: document for document in bundle.documents
        }
        chunks_by_file: dict[str, list[EvidenceRef]] = {
            file_name: [] for file_name in documents_by_file
        }
        for chunk in bundle.chunks:
            document = documents_by_file.get(chunk.document_name)
            if document is None:
                continue
            paper_id = _paper_id_for_domain(
                target_domain,
                document.document_id,
            )
            chunks_by_file[chunk.document_name].append(
                EvidenceRef(
                    chunk_id=_chunk_id_for_domain(
                        target_domain,
                        chunk.chunk_id,
                    ),
                    document_id=paper_id,
                    document_title=_document_title(document.title, document.subtitle),
                    file_name=document.file_name,
                    page_number=chunk.page_number,
                    quote=chunk.text,
                )
            )

        pdf_dir = self.knowledge_base_dir / "pdfs"
        written_evidence = 0
        expected_paper_ids: set[str] = set()
        for document in bundle.documents:
            title = _document_title(document.title, document.subtitle)
            paper_id = _paper_id_for_domain(
                target_domain,
                document.document_id,
            )
            expected_paper_ids.add(paper_id)
            paper = PaperRecord(
                paper_id=paper_id,
                domain_id=target_domain.domain_id,
                title=title,
                authors=document.authors,
                year=document.year,
                journal=document.journal,
                doi=document.doi,
                source_url=document.source_url,
                abstract="",
                keywords=document.keywords,
                pdf_path=str((pdf_dir / document.file_name).resolve()),
                source="项目内置真实论文清单",
                verification_status=(
                    "PDF SHA-256 已核验；书目信息来自 manifest.json"
                ),
                created_at=_now(),
            )
            self.store.upsert_paper(paper)
            evidence = tuple(chunks_by_file.get(document.file_name, []))
            self.store.replace_evidence(paper_id, evidence)
            written_evidence += len(evidence)

        papers_removed = self.store.delete_papers_except(
            target_domain.domain_id,
            expected_paper_ids,
        )
        self.store.set_metadata(metadata_key, signature)
        total_papers = self.store.paper_count(target_domain.domain_id)
        total_evidence = self.store.evidence_count(target_domain.domain_id)
        return LibrarySyncResult(
            domain=target_domain,
            document_count=len(bundle.documents),
            papers_written=len(bundle.documents),
            papers_removed=papers_removed,
            evidence_written=written_evidence,
            total_papers=total_papers,
            total_evidence=total_evidence,
            used_cached_sync=False,
            message=(
                f"已同步 {len(bundle.documents)} 篇真实论文和 "
                f"{written_evidence} 条带页码证据"
                + (
                    f"，并移除 {papers_removed} 篇已不属于当前论文库的记录。"
                    if papers_removed
                    else "。"
                )
            ),
        )

    def ensure_ready(
        self,
        domain_id: str | None = None,
    ) -> ResearchDomain:
        """确认领域存在并完成论文库同步。"""

        domain = (
            self.store.get_domain(domain_id)
            if domain_id
            else self.ensure_default_domain()
        )
        if domain is None:
            raise ResearchMindWorkflowError("研究领域不存在。")
        self.store.upsert_domain(domain)
        if self.store.paper_count(domain.domain_id) == 0:
            self.sync_bundled_library(domain=domain)
        return domain

    def run_analysis_pipeline(
        self,
        domain_id: str,
        *,
        topic: str = DEFAULT_DEBATE_TOPIC,
        generate_questions: bool = True,
        question_types: Iterable[str] | None = None,
    ) -> PipelineResult:
        """执行阅读、观点比较、知识体系和自动组卷。"""

        domain = self.ensure_ready(domain_id)
        if self.store.paper_count(domain.domain_id) == 0:
            raise ResearchMindWorkflowError("当前领域没有可分析论文。")
        analyses = self.reader.read_domain(domain.domain_id)
        debate = self.debate_analyzer.analyze(domain.domain_id, topic)
        synthesis = self.synthesizer.synthesize(domain.domain_id, debate=debate)
        questions: tuple[AssessmentQuestion, ...] = ()
        if generate_questions:
            questions = self.assessment.generate_question_set(
                domain.domain_id,
                question_types=question_types,
                debate_topic=topic,
            )
        return PipelineResult(
            domain=domain,
            analyses=analyses,
            debate=debate,
            synthesis=synthesis,
            questions=questions,
            summary=(
                f"完成 {len(analyses)} 篇论文结构化阅读、"
                f"{len(debate.viewpoints)} 个可追溯观点分析、"
                f"{len(synthesis.nodes)} 个知识图谱节点和 "
                f"{len(questions)} 道试题。"
            ),
        )

    def grade_learning_cycle(
        self,
        domain_id: str,
        *,
        answers: Mapping[str, str],
        topic: str = DEFAULT_DEBATE_TOPIC,
        generate_next: bool = True,
    ) -> LearningCycleResult:
        """批改一组答案，更新掌握度，并生成下一轮针对性试题。"""

        domain = self.ensure_ready(domain_id)
        graded: list[tuple[AssessmentQuestion, ScoreResult]] = []
        for question_id, answer in answers.items():
            question = self.store.get_question(question_id)
            if question is None:
                continue
            if question.domain_id != domain.domain_id:
                continue
            score = self.tutor.grade_and_record(question, str(answer))
            graded.append((question, score))
        if not graded:
            raise ResearchMindWorkflowError("没有可批改的有效题目。")

        mastery = self.tutor.update_mastery(domain.domain_id)
        next_questions: tuple[AssessmentQuestion, ...] = ()
        if generate_next:
            next_questions = self.assessment.generate_question_set(
                domain.domain_id,
                question_types=_next_question_types(mastery),
                debate_topic=topic,
            )
        recommendation = self.tutor.recommend(
            domain.domain_id,
            next_question_ids=tuple(
                question.question_id for question in next_questions
            ),
        )
        weak = "、".join(recommendation.weak_dimensions) or "暂无低于 60 分的维度"
        return LearningCycleResult(
            domain_id=domain.domain_id,
            scores=tuple(graded),
            mastery=mastery,
            recommendation=recommendation,
            next_questions=next_questions,
            summary=(
                f"已按评分细则批改 {len(graded)} 道题；薄弱维度：{weak}；"
                f"生成下一轮试题 {len(next_questions)} 道。"
            ),
        )

    def run_end_to_end_acceptance(
        self,
        *,
        domain_id: str | None = None,
        topic: str = DEFAULT_DEBATE_TOPIC,
    ) -> tuple[PipelineResult, LearningCycleResult]:
        """使用标准答案回填执行流程自检。

        该结果只用于验证系统链路，不代表真实学生成绩，也不能作为科研数据。
        """

        domain = self.ensure_ready(domain_id)
        pipeline = self.run_analysis_pipeline(
            domain.domain_id,
            topic=topic,
        )
        answers = {
            question.question_id: question.standard_answer
            for question in pipeline.questions
        }
        learning = self.grade_learning_cycle(
            domain.domain_id,
            answers=answers,
            topic=topic,
            generate_next=True,
        )
        return pipeline, learning


def _library_signature(
    root: Path,
    bundle: BundledKnowledgeBase,
) -> str:
    """用清单、证据索引、题名和 DOI 生成同步签名。"""

    digest = hashlib.sha256()
    for relative in ("manifest.json", "index/evidence_chunks.json"):
        path = root / relative
        if path.is_file():
            digest.update(path.read_bytes())
    digest.update(str(len(bundle.documents)).encode("utf-8"))
    digest.update(str(len(bundle.chunks)).encode("utf-8"))
    return digest.hexdigest()


def _document_title(title: str, subtitle: str) -> str:
    clean_subtitle = subtitle.strip()
    if not clean_subtitle:
        return title.strip()
    return f"{title.strip()}{clean_subtitle}"


def _paper_id_for_domain(
    domain: ResearchDomain,
    document_id: str,
) -> str:
    """默认领域沿用清单 ID，自定义领域使用稳定且不冲突的本地 ID。"""

    if domain.name == DEFAULT_DOMAIN_NAME:
        return document_id
    return f"{domain.domain_id}-paper-{document_id}"


def _chunk_id_for_domain(
    domain: ResearchDomain,
    chunk_id: str,
) -> str:
    """避免同一内置片段同步到多个领域时触发主键冲突。"""

    if domain.name == DEFAULT_DOMAIN_NAME:
        return chunk_id
    return f"{domain.domain_id}-chunk-{chunk_id}"


def _next_question_types(
    mastery: Mapping[str, float],
) -> tuple[str, ...]:
    """依据薄弱能力维度选择下一轮题型。"""

    requested: list[str] = []
    if mastery.get("概念记忆", 100.0) < 60.0:
        requested.extend(("单项选择题", "判断题"))
    if mastery.get("文献理解", 100.0) < 60.0:
        requested.extend(("多项选择题", "简答题"))
    if mastery.get("对比分析", 100.0) < 60.0:
        requested.append("文献对比分析题")
    if mastery.get("批判性思维", 100.0) < 60.0:
        requested.extend(("文献对比分析题", "论述题"))
    if mastery.get("学术应用", 100.0) < 60.0:
        requested.extend(("论述题", "简答题"))
    if not requested:
        requested.extend(("多项选择题", "文献对比分析题", "论述题"))
    return tuple(dict.fromkeys(requested))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


__all__ = [
    "DEFAULT_DEBATE_TOPIC",
    "DEFAULT_DOMAIN_NAME",
    "LearningCycleResult",
    "LibrarySyncResult",
    "PipelineResult",
    "ResearchMindWorkflow",
    "ResearchMindWorkflowError",
]
