"""ResearchMind 的严格数据结构。

所有学术结论都必须能够回指真实论文、证据片段和页码。这里不使用自由格式
字典承载核心结论，避免界面或 Agent 在传递过程中丢失来源信息。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


DIMENSIONS = (
    "概念记忆",
    "文献理解",
    "对比分析",
    "批判性思维",
    "学术应用",
)

QUESTION_TYPES = (
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

ASSESSMENT_MODULES = (
    "基础知识理解",
    "文献深度理解",
    "文献对比分析",
    "学术思辨训练",
    "个性化错题与补学",
)

OBJECTIVE_TYPES = frozenset(
    {"单项选择题", "多项选择题", "判断题"}
)


def _clean_text(value: str, fallback: str = "") -> str:
    """清理字段空白，并保持证据原文内容不被改写。"""

    cleaned = str(value or "").strip()
    return cleaned or fallback


@dataclass(frozen=True)
class EvidenceRef:
    """一条可定位到真实论文全文、摘要或元数据的证据。"""

    chunk_id: str
    document_id: str
    document_title: str
    file_name: str
    page_number: int
    quote: str
    source_type: str = "全文"
    section_hint: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ValueError("证据片段 ID 不能为空。")
        if not self.document_id.strip():
            raise ValueError("证据必须关联论文 ID。")
        if not self.document_title.strip():
            raise ValueError("证据必须关联论文题名。")
        if not self.file_name.strip():
            raise ValueError("证据必须关联 PDF 文件名。")
        if self.page_number <= 0:
            raise ValueError("证据页码必须是正整数。")
        if not self.quote.strip():
            raise ValueError("证据原文不能为空。")

    @property
    def citation(self) -> str:
        """返回可直接展示给用户的来源定位。"""

        if self.source_type == "摘要":
            return f"{self.document_title}，摘要证据（原文未提供可检索全文）"
        return (
            f"{self.document_title}（{self.file_name}），"
            f"第 {self.page_number} 页"
        )

    @property
    def location_label(self) -> str:
        if self.source_type == "摘要":
            return "摘要（未提供可检索全文）"
        if self.source_type == "元数据":
            return "论文元数据"
        return f"第 {self.page_number} 页"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "file_name": self.file_name,
            "page_number": self.page_number,
            "quote": self.quote,
            "source_type": self.source_type,
            "section_hint": self.section_hint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceRef":
        return cls(
            chunk_id=str(data["chunk_id"]),
            document_id=str(data["document_id"]),
            document_title=str(data["document_title"]),
            file_name=str(data["file_name"]),
            page_number=int(data["page_number"]),
            quote=str(data["quote"]),
            source_type=str(data.get("source_type", "全文")),
            section_hint=str(data.get("section_hint", "")),
        )


@dataclass(frozen=True)
class ResearchDomain:
    """一个可持续更新文献库的学术研究领域。"""

    domain_id: str
    name: str
    description: str
    keywords: tuple[str, ...]
    created_at: str

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        keywords: tuple[str, ...],
        created_at: str,
    ) -> "ResearchDomain":
        clean_name = _clean_text(name)
        if not clean_name:
            raise ValueError("领域名称不能为空。")
        digest = hashlib.sha1(clean_name.encode("utf-8")).hexdigest()[:12]
        return cls(
            domain_id=f"domain-{digest}",
            name=clean_name,
            description=_clean_text(description, "暂无领域说明。"),
            keywords=tuple(word.strip() for word in keywords if word.strip()),
            created_at=created_at,
        )


@dataclass(frozen=True)
class PaperRecord:
    """ResearchMind 论文库中的一篇真实论文。

    检索导入的纯元数据论文与 manifest 管理的全文论文共用该模型。缺少全文时
    ``parse_status`` 会明确标为“仅元数据”，不能当作已完成全文解析。
    """

    paper_id: str
    domain_id: str
    title: str
    authors: tuple[str, ...]
    year: str
    journal: str
    doi: str
    source_url: str
    abstract: str
    keywords: tuple[str, ...]
    pdf_path: str
    source: str
    verification_status: str
    created_at: str
    imported_at: str = field(default="", compare=False)
    category: str = "待分类"
    parse_status: str = "仅元数据"
    analysis_status: str = "未分析"
    is_deleted: bool = False
    deleted_at: str = ""
    normalized_title: str = field(default="", compare=False)
    normalization_version: str = "v1"
    import_origin: str = ""
    is_manifest_managed: bool = True

    @property
    def authors_text(self) -> str:
        return "、".join(self.authors) if self.authors else "作者待核对"

    @property
    def citation(self) -> str:
        doi_text = f" DOI: {self.doi}" if self.doi else ""
        return (
            f"{self.authors_text}. {self.title}[J]. "
            f"{self.journal}, {self.year}.{doi_text}"
        )


@dataclass(frozen=True)
class PaperAnalysis:
    """PaperReader 对一篇论文的结构化阅读结果。

    每个字段保存的是可核验原文证据。证据不足时字段为空元组，界面显示
    “证据不足”，而不是补写一个看似完整的结论。
    """

    paper_id: str
    purpose: tuple[EvidenceRef, ...] = ()
    population: tuple[EvidenceRef, ...] = ()
    methods: tuple[EvidenceRef, ...] = ()
    interventions: tuple[EvidenceRef, ...] = ()
    comparators: tuple[EvidenceRef, ...] = ()
    measures: tuple[EvidenceRef, ...] = ()
    results: tuple[EvidenceRef, ...] = ()
    statistics: tuple[EvidenceRef, ...] = ()
    conclusions: tuple[EvidenceRef, ...] = ()
    limitations: tuple[EvidenceRef, ...] = ()
    extraction_mode: str = "本地规则提取"
    status: str = "证据不足"

    def field_evidence(self, field_name: str) -> tuple[EvidenceRef, ...]:
        """按字段名返回证据，供界面和测评统一读取。"""

        allowed = {
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
        }
        if field_name not in allowed:
            raise ValueError(f"未知论文结构字段：{field_name}")
        return tuple(getattr(self, field_name))

    @property
    def all_evidence(self) -> tuple[EvidenceRef, ...]:
        unique: dict[str, EvidenceRef] = {}
        for field_name in (
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
        ):
            for evidence in self.field_evidence(field_name):
                unique.setdefault(evidence.chunk_id, evidence)
        return tuple(unique.values())

    @property
    def filled_field_count(self) -> int:
        return sum(
            bool(self.field_evidence(name))
            for name in (
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

    def to_dict(self) -> dict[str, Any]:
        """返回可缓存、可序列化的结构化阅读结果。"""

        return {
            "paper_id": self.paper_id,
            "purpose": [item.to_dict() for item in self.purpose],
            "population": [item.to_dict() for item in self.population],
            "methods": [item.to_dict() for item in self.methods],
            "interventions": [item.to_dict() for item in self.interventions],
            "comparators": [item.to_dict() for item in self.comparators],
            "measures": [item.to_dict() for item in self.measures],
            "results": [item.to_dict() for item in self.results],
            "statistics": [item.to_dict() for item in self.statistics],
            "conclusions": [item.to_dict() for item in self.conclusions],
            "limitations": [item.to_dict() for item in self.limitations],
            "extraction_mode": self.extraction_mode,
            "status": self.status,
        }


@dataclass(frozen=True)
class CandidatePaper:
    """学术元数据源返回的候选论文，尚未视为已核验入库。"""

    candidate_id: str
    title: str
    authors: tuple[str, ...]
    year: str
    journal: str
    doi: str
    source_url: str
    abstract: str
    keywords: tuple[str, ...]
    source: str
    pdf_url: str = ""
    is_open_access: bool = False

    @property
    def authors_text(self) -> str:
        """返回适合中文界面展示的作者串。"""

        return "、".join(self.authors) if self.authors else "作者待核对"

    @property
    def has_open_pdf(self) -> bool:
        """只有来源明确标记开放获取且提供 PDF 地址时才允许下载。"""

        return self.is_open_access and bool(self.pdf_url.strip())


@dataclass(frozen=True)
class Viewpoint:
    """争议中的一个观点，必须关联原文和页码。"""

    claim: str
    stance: str
    supporting_papers: tuple[str, ...]
    evidence: tuple[EvidenceRef, ...]
    claim_origin: str = "原文摘录"
    stance_origin: str = "本地规则推断"

    def __post_init__(self) -> None:
        if not self.claim.strip():
            raise ValueError("观点内容不能为空。")
        if not self.supporting_papers:
            raise ValueError("观点必须关联至少一篇论文。")
        if not self.evidence:
            raise ValueError("观点必须提供原文证据和页码。")

    @property
    def page_numbers(self) -> tuple[int, ...]:
        return tuple(sorted({item.page_number for item in self.evidence}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "stance": self.stance,
            "supporting_papers": list(self.supporting_papers),
            "evidence": [item.to_dict() for item in self.evidence],
            "page_numbers": list(self.page_numbers),
            "claim_origin": self.claim_origin,
            "stance_origin": self.stance_origin,
        }


@dataclass(frozen=True)
class QuestionSynthesis:
    """一个具体研究问题的论文级证据综合。"""

    question_id: str
    label: str
    question: str
    conclusion: str
    status: str
    evidence_certainty: str
    certainty_reason: str
    supporting_paper_ids: tuple[str, ...]
    opposing_paper_ids: tuple[str, ...]
    inconclusive_paper_ids: tuple[str, ...]
    representative_evidence: tuple[EvidenceRef, ...]
    outcome_measures: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    analysis_time: str = ""
    negative_paper_ids: tuple[str, ...] = ()
    null_paper_ids: tuple[str, ...] = ()
    mixed_paper_ids: tuple[str, ...] = ()
    review_status: str = "未审核"

    def __post_init__(self) -> None:
        if not self.question_id.strip() or not self.question.strip():
            raise ValueError("研究问题不能为空。")
        if not self.status.strip():
            raise ValueError("证据综合状态不能为空。")

    @property
    def related_paper_count(self) -> int:
        return len(
            set(self.supporting_paper_ids)
            | set(self.opposing_paper_ids)
            | set(self.inconclusive_paper_ids)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "label": self.label,
            "question": self.question,
            "conclusion": self.conclusion,
            "status": self.status,
            "evidence_certainty": self.evidence_certainty,
            "certainty_reason": self.certainty_reason,
            "supporting_paper_ids": list(self.supporting_paper_ids),
            "opposing_paper_ids": list(self.opposing_paper_ids),
            "inconclusive_paper_ids": list(self.inconclusive_paper_ids),
            "representative_evidence": [
                item.to_dict() for item in self.representative_evidence
            ],
            "outcome_measures": list(self.outcome_measures),
            "limitations": list(self.limitations),
            "analysis_time": self.analysis_time,
            "negative_paper_ids": list(self.negative_paper_ids),
            "null_paper_ids": list(self.null_paper_ids),
            "mixed_paper_ids": list(self.mixed_paper_ids),
            "review_status": self.review_status,
        }


@dataclass(frozen=True)
class ResearchConflict:
    """同一研究问题中由论文级结果差异形成的学术争议。"""

    conflict_id: str
    question_id: str
    question: str
    viewpoint_a: str
    paper_ids_a: tuple[str, ...]
    evidence_a: tuple[EvidenceRef, ...]
    viewpoint_b: str
    paper_ids_b: tuple[str, ...]
    evidence_b: tuple[EvidenceRef, ...]
    divergence_reasons: tuple[str, ...]
    current_conclusion: str
    unresolved_questions: tuple[str, ...]
    evidence_certainty: str
    analysis_time: str = ""
    review_status: str = "待审核候选"
    comparability_status: str = "未自动判定，需同结局与干预条件人工复核"
    significance_note: str = (
        "仅将正向与负向结果列为候选对立方向；"
        "未达统计显著不视为方向相反"
    )

    def __post_init__(self) -> None:
        if not self.paper_ids_a or not self.paper_ids_b:
            raise ValueError("争议双方都必须关联真实论文。")
        if not self.evidence_a or not self.evidence_b:
            raise ValueError("争议双方都必须提供可定位原文证据。")

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "question_id": self.question_id,
            "question": self.question,
            "viewpoint_a": self.viewpoint_a,
            "paper_ids_a": list(self.paper_ids_a),
            "evidence_a": [item.to_dict() for item in self.evidence_a],
            "viewpoint_b": self.viewpoint_b,
            "paper_ids_b": list(self.paper_ids_b),
            "evidence_b": [item.to_dict() for item in self.evidence_b],
            "divergence_reasons": list(self.divergence_reasons),
            "current_conclusion": self.current_conclusion,
            "unresolved_questions": list(self.unresolved_questions),
            "evidence_certainty": self.evidence_certainty,
            "analysis_time": self.analysis_time,
            "review_status": self.review_status,
            "comparability_status": self.comparability_status,
            "significance_note": self.significance_note,
        }


@dataclass(frozen=True)
class DebateAnalysis:
    """学术分歧分析结果，字段与任务要求的数据结构一致。"""

    topic: str
    viewpoints: tuple[Viewpoint, ...]
    possible_reasons: tuple[str, ...]
    limitations: tuple[str, ...]
    evidence_strength: str
    research_gaps: tuple[str, ...]
    consensus_summary: str = "证据不足，暂不能判定学术共识。"
    disagreement_summary: str = "当前证据不足以确认学术分歧。"
    analysis_mode: str = "本地规则证据归纳"
    considered_paper_ids: tuple[str, ...] = ()
    evidence_covered_paper_ids: tuple[str, ...] = ()
    coverage_note: str = ""
    question_syntheses: tuple[QuestionSynthesis, ...] = ()
    conflicts: tuple[ResearchConflict, ...] = ()
    analysis_time: str = ""
    engine_version: str = ""

    def __post_init__(self) -> None:
        if not self.topic.strip():
            raise ValueError("争议主题不能为空。")
        for viewpoint in self.viewpoints:
            if not viewpoint.evidence:
                raise ValueError("争议观点缺少来源证据。")

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "viewpoints": [item.to_dict() for item in self.viewpoints],
            "possible_reasons": list(self.possible_reasons),
            "limitations": list(self.limitations),
            "evidence_strength": self.evidence_strength,
            "research_gaps": list(self.research_gaps),
            "consensus_summary": self.consensus_summary,
            "disagreement_summary": self.disagreement_summary,
            "analysis_mode": self.analysis_mode,
            "considered_paper_ids": list(self.considered_paper_ids),
            "evidence_covered_paper_ids": list(
                self.evidence_covered_paper_ids
            ),
            "coverage_note": self.coverage_note,
            "question_syntheses": [
                item.to_dict() for item in self.question_syntheses
            ],
            "conflicts": [item.to_dict() for item in self.conflicts],
            "analysis_time": self.analysis_time,
            "engine_version": self.engine_version,
        }


@dataclass(frozen=True)
class KnowledgeNode:
    """知识图谱节点。"""

    node_id: str
    label: str
    node_type: str
    description: str
    evidence_count: int


@dataclass(frozen=True)
class KnowledgeEdge:
    """知识图谱关系；关系没有证据时应由调用方跳过。"""

    source_id: str
    target_id: str
    relation: str
    evidence_count: int


@dataclass(frozen=True)
class ResearchSynthesis:
    """KnowledgeSynthesizer 生成的领域知识体系。"""

    domain_id: str
    concepts: tuple[str, ...]
    theory_framework: tuple[str, ...]
    consensus: tuple[str, ...]
    disagreements: tuple[str, ...]
    limitations: tuple[str, ...]
    research_gaps: tuple[str, ...]
    innovation_directions: tuple[str, ...]
    source_paper_ids: tuple[str, ...]
    generated_at: str
    nodes: tuple[KnowledgeNode, ...] = ()
    edges: tuple[KnowledgeEdge, ...] = ()
    mode: str = "本地证据归纳"

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "concepts": list(self.concepts),
            "theory_framework": list(self.theory_framework),
            "consensus": list(self.consensus),
            "disagreements": list(self.disagreements),
            "limitations": list(self.limitations),
            "research_gaps": list(self.research_gaps),
            "innovation_directions": list(self.innovation_directions),
            "source_paper_ids": list(self.source_paper_ids),
            "generated_at": self.generated_at,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "label": node.label,
                    "node_type": node.node_type,
                    "description": node.description,
                    "evidence_count": node.evidence_count,
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relation": edge.relation,
                    "evidence_count": edge.evidence_count,
                }
                for edge in self.edges
            ],
            "mode": self.mode,
        }


@dataclass(frozen=True)
class RubricCriterion:
    """主观题评分细则中的一项。"""

    label: str
    description: str
    points: float
    expected_keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class AssessmentQuestion:
    """可追溯来源的测评题目。"""

    question_id: str
    domain_id: str
    question_type: str
    prompt: str
    options: tuple[str, ...]
    standard_answer: str
    rubric: tuple[RubricCriterion, ...]
    knowledge_tags: tuple[str, ...]
    difficulty: str
    source_paper_ids: tuple[str, ...]
    evidence: tuple[EvidenceRef, ...]
    created_at: str
    scoring_mode: str = "客观题自动评分"
    explanation: str = ""
    assessment_module: str = "基础知识理解"

    def __post_init__(self) -> None:
        if self.question_type not in QUESTION_TYPES:
            raise ValueError(f"不支持的题型：{self.question_type}")
        if self.assessment_module not in ASSESSMENT_MODULES:
            raise ValueError(f"不支持的测评模块：{self.assessment_module}")
        if not self.prompt.strip():
            raise ValueError("题目不能为空。")
        if not self.source_paper_ids:
            raise ValueError("每道题必须关联来源论文。")
        if not self.evidence:
            raise ValueError("每道题必须提供原文证据与页码。")
        if not self.rubric:
            raise ValueError("每道题必须提供评分标准。")

    @property
    def is_objective(self) -> bool:
        return self.question_type in OBJECTIVE_TYPES

    @property
    def max_score(self) -> float:
        return sum(item.points for item in self.rubric)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "domain_id": self.domain_id,
            "question_type": self.question_type,
            "prompt": self.prompt,
            "options": list(self.options),
            "standard_answer": self.standard_answer,
            "rubric": [
                {
                    "label": item.label,
                    "description": item.description,
                    "points": item.points,
                    "expected_keywords": list(item.expected_keywords),
                }
                for item in self.rubric
            ],
            "knowledge_tags": list(self.knowledge_tags),
            "difficulty": self.difficulty,
            "source_paper_ids": list(self.source_paper_ids),
            "evidence": [item.to_dict() for item in self.evidence],
            "created_at": self.created_at,
            "scoring_mode": self.scoring_mode,
            "explanation": self.explanation,
            "assessment_module": self.assessment_module,
        }


@dataclass(frozen=True)
class CriterionScore:
    """评分细则中一个维度的实际得分。"""

    label: str
    score: float
    max_score: float
    reason: str


@dataclass(frozen=True)
class ScoreResult:
    """一次答题评分及依据。"""

    score: float
    max_score: float
    feedback: str
    criteria: tuple[CriterionScore, ...]
    mode: str
    correct: bool | None = None
    user_answer: str = ""


@dataclass(frozen=True)
class MasteryRecord:
    """用户在一个能力维度上的当前掌握度。"""

    domain_id: str
    dimension: str
    score: float
    attempts_count: int
    updated_at: str


@dataclass(frozen=True)
class StudyRecommendation:
    """PersonalTutorAgent 给出的学习建议。"""

    weak_dimensions: tuple[str, ...]
    weak_tags: tuple[str, ...]
    recommended_paper_ids: tuple[str, ...]
    review_plan: tuple[str, ...]
    next_question_ids: tuple[str, ...]
    summary: str
    generated_at: str


@dataclass(frozen=True)
class AttemptRecord:
    """持久化的答题记录。"""

    attempt_id: str
    domain_id: str
    question_id: str
    user_answer: str
    score: float
    max_score: float
    feedback: str
    mode: str
    knowledge_tags: tuple[str, ...]
    created_at: str
    question_type: str = ""
    prompt: str = ""


@dataclass(frozen=True)
class ResearchRun:
    """一次工作流的可展示执行记录。"""

    run_id: str
    domain_id: str
    task: str
    status: str
    summary: str
    created_at: str
    details: dict[str, Any] = field(default_factory=dict)
