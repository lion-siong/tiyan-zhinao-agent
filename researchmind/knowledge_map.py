"""ResearchMind 五层知识图谱的数据构建与证据归纳。

本模块只把已经入库的论文、原文证据和已有 Agent 结果整理成层级结构，
不负责调用大模型，也不补写不存在的论文、DOI、页码或研究结论。
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Iterable, Mapping

from researchmind.adolescent_topic import (
    CONTENT_BY_KIND,
    CONTENT_ITEMS,
    CORE_RESEARCH_DIRECTION,
    INSUFFICIENT_EVIDENCE_TEXT,
    THEME_BY_ID,
    THEMES,
    TopicThemeDefinition,
    assess_topic_relevance,
)
from researchmind.evidence_quality import evidence_quality_flags
from researchmind.evidence_quality import RESEARCH_QUESTION_BY_ID
from researchmind.models import (
    DebateAnalysis,
    EvidenceRef,
    PaperAnalysis,
    PaperRecord,
    QuestionSynthesis,
    ResearchConflict,
    ResearchSynthesis,
    Viewpoint,
)
from researchmind.research_graph_taxonomy import (
    DIRECTION_BY_ID,
    RESEARCH_DIRECTIONS,
    VariableDefinition,
)


CORE_DIRECTION = CORE_RESEARCH_DIRECTION
CORE_DIRECTION_ID = "core:direction"
KNOWLEDGE_MAP_ENGINE_VERSION = "knowledge-map-v5"
RETRIEVAL_SCOPE_NOTE = (
    "本报告仅归纳当前项目论文库中通过题名、摘要或研究对象原文确认属于"
    "青少年范围，并能定位到全文页码或明确题录摘要的论文。摘要级证据"
    "不会伪装成全文证据。未纳入当前库的研究不等同于整个学术领域不存在"
    "相关研究。"
)


KnowledgeThemeDefinition = TopicThemeDefinition

_TEEN_MARKERS = (
    "青少年",
    "中学生",
    "初中生",
    "高中生",
    "儿童",
    "adolescent",
    "teenager",
    "youth",
    "school-aged",
    "school student",
    "children",
)
_COLLEGE_MARKERS = (
    "大学生",
    "本科生",
    "研究生",
    "高校学生",
    "college student",
    "university student",
    "undergraduate",
    "postgraduate",
    "higher education",
)
_INTERVENTION_MARKERS = (
    "干预",
    "训练",
    "运动处方",
    "随机对照",
    "intervention",
    "training",
    "exercise program",
    "exercise prescription",
    "randomized controlled",
    "randomised controlled",
)
_MEASUREMENT_MARKERS = (
    "测试",
    "测量",
    "指标",
    "问卷",
    "量表",
    "instrument",
    "measure",
    "measurement",
    "questionnaire",
    "test",
    "outcome",
    "index",
)
_METHOD_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "系统综述 / Meta 分析",
        ("systematic review", "meta-analysis", "meta analysis", "系统综述", "元分析"),
    ),
    (
        "随机对照试验",
        ("randomized controlled", "randomised controlled", "随机对照", "rct"),
    ),
    (
        "横断面 / 调查研究",
        ("cross-sectional", "survey", "横断面", "问卷调查"),
    ),
    (
        "纵向 / 队列研究",
        ("longitudinal", "cohort", "follow-up", "纵向", "队列", "追踪"),
    ),
    (
        "准实验 / 对照研究",
        ("quasi-experimental", "controlled trial", "对照组", "准实验"),
    ),
)


@dataclass(frozen=True)
class KnowledgeMapPaper:
    """知识地图中的真实论文及其范围判定。"""

    paper_id: str
    title: str
    authors: tuple[str, ...]
    year: str
    journal: str
    doi: str
    source_url: str
    pdf_path: str
    source: str
    scope_status: str
    scope_reason: str
    theme_ids: tuple[str, ...]
    evidence: tuple[EvidenceRef, ...]
    population: tuple[EvidenceRef, ...] = ()
    methods: tuple[EvidenceRef, ...] = ()
    results: tuple[EvidenceRef, ...] = ()
    conclusions: tuple[EvidenceRef, ...] = ()
    limitations: tuple[EvidenceRef, ...] = ()
    measures: tuple[EvidenceRef, ...] = ()
    interventions: tuple[EvidenceRef, ...] = ()
    comparators: tuple[EvidenceRef, ...] = ()
    statistics: tuple[EvidenceRef, ...] = ()
    abstract: str = ""
    keywords: tuple[str, ...] = ()
    metadata_evidence: EvidenceRef | None = None

    @property
    def authors_text(self) -> str:
        return "、".join(self.authors) if self.authors else "作者待核对"

    @property
    def first_author(self) -> str:
        return self.authors[0] if self.authors else "作者待核对"

    @property
    def is_in_scope(self) -> bool:
        return self.scope_status == "范围内"

    @property
    def analysis_status(self) -> str:
        if self.methods or self.results or self.conclusions:
            return "已完成分析"
        if self.evidence:
            return "已获取全文"
        return "检索到"

    @property
    def conclusion_text(self) -> str:
        return _evidence_preview(self.conclusions)

    @property
    def method_text(self) -> str:
        return _evidence_preview(self.methods)

    @property
    def limitation_text(self) -> str:
        return _evidence_preview(self.limitations)

    @property
    def method_category(self) -> str:
        text = f"{self.title} {self.method_text}".casefold()
        matched = tuple(
            label
            for label, terms in _METHOD_CATEGORIES
            if any(term.casefold() in text for term in terms)
        )
        if matched:
            return "、".join(matched)
        return "其他/方法未明确" if self.methods else "方法证据不足"


@dataclass(frozen=True)
class KnowledgeMapNode:
    """五层知识地图节点。"""

    node_id: str
    label: str
    node_type: str
    level: int
    description: str
    parent_ids: tuple[str, ...]
    paper_ids: tuple[str, ...]
    evidence: tuple[EvidenceRef, ...]
    content_kind: str = ""
    variable_role: str = ""
    direction_ids: tuple[str, ...] = ()

    @property
    def paper_count(self) -> int:
        return len(set(self.paper_ids))

    @property
    def evidence_count(self) -> int:
        return len({item.chunk_id for item in self.evidence})

    @property
    def evidence_types(self) -> tuple[str, ...]:
        return tuple(
            sorted({item.source_type or "全文" for item in self.evidence})
        )


@dataclass(frozen=True)
class KnowledgeMapEdge:
    """知识地图关系；每条关系都携带可定位的原文证据。"""

    edge_id: str
    source_id: str
    target_id: str
    relation: str
    edge_kind: str
    evidence: tuple[EvidenceRef, ...]
    paper_ids: tuple[str, ...]
    relation_status: str = ""
    relation_origin: str = "原文证据"

    @property
    def evidence_count(self) -> int:
        return len({item.chunk_id for item in self.evidence})


@dataclass(frozen=True)
class KnowledgeFinding:
    """图谱总结中的共识、争议、局限、空白或创新候选。"""

    finding_id: str
    category: str
    title: str
    statement: str
    status: str
    evidence: tuple[EvidenceRef, ...]
    paper_ids: tuple[str, ...]
    reliability: str
    limitations: tuple[str, ...] = ()
    analysis_time: str = ""

    @property
    def is_supported(self) -> bool:
        return bool(self.evidence) and self.status != "证据不足"


@dataclass(frozen=True)
class ThemeStatistic:
    """一个研究主题的可追溯统计与证据化摘要。"""

    theme_id: str
    label: str
    paper_count: int
    evidence_count: int
    method_summary: str = INSUFFICIENT_EVIDENCE_TEXT
    population_summary: str = INSUFFICIENT_EVIDENCE_TEXT
    measure_summary: str = INSUFFICIENT_EVIDENCE_TEXT
    conclusion_summary: str = INSUFFICIENT_EVIDENCE_TEXT
    controversy_summary: str = INSUFFICIENT_EVIDENCE_TEXT
    limitation_summary: str = INSUFFICIENT_EVIDENCE_TEXT
    representative_paper_ids: tuple[str, ...] = ()
    source_evidence: tuple[EvidenceRef, ...] = ()
    direction_count: int = 0
    variable_count: int = 0


@dataclass(frozen=True)
class KnowledgeGraphStatistics:
    """知识图谱结构规模的可核验统计。"""

    theme_count: int
    direction_count: int
    variable_count: int
    paper_node_count: int
    hierarchy_relation_count: int
    research_relation_count: int
    paper_variable_relation_count: int
    in_scope_paper_count: int
    pending_paper_count: int
    pending_direction_count: int = 0
    typed_research_relation_count: int = 0
    inconsistent_relation_count: int = 0
    question_variable_relation_count: int = 0
    consensus_node_count: int = 0
    controversy_node_count: int = 0
    research_gap_node_count: int = 0
    synthesis_node_count: int = 0

    @property
    def effective_relation_count(self) -> int:
        return (
            self.hierarchy_relation_count
            + self.research_relation_count
            + self.paper_variable_relation_count
        )


@dataclass(frozen=True)
class KnowledgeMap:
    """完整知识地图及科研总结所需的可追溯数据。"""

    core_direction: str
    domain_name: str
    nodes: tuple[KnowledgeMapNode, ...]
    edges: tuple[KnowledgeMapEdge, ...]
    papers: tuple[KnowledgeMapPaper, ...]
    findings: tuple[KnowledgeFinding, ...]
    theme_statistics: tuple[ThemeStatistic, ...]
    year_distribution: tuple[tuple[str, int], ...]
    method_distribution: tuple[tuple[str, int], ...]
    stance_distribution: tuple[tuple[str, int], ...]
    generated_at: str
    mode: str = "本地证据归纳（不调用大模型）"
    scope_note: str = RETRIEVAL_SCOPE_NOTE
    question_syntheses: tuple[QuestionSynthesis, ...] = ()
    conflicts: tuple[ResearchConflict, ...] = ()
    analysis_time: str = ""
    analysis_engine_version: str = ""
    graph_statistics: KnowledgeGraphStatistics = field(
        default_factory=lambda: KnowledgeGraphStatistics(
            0, 0, 0, 0, 0, 0, 0, 0, 0
        )
    )
    graph_insights: tuple[KnowledgeFinding, ...] = ()

    @property
    def in_scope_papers(self) -> tuple[KnowledgeMapPaper, ...]:
        return tuple(paper for paper in self.papers if paper.is_in_scope)

    @property
    def out_of_scope_papers(self) -> tuple[KnowledgeMapPaper, ...]:
        return tuple(paper for paper in self.papers if not paper.is_in_scope)

    @property
    def default_nodes(self) -> tuple[KnowledgeMapNode, ...]:
        return tuple(node for node in self.nodes if node.level <= 2)

    @property
    def direction_nodes(self) -> tuple[KnowledgeMapNode, ...]:
        return tuple(node for node in self.nodes if node.node_type == "研究方向")

    @property
    def variable_nodes(self) -> tuple[KnowledgeMapNode, ...]:
        return tuple(node for node in self.nodes if node.node_type == "研究变量")

    @property
    def paper_nodes(self) -> tuple[KnowledgeMapNode, ...]:
        return tuple(node for node in self.nodes if node.node_type == "论文")

    @property
    def synthesis_nodes(self) -> tuple[KnowledgeMapNode, ...]:
        return tuple(
            node
            for node in self.nodes
            if node.level == 5
            and node.node_type
            in {"学术共识", "学术争议", "研究空白", "学术发现"}
        )

    @property
    def supported_findings(self) -> tuple[KnowledgeFinding, ...]:
        return tuple(item for item in self.findings if item.is_supported)

    def finding_count(self, category: str, *, supported_only: bool = True) -> int:
        items = (
            self.supported_findings
            if supported_only
            else self.findings
        )
        return sum(item.category == category for item in items)


def build_knowledge_map(
    *,
    domain_name: str,
    papers: Iterable[PaperRecord],
    analyses: Mapping[str, PaperAnalysis | None],
    evidence_by_paper: Mapping[str, Iterable[EvidenceRef]],
    debate: DebateAnalysis | None = None,
    synthesis: ResearchSynthesis | None = None,
) -> KnowledgeMap:
    """从真实论文和原文证据构建五层知识地图。

    默认图只包含核心方向和六个主题。研究内容与论文节点只有在其关系能够
    回指原文证据时才创建，范围外论文不会进入主题分支。
    """

    paper_records = tuple(papers)
    analyses_by_id = dict(analyses)
    evidence_by_id = {
        paper_id: _unique_evidence(items)
        for paper_id, items in evidence_by_paper.items()
    }

    mapped_papers: list[KnowledgeMapPaper] = []
    theme_evidence: dict[str, dict[str, tuple[EvidenceRef, ...]]] = defaultdict(dict)
    for paper in paper_records:
        evidence = evidence_by_id.get(paper.paper_id, ())
        analysis = analyses_by_id.get(paper.paper_id)
        scope_status, scope_reason = _classify_scope(paper, analysis, evidence)
        metadata_evidence = _metadata_evidence(paper)
        mapped_paper = KnowledgeMapPaper(
            paper_id=paper.paper_id,
            title=paper.title,
            authors=tuple(paper.authors),
            year=paper.year,
            journal=paper.journal,
            doi=paper.doi,
            source_url=paper.source_url,
            pdf_path=paper.pdf_path,
            source=paper.source,
            scope_status=scope_status,
            scope_reason=scope_reason,
            theme_ids=(),
            evidence=evidence,
            population=analysis.population if analysis else (),
            methods=analysis.methods if analysis else (),
            results=analysis.results if analysis else (),
            conclusions=analysis.conclusions if analysis else (),
            limitations=analysis.limitations if analysis else (),
            interventions=analysis.interventions if analysis else (),
            comparators=analysis.comparators if analysis else (),
            statistics=analysis.statistics if analysis else (),
            abstract=paper.abstract,
            keywords=tuple(paper.keywords),
            measures=_analysis_evidence_matching(
                analysis,
                CONTENT_BY_KIND["measures"].analysis_fields,
                CONTENT_BY_KIND["measures"].match_terms,
            ),
            metadata_evidence=metadata_evidence,
        )
        matches = (
            _match_themes(paper, evidence, analysis)
            if scope_status == "范围内"
            else {}
        )
        if scope_status == "范围内":
            for theme_id, refs in _match_direction_themes(
                mapped_paper,
                analysis,
            ).items():
                matches.setdefault(theme_id, refs)
        if scope_status == "范围内" and not matches and metadata_evidence is not None:
            matches = _match_metadata_themes(paper, metadata_evidence)
        theme_ids = tuple(theme_id for theme_id in matches)
        for theme_id, refs in matches.items():
            theme_evidence[theme_id][paper.paper_id] = refs
        mapped_papers.append(replace(mapped_paper, theme_ids=theme_ids))

    papers_by_id = {paper.paper_id: paper for paper in mapped_papers}
    in_scope_ids = {
        paper.paper_id for paper in mapped_papers if paper.is_in_scope
    }
    nodes, edges = _build_multilevel_research_graph(
        theme_evidence=theme_evidence,
        papers=papers_by_id,
        analyses=analyses_by_id,
        in_scope_ids=in_scope_ids,
    )
    graph_insights = _build_graph_insights(
        nodes=tuple(nodes),
        papers=papers_by_id,
    )
    theme_direction_counts = Counter(
        node.parent_ids[0]
        for node in nodes
        if node.node_type == "研究方向" and node.parent_ids
    )
    theme_variable_counts = Counter(
        theme_id
        for node in nodes
        if node.node_type == "研究变量"
        for theme_id in node.parent_ids
    )

    findings = list(
        _build_findings(
            analyses=analyses_by_id,
            papers=papers_by_id,
            in_scope_ids=in_scope_ids,
            theme_evidence=theme_evidence,
            debate=debate,
            synthesis=synthesis,
        )
    )
    _append_synthesis_nodes(
        nodes=nodes,
        edges=edges,
        findings=tuple(findings),
        graph_insights=graph_insights,
        papers=papers_by_id,
        core_id=CORE_DIRECTION_ID,
    )
    deduplicated_edges = _deduplicate_edges(edges)
    graph_statistics = _graph_statistics(
        tuple(nodes),
        deduplicated_edges,
        papers_by_id,
    )
    theme_statistics = tuple(
        _theme_statistic(
            theme=theme,
            theme_refs=theme_evidence.get(theme.theme_id, {}),
            papers=papers_by_id,
            debate=_filter_debate(debate, in_scope_ids),
            direction_count=theme_direction_counts.get(theme.theme_id, 0),
            variable_count=theme_variable_counts.get(theme.theme_id, 0),
        )
        for theme in THEMES
    )
    year_distribution = tuple(
        sorted(
            Counter(
                papers_by_id[paper_id].year
                for paper_id in in_scope_ids
                if papers_by_id[paper_id].year
            ).items()
        )
    )
    method_distribution = _method_distribution(
        in_scope_ids,
        analyses_by_id,
    )
    scoped_debate = _filter_debate(debate, in_scope_ids)
    stance_distribution = tuple(
        sorted(
            Counter(
                viewpoint.stance
                for viewpoint in (
                    scoped_debate.viewpoints if scoped_debate else ()
                )
                if viewpoint.evidence
            ).items()
        )
    )
    return KnowledgeMap(
        core_direction=CORE_DIRECTION,
        domain_name=domain_name,
        nodes=tuple(nodes),
        edges=deduplicated_edges,
        papers=tuple(mapped_papers),
        findings=tuple(findings),
        theme_statistics=theme_statistics,
        year_distribution=year_distribution,
        method_distribution=method_distribution,
        stance_distribution=stance_distribution,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        question_syntheses=(
            tuple(debate.question_syntheses) if debate is not None else ()
        ),
        conflicts=tuple(debate.conflicts) if debate is not None else (),
        analysis_time=debate.analysis_time if debate is not None else "",
        analysis_engine_version=(
            debate.engine_version if debate is not None else ""
        ),
        graph_statistics=graph_statistics,
        graph_insights=graph_insights,
    )


def build_markdown_report(knowledge_map: KnowledgeMap) -> str:
    """生成只包含真实入库论文和原文证据的 Markdown 分析报告。"""

    full_text_count = sum(bool(paper.evidence) for paper in knowledge_map.papers)
    abstract_count = sum(
        not paper.evidence
        and any(
            item.source_type == "摘要"
            for item in (
                *paper.population,
                *paper.methods,
                *paper.results,
                *paper.conclusions,
            )
        )
        for paper in knowledge_map.papers
    )
    lines = [
        "# 体研智脑 Agent 知识图谱总结报告",
        "",
        f"- 核心研究方向：{knowledge_map.core_direction}",
        f"- 当前领域：{knowledge_map.domain_name}",
        f"- 范围内论文：{len(knowledge_map.in_scope_papers)} 篇",
        f"- 范围外/待审核论文：{len(knowledge_map.out_of_scope_papers)} 篇",
        f"- 范围内证据：{len({item.chunk_id for paper in knowledge_map.in_scope_papers for item in paper.evidence})} 条",
        f"- 全文证据论文：{full_text_count} 篇",
        f"- 摘要级分析论文：{abstract_count} 篇",
        f"- 知识图谱层：{knowledge_map.graph_statistics.theme_count} 个主题、"
        f"{knowledge_map.graph_statistics.direction_count} 个研究方向、"
        f"{knowledge_map.graph_statistics.pending_direction_count} 个待细分方向、"
        f"{knowledge_map.graph_statistics.variable_count} 个变量、"
        f"{knowledge_map.graph_statistics.paper_node_count} 个论文节点",
        f"- 研究关系：{knowledge_map.graph_statistics.research_relation_count} 条，"
        f"其中有明确结果方向的关系 "
        f"{knowledge_map.graph_statistics.typed_research_relation_count} 条",
        f"- 分析时间：{knowledge_map.analysis_time or '未记录'}",
        f"- 分析引擎：{knowledge_map.analysis_engine_version or '未记录'}",
        f"- 生成方式：{knowledge_map.mode}",
        "",
        "> 检索范围说明：",
        f"> {knowledge_map.scope_note}",
        "",
        "## 研究主题分布",
        "",
        "| 研究主题 | 范围内论文 | 可追溯证据 |",
        "| --- | ---: | ---: |",
    ]
    for item in knowledge_map.theme_statistics:
        lines.append(f"| {item.label} | {item.paper_count} | {item.evidence_count} |")

    lines.extend(
        [
            "",
            "## 研究方向结构",
            "",
            "| 研究方向 | 所属主题 | 论文数 | 证据数 |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    theme_label_by_id = {theme.theme_id: theme.label for theme in THEMES}
    for node in sorted(
        knowledge_map.direction_nodes,
        key=lambda item: (item.parent_ids[0] if item.parent_ids else "", item.label),
    ):
        theme_label = theme_label_by_id.get(
            node.parent_ids[0] if node.parent_ids else "",
            "未记录",
        )
        lines.append(
            f"| {node.label} | {theme_label} | {node.paper_count} | "
            f"{node.evidence_count} |"
        )

    lines.extend(
        [
            "",
            "## 研究变量节点",
            "",
            "| 研究变量 | 变量角色 | 相关论文 | 证据数 |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for node in sorted(knowledge_map.variable_nodes, key=lambda item: item.label):
        lines.append(
            f"| {node.label} | {node.variable_role or '未记录'} | "
            f"{node.paper_count} | {node.evidence_count} |"
        )

    lines.extend(
        [
            "",
            "## 有证据的研究变量关系",
            "",
            "| 变量关系 | 结果状态 | 论文数 | 证据数 |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    variable_label_by_id = {
        node.node_id: node.label for node in knowledge_map.variable_nodes
    }
    for edge in knowledge_map.edges:
        if edge.edge_kind != "research_relation":
            continue
        source = variable_label_by_id.get(edge.source_id, edge.source_id)
        target = variable_label_by_id.get(edge.target_id, edge.target_id)
        lines.append(
            f"| {source} → {target} | {edge.relation_status or '证据不足'} | "
            f"{len(set(edge.paper_ids))} | {edge.evidence_count} |"
        )

    lines.extend(["", "## 图谱科研洞察", ""])
    for insight in knowledge_map.graph_insights:
        lines.extend(
            [
                f"### {insight.category}：{insight.title}",
                "",
                insight.statement,
                "",
                f"- 状态：{insight.status}",
                f"- 可靠性：{insight.reliability}",
            ]
        )
        for evidence in insight.evidence[:2]:
            lines.append(
                f"- 原文证据：{evidence.document_title}，"
                f"{evidence.location_label}：{evidence.quote}"
            )
        lines.append("")

    lines.extend(
        [
            "",
            "## 六大研究主题证据摘要",
            "",
            "> 以下摘要仅摘录当前库内已定位到页码的原文。"
            "没有对应证据时明确标记为证据不足。",
        ]
    )
    papers_by_id = {
        paper.paper_id: paper for paper in knowledge_map.papers
    }
    for item in knowledge_map.theme_statistics:
        lines.extend(
            [
                "",
                f"### {item.label}",
                "",
                f"- 相关论文数量：{item.paper_count}",
                f"- 可追溯证据：{item.evidence_count} 条",
                f"- 主要研究方法：{item.method_summary or INSUFFICIENT_EVIDENCE_TEXT}",
                f"- 主要研究对象：{item.population_summary or INSUFFICIENT_EVIDENCE_TEXT}",
                f"- 常用评价指标：{item.measure_summary or INSUFFICIENT_EVIDENCE_TEXT}",
                f"- 代表性研究结论：{item.conclusion_summary or INSUFFICIENT_EVIDENCE_TEXT}",
                f"- 相关学术争议：{item.controversy_summary or INSUFFICIENT_EVIDENCE_TEXT}",
                f"- 研究不足：{item.limitation_summary or INSUFFICIENT_EVIDENCE_TEXT}",
            ]
        )
        representative_papers = tuple(
            papers_by_id[paper_id]
            for paper_id in item.representative_paper_ids
            if paper_id in papers_by_id
        )
        if representative_papers:
            lines.append("- 代表性论文：")
            for paper in representative_papers:
                lines.append(
                    f"  - {paper.title}（{paper.first_author}，"
                    f"{paper.year or '年份待核对'}；"
                    f"DOI：{paper.doi or '未提供'}）"
                )
        else:
            lines.append("- 代表性论文：证据不足，暂不能判断")

        if item.source_evidence:
            lines.append("- 可追溯证据：")
            for evidence in item.source_evidence[:6]:
                lines.append(
                    f"  - {evidence.document_title}，"
                    f"{evidence.location_label}，"
                    f"{evidence.file_name}：{evidence.quote}"
                )
        else:
            lines.append("- 可追溯证据：证据不足，暂不能判断")

    if knowledge_map.question_syntheses:
        lines.extend(
            [
                "",
                "## 按研究问题的证据综合",
                "",
                "> 证据确定性为保守的自动初筛，不是正式 GRADE 专家评价。",
                "",
            ]
        )
        for synthesis in knowledge_map.question_syntheses:
            lines.extend(
                [
                    f"### {synthesis.label}",
                    "",
                    f"- 研究问题：{synthesis.question}",
                    f"- 状态：{synthesis.status}",
                    f"- 综合结论：{synthesis.conclusion}",
                    f"- 支持论文数：{len(synthesis.supporting_paper_ids)}",
                    f"- 明确负向结果论文数：{len(synthesis.opposing_paper_ids)}",
                    f"- 方向无法判定论文数：{len(synthesis.inconclusive_paper_ids)}",
                    f"- 证据确定性：{synthesis.evidence_certainty}",
                    f"- 判定依据：{synthesis.certainty_reason}",
                ]
            )
            related_ids = tuple(
                dict.fromkeys(
                    (
                        *synthesis.supporting_paper_ids,
                        *synthesis.opposing_paper_ids,
                        *synthesis.inconclusive_paper_ids,
                    )
                )
            )
            for paper_id in related_ids:
                paper = papers_by_id.get(paper_id)
                if paper is not None:
                    lines.append(
                        f"- 代表论文：{paper.title}（{paper.year or '年份待核对'}；"
                        f"DOI：{paper.doi or '未提供'}）"
                    )
            for evidence in synthesis.representative_evidence:
                lines.append(
                    f"- 原文证据：{evidence.document_title}，"
                    f"{evidence.location_label}：{evidence.quote}"
                )
            lines.append("")

    lines.extend(["", "## 范围内论文", ""])
    if knowledge_map.in_scope_papers:
        for paper in knowledge_map.in_scope_papers:
            themes = "、".join(
                THEME_BY_ID[theme_id].label
                for theme_id in paper.theme_ids
                if theme_id in THEME_BY_ID
            )
            lines.extend(
                [
                    f"### {paper.title}",
                    "",
                    f"- 作者：{paper.authors_text}",
                    f"- 年份：{paper.year or '未提供'}",
                    f"- 期刊：{paper.journal or '未提供'}",
                    f"- DOI：{paper.doi or '未提供'}",
                    f"- 来源：{paper.source or '未提供'}",
                    f"- 本地 PDF：{paper.pdf_path or '未提供'}",
                    f"- 主题：{themes or '未分类'}",
                    "",
                ]
            )
    else:
        lines.append(
            "当前论文库没有通过青少年研究对象范围核验的论文。"
            "该结果只表示当前库覆盖不足。"
        )

    for category in (
        "学术共识",
        "学术共识线索",
        "学术争议",
        "研究局限",
        "研究空白",
        "创新方向",
    ):
        items = tuple(
            item for item in knowledge_map.findings if item.category == category
        )
        if not items:
            continue
        lines.extend(["", f"## {category}", ""])
        for item in items:
            lines.extend(
                [
                    f"### {item.title}",
                    "",
                    item.statement,
                    "",
                    f"- 证据状态：{item.status}",
                    f"- 可靠性：{item.reliability}",
                ]
            )
            for evidence in item.evidence:
                lines.append(
                    f"- 原文证据：{evidence.document_title}，"
                    f"{evidence.location_label}，"
                    f"{evidence.file_name}：{evidence.quote}"
                )
            if not item.evidence:
                lines.append("- 证据状态：证据不足，待进一步验证")
            lines.append("")

    lines.extend(["", "## 范围外/待审核论文", ""])
    if knowledge_map.out_of_scope_papers:
        lines.extend(
            [
                "| 论文 | 年份 | DOI | 判定理由 |",
                "| --- | --- | --- | --- |",
            ]
        )
        for paper in knowledge_map.out_of_scope_papers:
            title = paper.title.replace("|", "\\|")
            reason = paper.scope_reason.replace("|", "\\|")
            lines.append(
                f"| {title} | {paper.year} | {paper.doi or '未提供'} | {reason} |"
            )
    else:
        lines.append("没有范围外论文。")
    lines.append("")
    return "\n".join(lines)


def _build_multilevel_research_graph(
    *,
    theme_evidence: Mapping[str, Mapping[str, tuple[EvidenceRef, ...]]],
    papers: Mapping[str, KnowledgeMapPaper],
    analyses: Mapping[str, PaperAnalysis | None],
    in_scope_ids: set[str],
) -> tuple[list[KnowledgeMapNode], list[KnowledgeMapEdge]]:
    """构建领域—主题—问题/变量—论文证据—学术综合五层图。"""

    nodes: list[KnowledgeMapNode] = [
        KnowledgeMapNode(
            node_id=CORE_DIRECTION_ID,
            label=CORE_DIRECTION,
            node_type="核心方向",
            level=1,
            description="当前知识地图唯一的核心研究方向。",
            parent_ids=(),
            paper_ids=tuple(sorted(in_scope_ids)),
            evidence=_evidence_for_papers(in_scope_ids, theme_evidence),
        )
    ]
    edges: list[KnowledgeMapEdge] = []
    direction_nodes: dict[str, KnowledgeMapNode] = {}
    direction_refs: dict[str, dict[str, tuple[EvidenceRef, ...]]] = defaultdict(dict)
    variable_evidence: dict[str, list[EvidenceRef]] = defaultdict(list)
    variable_papers: dict[str, set[str]] = defaultdict(set)
    variable_directions: dict[str, set[str]] = defaultdict(set)
    variable_roles: dict[str, set[str]] = defaultdict(set)
    variable_mentions_by_paper: dict[
        str,
        dict[str, tuple[EvidenceRef, ...]],
    ] = defaultdict(dict)
    paper_direction_ids: dict[str, set[str]] = defaultdict(set)
    analysis_evidence_by_paper = {
        paper_id: _paper_analysis_evidence(
            paper,
            analyses.get(paper_id),
        )
        for paper_id, paper in papers.items()
    }

    for theme in THEMES:
        paper_refs = theme_evidence.get(theme.theme_id, {})
        theme_paper_ids = tuple(paper_refs)
        theme_evidence_items = _flatten(paper_refs.values())
        nodes.append(
            KnowledgeMapNode(
                node_id=theme.theme_id,
                label=theme.label,
                node_type="研究主题",
                level=2,
                description=theme.description,
                parent_ids=(CORE_DIRECTION_ID,),
                paper_ids=theme_paper_ids,
                evidence=theme_evidence_items,
            )
        )
        if theme_evidence_items and theme_paper_ids:
            edges.append(
                KnowledgeMapEdge(
                    edge_id=f"edge:{CORE_DIRECTION_ID}:{theme.theme_id}",
                    source_id=CORE_DIRECTION_ID,
                    target_id=theme.theme_id,
                    relation="核心方向包含研究主题",
                    edge_kind="hierarchy",
                    evidence=theme_evidence_items,
                    paper_ids=theme_paper_ids,
                )
            )
        if not theme_paper_ids:
            continue

        for direction in (
            item for item in RESEARCH_DIRECTIONS if item.theme_id == theme.theme_id
        ):
            matched: dict[str, tuple[EvidenceRef, ...]] = {}
            for paper_id in theme_paper_ids:
                refs = _direction_evidence_for_paper(
                    papers[paper_id],
                    analyses.get(paper_id),
                    direction.terms,
                    evidence_pool=analysis_evidence_by_paper[paper_id],
                )
                if refs:
                    matched[paper_id] = refs
            if not matched:
                continue
            direction_paper_ids = tuple(matched)
            direction_evidence_items = _flatten(matched.values())
            direction_node = KnowledgeMapNode(
                node_id=direction.direction_id,
                label=direction.label,
                node_type="研究方向",
                level=3,
                description=direction.description,
                parent_ids=(theme.theme_id,),
                paper_ids=direction_paper_ids,
                evidence=direction_evidence_items,
                content_kind="research_direction",
            )
            direction_nodes[direction.direction_id] = direction_node
            direction_refs[direction.direction_id] = matched
            nodes.append(direction_node)
            edges.append(
                KnowledgeMapEdge(
                    edge_id=f"edge:{theme.theme_id}:{direction.direction_id}",
                    source_id=theme.theme_id,
                    target_id=direction.direction_id,
                    relation="主题细分研究方向",
                    edge_kind="hierarchy",
                    evidence=direction_evidence_items,
                    paper_ids=direction_paper_ids,
                )
            )
            for paper_id in direction_paper_ids:
                paper_direction_ids[paper_id].add(direction.direction_id)
            for variable in direction.variables:
                variable_id = _graph_variable_id(variable.label)
                for paper_id in direction_paper_ids:
                    refs = _variable_evidence_for_paper(
                        papers[paper_id],
                        analyses.get(paper_id),
                        variable,
                    )
                    if not refs:
                        continue
                    variable_evidence[variable_id].extend(refs)
                    variable_papers[variable_id].add(paper_id)
                    variable_directions[variable_id].add(direction.direction_id)
                    variable_roles[variable_id].add(variable.role)
                    existing = variable_mentions_by_paper[paper_id].get(
                        variable_id,
                        (),
                    )
                    variable_mentions_by_paper[paper_id][variable_id] = (
                        _unique_evidence((*existing, *refs))
                    )

        unmatched_paper_ids = tuple(
            paper_id
            for paper_id in theme_paper_ids
            if paper_id not in paper_direction_ids
        )
        if unmatched_paper_ids:
            pending_id = f"direction:pending:{theme.theme_id.split(':', 1)[-1]}"
            pending_refs = {
                paper_id: paper_refs[paper_id]
                for paper_id in unmatched_paper_ids
            }
            pending_evidence = _flatten(pending_refs.values())
            direction_refs[pending_id] = pending_refs
            nodes.append(
                KnowledgeMapNode(
                    node_id=pending_id,
                    label="待细分研究",
                    node_type="待细分方向",
                    level=3,
                    description=(
                        f"{len(unmatched_paper_ids)} 篇论文已确认属于本主题，"
                        "但当前可定位文本未命中明确研究方向词。未强行归类到"
                        "具体方向。"
                    ),
                    parent_ids=(theme.theme_id,),
                    paper_ids=unmatched_paper_ids,
                    evidence=pending_evidence,
                    content_kind="pending_direction",
                )
            )
            edges.append(
                KnowledgeMapEdge(
                    edge_id=f"edge:{theme.theme_id}:{pending_id}",
                    source_id=theme.theme_id,
                    target_id=pending_id,
                    relation="主题内待细分论文",
                    edge_kind="hierarchy",
                    evidence=pending_evidence,
                    paper_ids=unmatched_paper_ids,
                )
            )
            for paper_id in unmatched_paper_ids:
                paper_direction_ids[paper_id].add(pending_id)

    variable_by_id = {
        _graph_variable_id(variable.label): variable
        for direction in RESEARCH_DIRECTIONS
        for variable in direction.variables
    }
    for variable_id in sorted(variable_evidence, key=lambda item: variable_by_id[item].label):
        variable = variable_by_id[variable_id]
        evidence_items = _unique_evidence(variable_evidence[variable_id])
        if not evidence_items:
            continue
        parent_direction_ids = tuple(sorted(variable_directions[variable_id]))
        parent_theme_ids = tuple(
            sorted(
                {
                    direction_nodes[direction_id].parent_ids[0]
                    for direction_id in parent_direction_ids
                    if direction_id in direction_nodes
                    and direction_nodes[direction_id].parent_ids
                }
            )
        )
        variable_node = KnowledgeMapNode(
            node_id=variable_id,
            label=variable.label,
            node_type="研究变量",
            level=3,
            description=(
                f"{len(variable_papers[variable_id])} 篇论文在 "
                f"{len(parent_direction_ids)} 个研究方向提供该变量的原文证据；"
                f"变量角色：{'、'.join(sorted(variable_roles[variable_id]))}。"
            ),
            parent_ids=parent_theme_ids,
            paper_ids=tuple(sorted(variable_papers[variable_id])),
            evidence=evidence_items,
            content_kind="variable",
            variable_role="、".join(sorted(variable_roles[variable_id])),
            direction_ids=parent_direction_ids,
        )
        nodes.append(variable_node)
        for direction_id in parent_direction_ids:
            direction = direction_nodes.get(direction_id)
            if direction is None:
                continue
            direction_evidence_for_variable = tuple(
                item
                for paper_id in variable_papers[variable_id]
                if paper_id in direction.paper_ids
                for item in variable_mentions_by_paper[paper_id].get(
                    variable_id,
                    (),
                )
            )
            if not direction_evidence_for_variable:
                continue
            edges.append(
                KnowledgeMapEdge(
                    edge_id=f"edge:{direction_id}:{variable_id}",
                    source_id=direction_id,
                    target_id=variable_id,
                    relation="研究方向涉及变量",
                    edge_kind="question_variable",
                    evidence=_unique_evidence(direction_evidence_for_variable),
                    paper_ids=tuple(
                        paper_id
                        for paper_id in variable_papers[variable_id]
                        if paper_id in direction.paper_ids
                    ),
                )
            )

    for paper_id in sorted(paper_direction_ids):
        paper = papers[paper_id]
        paper_node_id = f"paper:{paper_id}"
        parent_direction_ids = tuple(sorted(paper_direction_ids[paper_id]))
        nodes.append(
            KnowledgeMapNode(
                node_id=paper_node_id,
                label=paper.title,
                node_type="论文",
                level=4,
                description=(
                    f"{paper.authors_text}，{paper.year or '年份待核对'}，"
                    f"{paper.journal or '期刊待核对'}。"
                    f"DOI：{paper.doi or '未提供'}"
                ),
                parent_ids=parent_direction_ids,
                paper_ids=(paper_id,),
                evidence=_paper_analysis_evidence(
                    paper,
                    analyses.get(paper_id),
                ),
                content_kind="paper",
                direction_ids=parent_direction_ids,
            )
        )
        for direction_id in parent_direction_ids:
            evidence_items = direction_refs[direction_id].get(paper_id, ())
            if not evidence_items:
                continue
            edges.append(
                KnowledgeMapEdge(
                    edge_id=f"edge:{direction_id}:{paper_node_id}",
                    source_id=direction_id,
                    target_id=paper_node_id,
                    relation="论文归属研究方向",
                    edge_kind="hierarchy",
                    evidence=evidence_items,
                    paper_ids=(paper_id,),
                )
            )
        for variable_id, evidence_items in variable_mentions_by_paper[
            paper_id
        ].items():
            if not evidence_items:
                continue
            edges.append(
                KnowledgeMapEdge(
                    edge_id=f"edge:{paper_node_id}:{variable_id}",
                    source_id=paper_node_id,
                    target_id=variable_id,
                    relation="论文提供变量原文证据",
                    edge_kind="paper_variable",
                    evidence=evidence_items,
                    paper_ids=(paper_id,),
                    relation_status="变量原文",
                    relation_origin="论文结构化阅读证据",
                )
            )

    edges.extend(
        _build_research_relation_edges(
            papers=papers,
            analyses=analyses,
            variable_mentions_by_paper=variable_mentions_by_paper,
            variable_by_id=variable_by_id,
        )
    )
    return nodes, edges


def _direction_evidence_for_paper(
    paper: KnowledgeMapPaper,
    analysis: PaperAnalysis | None,
    terms: Iterable[str],
    *,
    evidence_pool: tuple[EvidenceRef, ...] | None = None,
) -> tuple[EvidenceRef, ...]:
    """返回能够支持研究方向归属的论文原文或摘要证据。"""

    pool = evidence_pool or _paper_analysis_evidence(paper, analysis)
    direct = tuple(
        item for item in pool if _contains_any(item.quote, terms)
    )
    if direct:
        return _unique_evidence(direct)[:4]
    metadata = " ".join(
        (paper.title, paper.abstract, *paper.keywords)
    )
    if not _contains_any(metadata, terms):
        return ()
    abstract_refs = tuple(
        item for item in pool if item.source_type == "摘要"
    )
    if abstract_refs:
        return abstract_refs[:2]
    structured = tuple(
        item
        for item in (
            *paper.conclusions,
            *paper.results,
            *paper.methods,
        )
        if item
    )
    return _unique_evidence(structured)[:2]


def _variable_evidence_for_paper(
    paper: KnowledgeMapPaper,
    analysis: PaperAnalysis | None,
    variable: VariableDefinition,
) -> tuple[EvidenceRef, ...]:
    """仅在论文已有字段或证据文本明确出现变量时返回证据。"""

    if analysis is None:
        return ()
    field_names = {
        "exposure": (
            "methods",
            "interventions",
            "comparators",
            "measures",
            "results",
            "conclusions",
        ),
        "outcome": (
            "population",
            "measures",
            "results",
            "statistics",
            "conclusions",
        ),
        "mediator": (
            "methods",
            "results",
            "statistics",
            "conclusions",
        ),
        "moderator": (
            "population",
            "methods",
            "results",
            "conclusions",
        ),
    }.get(variable.role_kind, ("methods", "results", "conclusions"))
    matched = _analysis_evidence_matching(
        analysis,
        field_names,
        variable.terms,
    )
    if matched:
        return matched[:4]
    return ()


def _paper_analysis_evidence(
    paper: KnowledgeMapPaper,
    analysis: PaperAnalysis | None,
) -> tuple[EvidenceRef, ...]:
    if analysis is None:
        return _unique_evidence(
            (
                *paper.evidence,
                *((paper.metadata_evidence,) if paper.metadata_evidence else ()),
            )
        )
    return _unique_evidence(
        (
            *analysis.all_evidence,
            *((paper.metadata_evidence,) if paper.metadata_evidence else ()),
            *(
                item
                for item in paper.evidence
                if evidence_quality_flags(
                    item.quote,
                    section_hint=item.section_hint,
                    source_type=item.source_type,
                )[0]
            ),
        )
    )


def _build_research_relation_edges(
    *,
    papers: Mapping[str, KnowledgeMapPaper],
    analyses: Mapping[str, PaperAnalysis | None],
    variable_mentions_by_paper: Mapping[
        str,
        Mapping[str, tuple[EvidenceRef, ...]],
    ],
    variable_by_id: Mapping[str, VariableDefinition],
) -> tuple[KnowledgeMapEdge, ...]:
    """逐篇检查变量共现和结果方向，再聚合同一变量关系。"""

    aggregate: dict[
        tuple[str, str],
        dict[str, object],
    ] = {}
    for paper_id, mentions in variable_mentions_by_paper.items():
        paper = papers.get(paper_id)
        analysis = analyses.get(paper_id)
        if paper is None or analysis is None:
            continue
        source_ids = tuple(
            variable_id
            for variable_id in mentions
            if variable_by_id[variable_id].role_kind
            in {"exposure", "mediator", "moderator"}
        )
        target_ids = tuple(
            variable_id
            for variable_id in mentions
            if variable_by_id[variable_id].role_kind == "outcome"
        )
        results = _unique_evidence((*analysis.results, *analysis.conclusions))
        if not results:
            results = tuple(
                item
                for item in analysis.all_evidence
                if item.source_type == "摘要"
            )
        if not results:
            continue
        context_evidence = _unique_evidence(
            (
                *results,
                *analysis.methods,
                *analysis.interventions,
                *analysis.comparators,
                *analysis.measures,
                *paper.population,
            )
        )
        relation_count = 0
        for source_id in source_ids:
            source = variable_by_id[source_id]
            for target_id in target_ids:
                if source_id == target_id:
                    continue
                target = variable_by_id[target_id]
                relation_results = tuple(
                    item
                    for item in results
                    if _contains_any(item.quote, target.terms)
                )
                if not relation_results:
                    continue
                source_supported = any(
                    _contains_any(item.quote, source.terms)
                    for item in context_evidence
                )
                if not source_supported:
                    continue
                direct_relation_results = tuple(
                    item
                    for item in relation_results
                    if _contains_any(item.quote, source.terms)
                )
                source_context = tuple(
                    item
                    for item in context_evidence
                    if _contains_any(item.quote, source.terms)
                )[:1]
                evidence_items = _unique_evidence(
                    (
                        *(direct_relation_results or relation_results)[:2],
                        *source_context,
                    )
                )
                status = (
                    _classify_research_relation(
                        " ".join(item.quote for item in direct_relation_results),
                        target_label=target.label,
                    )
                    if direct_relation_results
                    else "证据不足"
                )
                key = (source_id, target_id)
                item = aggregate.setdefault(
                    key,
                    {
                        "statuses": [],
                        "evidence": [],
                        "paper_ids": set(),
                    },
                )
                item["statuses"].append(status)
                item["evidence"].extend(evidence_items)
                item["paper_ids"].add(paper_id)
                relation_count += 1
                if relation_count >= 12:
                    break
            if relation_count >= 12:
                break

    relations: list[KnowledgeMapEdge] = []
    for (source_id, target_id), item in sorted(aggregate.items()):
        evidence_items = _unique_evidence(item["evidence"])
        paper_ids = tuple(sorted(item["paper_ids"]))
        if not evidence_items or not paper_ids:
            continue
        status = _aggregate_relation_status(item["statuses"])
        source = variable_by_id[source_id]
        target = variable_by_id[target_id]
        relations.append(
            KnowledgeMapEdge(
                edge_id=f"edge:research:{source_id}:{target_id}",
                source_id=source_id,
                target_id=target_id,
                relation=(
                    f"{source.label} → {target.label}：{status}"
                    "（相关证据，不自动解释为因果关系）"
                ),
                edge_kind="research_relation",
                evidence=evidence_items,
                paper_ids=paper_ids,
                relation_status=status,
                relation_origin="逐篇原文结果方向归纳",
            )
        )
    return tuple(relations)


def _classify_research_relation(
    text: str,
    *,
    target_label: str = "",
) -> str:
    lowered = text.casefold()
    conditional = (
        "depending on",
        "moderated",
        "only among",
        "condition",
        "仅在",
        "取决于",
        "条件下",
    )
    positive = (
        "positive association",
        "positive association",
        "associated with higher",
        "associated with greater",
        "improved",
        "increased",
        "beneficial",
        "改善",
        "提高",
        "正向关联",
        "正相关",
    )
    negative = (
        "negative association",
        "associated with lower",
        "decreased",
        "reduced",
        "worse",
        "降低",
        "下降",
        "负向关联",
        "负相关",
    )
    null = (
        "no significant",
        "not significant",
        "no association",
        "no difference",
        "无显著",
        "不显著",
        "无关联",
    )
    has_positive = any(term in lowered for term in positive)
    has_negative = any(term in lowered for term in negative)
    reduction_benefit = any(
        term in target_label
        for term in ("体脂", "BMI", "肥胖", "抑郁", "焦虑", "久坐", "损伤")
    )
    if reduction_benefit and any(
        term in lowered
        for term in (
            "reduced",
            "decreased",
            "lower",
            "improved",
            "缓解",
            "降低",
            "减少",
            "改善",
        )
    ):
        has_positive = True
    if reduction_benefit and any(
        term in lowered
        for term in (
            "increased body fat",
            "higher body fat",
            "increased bmi",
            "higher bmi",
            "worsened",
            "加重",
            "升高",
        )
    ):
        has_negative = True
    has_null = any(term in lowered for term in null)
    if any(term in lowered for term in conditional):
        return "存在条件性关系"
    if sum((has_positive, has_negative, has_null)) >= 2:
        return "研究结果不一致"
    if has_null:
        return "无显著关联"
    if has_negative:
        return "负向关联"
    if has_positive:
        return "正向关联"
    return "证据不足"


def _aggregate_relation_status(statuses: Iterable[str]) -> str:
    values = tuple(statuses)
    if "存在条件性关系" in values:
        return "存在条件性关系"
    meaningful = {
        item
        for item in values
        if item in {"正向关联", "负向关联", "无显著关联"}
    }
    if len(meaningful) > 1:
        return "研究结果不一致"
    if meaningful:
        return next(iter(meaningful))
    return "证据不足"


def _graph_variable_id(label: str) -> str:
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:12]
    return f"variable:{digest}"


def _append_synthesis_nodes(
    *,
    nodes: list[KnowledgeMapNode],
    edges: list[KnowledgeMapEdge],
    findings: tuple[KnowledgeFinding, ...],
    graph_insights: tuple[KnowledgeFinding, ...],
    papers: Mapping[str, KnowledgeMapPaper],
    core_id: str,
) -> None:
    """把共识、争议和覆盖不足追加为第五层可追溯节点。"""

    accepted_categories = {
        "学术共识",
        "学术共识线索",
        "学术争议",
        "研究空白",
    }
    candidates = [
        item for item in findings if item.category in accepted_categories
    ] + [
        item
        for item in graph_insights
        if item.category == "研究空白"
    ]
    seen: set[str] = set()
    node_by_id = {node.node_id: node for node in nodes}
    direction_nodes = tuple(
        node for node in nodes if node.node_type == "研究方向"
    )
    theme_nodes = tuple(
        node for node in nodes if node.node_type == "研究主题"
    )
    paper_nodes = {
        paper_id: node
        for node in nodes
        if node.node_type == "论文"
        for paper_id in node.paper_ids
    }
    for finding in candidates:
        if finding.finding_id in seen:
            continue
        seen.add(finding.finding_id)
        node_type = {
            "学术共识": "学术共识",
            "学术共识线索": "学术共识",
            "学术争议": "学术争议",
            "研究空白": "研究空白",
        }[finding.category]
        related_directions = sorted(
            (
                node
                for node in direction_nodes
                if set(node.paper_ids) & set(finding.paper_ids)
            ),
            key=lambda node: (
                -len(set(node.paper_ids) & set(finding.paper_ids)),
                node.label,
            ),
        )[:3]
        if not related_directions and node_type == "研究空白":
            direction_label = finding.title.split("：", 1)[0]
            related_directions = sorted(
                (
                    node
                    for node in direction_nodes
                    if direction_label in node.label
                    or node.label in direction_label
                ),
                key=lambda node: node.label,
            )[:3]
        parent_ids = [node.node_id for node in related_directions]
        if not parent_ids and finding.paper_ids:
            paper_theme_ids = {
                theme_id
                for paper_id in finding.paper_ids
                if paper_id in papers
                for theme_id in papers[paper_id].theme_ids
            }
            parent_ids = [
                node.node_id
                for node in theme_nodes
                if node.node_id in paper_theme_ids
            ]
        if not parent_ids:
            parent_ids = [core_id]

        context_evidence: list[EvidenceRef] = list(finding.evidence)
        if not context_evidence:
            for parent_id in parent_ids:
                parent_node = node_by_id.get(parent_id)
                if parent_node is not None:
                    context_evidence.extend(parent_node.evidence[:2])
        evidence_items = _unique_evidence(context_evidence)[:6]
        if not evidence_items:
            # 没有可定位证据时不生成确定性第五层节点。
            continue
        synthesis_id = f"synthesis:{finding.finding_id}"
        if synthesis_id in node_by_id:
            continue
        synthesis_node = KnowledgeMapNode(
            node_id=synthesis_id,
            label=finding.title,
            node_type=node_type,
            level=5,
            description=(
                f"{finding.statement} "
                f"状态：{finding.status}；可靠性：{finding.reliability}。"
            ),
            parent_ids=tuple(dict.fromkeys(parent_ids)),
            paper_ids=finding.paper_ids,
            evidence=evidence_items,
            content_kind="synthesis",
            variable_role=finding.status,
        )
        nodes.append(synthesis_node)
        node_by_id[synthesis_id] = synthesis_node
        for parent_id in synthesis_node.parent_ids:
            edges.append(
                KnowledgeMapEdge(
                    edge_id=f"edge:{parent_id}:{synthesis_id}",
                    source_id=parent_id,
                    target_id=synthesis_id,
                    relation=f"形成{node_type}节点",
                    edge_kind="synthesis_relation",
                    evidence=evidence_items,
                    paper_ids=finding.paper_ids,
                    relation_status=finding.status,
                    relation_origin="学术综合与覆盖分析",
                )
            )
        for paper_id in finding.paper_ids:
            paper_node = paper_nodes.get(paper_id)
            if paper_node is None:
                continue
            paper_evidence = tuple(
                item
                for item in finding.evidence
                if item.document_id == paper_id
            ) or paper_node.evidence[:2]
            if not paper_evidence:
                continue
            edges.append(
                KnowledgeMapEdge(
                    edge_id=f"edge:{paper_node.node_id}:{synthesis_id}",
                    source_id=paper_node.node_id,
                    target_id=synthesis_id,
                    relation="论文证据参与学术综合",
                    edge_kind="evidence_synthesis",
                    evidence=_unique_evidence(paper_evidence),
                    paper_ids=(paper_id,),
                    relation_status=finding.status,
                    relation_origin="论文证据与综合结论关联",
                )
            )


def _graph_statistics(
    nodes: tuple[KnowledgeMapNode, ...],
    edges: tuple[KnowledgeMapEdge, ...],
    papers: Mapping[str, KnowledgeMapPaper],
) -> KnowledgeGraphStatistics:
    graph_edges = tuple(
        edge for edge in edges if edge.evidence and edge.paper_ids
    )
    return KnowledgeGraphStatistics(
        theme_count=sum(node.node_type == "研究主题" for node in nodes),
        direction_count=sum(node.node_type == "研究方向" for node in nodes),
        variable_count=sum(node.node_type == "研究变量" for node in nodes),
        paper_node_count=sum(node.node_type == "论文" for node in nodes),
        hierarchy_relation_count=sum(
            edge.edge_kind == "hierarchy" for edge in graph_edges
        ),
        research_relation_count=sum(
            edge.edge_kind == "research_relation" for edge in graph_edges
        ),
        paper_variable_relation_count=sum(
            edge.edge_kind == "paper_variable" for edge in graph_edges
        ),
        in_scope_paper_count=sum(paper.is_in_scope for paper in papers.values()),
        pending_paper_count=sum(
            paper.scope_status == "待审核" for paper in papers.values()
        ),
        pending_direction_count=sum(
            node.node_type == "待细分方向" for node in nodes
        ),
        typed_research_relation_count=sum(
            edge.edge_kind == "research_relation"
            and edge.relation_status
            in {
                "正向关联",
                "负向关联",
                "无显著关联",
                "存在条件性关系",
                "研究结果不一致",
            }
            for edge in graph_edges
        ),
        inconsistent_relation_count=sum(
            edge.edge_kind == "research_relation"
            and edge.relation_status == "研究结果不一致"
            for edge in graph_edges
        ),
        question_variable_relation_count=sum(
            edge.edge_kind == "question_variable" for edge in graph_edges
        ),
        consensus_node_count=sum(
            node.node_type == "学术共识" for node in nodes
        ),
        controversy_node_count=sum(
            node.node_type == "学术争议" for node in nodes
        ),
        research_gap_node_count=sum(
            node.node_type == "研究空白" for node in nodes
        ),
        synthesis_node_count=sum(
            node.level == 5
            and node.node_type
            in {"学术共识", "学术争议", "研究空白", "学术发现"}
            for node in nodes
        ),
    )


def _build_graph_insights(
    *,
    nodes: tuple[KnowledgeMapNode, ...],
    papers: Mapping[str, KnowledgeMapPaper],
) -> tuple[KnowledgeFinding, ...]:
    """从当前图谱节点和真实论文年份生成保守科研洞察。"""

    directions = tuple(
        node for node in nodes if node.node_type == "研究方向"
    )
    themes = tuple(node for node in nodes if node.node_type == "研究主题")
    actual_direction_ids = {node.node_id for node in directions}
    insights: list[KnowledgeFinding] = []

    def direction_recent_count(node: KnowledgeMapNode) -> int:
        return sum(
            (papers[paper_id].year or "0") >= "2024"
            for paper_id in node.paper_ids
            if paper_id in papers
        )

    hotspots = sorted(
        directions,
        key=lambda node: (
            -node.paper_count,
            -direction_recent_count(node),
            -node.evidence_count,
            node.label,
        ),
    )[:5]
    for node in hotspots:
        if node.paper_count < 1:
            continue
        insights.append(
            KnowledgeFinding(
                finding_id=f"finding:hotspot:{node.node_id}",
                category="研究热点",
                title=node.label,
                statement=(
                    f"当前库有 {node.paper_count} 篇论文、"
                    f"{node.evidence_count} 条可定位证据，"
                    f"其中 {direction_recent_count(node)} 篇发表于 2024—2026 年。"
                    "热点由当前库覆盖统计得到，不代表领域总体研究热度。"
                ),
                status="当前库活跃方向",
                evidence=node.evidence,
                paper_ids=node.paper_ids,
                reliability="论文数和证据量自动统计，非质量评价",
                limitations=("受当前 39 篇论文库的检索范围限制。",),
            )
        )

    emerging = sorted(
        (
            node
            for node in directions
            if node.paper_count >= 2
            and direction_recent_count(node) / node.paper_count >= 0.6
        ),
        key=lambda node: (
            -direction_recent_count(node) / node.paper_count,
            -node.paper_count,
            node.label,
        ),
    )[:5]
    for node in emerging:
        recent = direction_recent_count(node)
        insights.append(
            KnowledgeFinding(
                finding_id=f"finding:emerging:{node.node_id}",
                category="新兴方向",
                title=node.label,
                statement=(
                    f"该方向 {node.paper_count} 篇论文中有 {recent} 篇发表于"
                    "2024—2026 年，近期占比较高。该判断仅描述当前库的"
                    "新增集中度，不能单独证明领域正在形成新范式。"
                ),
                status="近期论文集中",
                evidence=tuple(
                    item
                    for item in node.evidence
                    if papers.get(item.document_id)
                    and (papers[item.document_id].year or "0") >= "2024"
                ),
                paper_ids=tuple(
                    paper_id
                    for paper_id in node.paper_ids
                    if paper_id in papers
                    and (papers[paper_id].year or "0") >= "2024"
                ),
                reliability="基于年份和当前库论文数，非趋势因果判断",
            )
        )

    shared_items: list[tuple[int, KnowledgeMapNode, KnowledgeMapNode, tuple[str, ...]]] = []
    for index, left in enumerate(directions):
        for right in directions[index + 1 :]:
            shared = tuple(sorted(set(left.paper_ids) & set(right.paper_ids)))
            if len(shared) >= 2:
                shared_items.append((len(shared), left, right, shared))
    for count, left, right, shared in sorted(
        shared_items,
        key=lambda item: (-item[0], item[1].label, item[2].label),
    )[:5]:
        evidence = tuple(
            item
            for item in (*left.evidence, *right.evidence)
            if item.document_id in shared
        )
        insights.append(
            KnowledgeFinding(
                finding_id=(
                    f"finding:intersection:{left.node_id}:{right.node_id}"
                ),
                category="研究交叉",
                title=f"{left.label} × {right.label}",
                statement=(
                    f"两个研究方向共享 {count} 篇论文，说明当前库存在可比较的"
                    "交叉证据。共享论文不等于两个方向具有相同结论。"
                ),
                status="共享论文证据",
                evidence=_unique_evidence(evidence),
                paper_ids=shared,
                reliability="基于共同论文的图谱结构关系",
            )
        )

    for direction in RESEARCH_DIRECTIONS:
        if direction.direction_id in actual_direction_ids:
            continue
        theme = THEME_BY_ID.get(direction.theme_id)
        insights.append(
            KnowledgeFinding(
                finding_id=f"finding:coverage:{direction.direction_id}",
                category="研究空白",
                title=f"{direction.label}：当前库覆盖不足",
                statement=(
                    f"当前库未在“{theme.label if theme else direction.theme_id}”"
                    "下定位到该方向的可核验论文文本。该结果仅表示当前库覆盖"
                    "缺口，不能据此断言整个学术领域没有相关研究。"
                ),
                status="当前库覆盖不足",
                evidence=(),
                paper_ids=(),
                reliability="覆盖缺口，不是领域空白结论",
                limitations=("当前库 39 篇论文及检索范围有限。",),
            )
        )

    for node in sorted(
        (item for item in directions if item.paper_count < 3),
        key=lambda item: (item.paper_count, item.label),
    ):
        insights.append(
            KnowledgeFinding(
                finding_id=f"finding:limited:{node.node_id}",
                category="研究空白",
                title=f"{node.label}：当前库直接证据仅 {node.paper_count} 篇",
                statement=(
                    f"该方向当前只有 {node.paper_count} 篇范围内论文和 "
                    f"{node.evidence_count} 条可定位证据，暂不足以判断结果"
                    "是否稳定。论文少可能是当前检索范围造成的覆盖缺口，"
                    "不能直接宣布为学术空白。"
                ),
                status="当前库覆盖较少",
                evidence=node.evidence,
                paper_ids=node.paper_ids,
                reliability="低覆盖统计，不是领域空白结论",
                limitations=("需要扩大检索范围并人工核验。",),
            )
        )

    year_theme_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for theme in themes:
        for paper_id in theme.paper_ids:
            paper = papers.get(paper_id)
            if paper and paper.year:
                year_theme_counts[paper.year][theme.label] += 1
    for year in sorted(year_theme_counts):
        theme_label, count = year_theme_counts[year].most_common(1)[0]
        theme_node = next(
            (node for node in themes if node.label == theme_label),
            None,
        )
        evidence = tuple(
            item
            for item in (theme_node.evidence if theme_node else ())
            if papers.get(item.document_id)
            and papers[item.document_id].year == year
        )
        insights.append(
            KnowledgeFinding(
                finding_id=f"finding:trend:{year}",
                category="研究趋势",
                title=f"{year} 年当前库主要主题",
                statement=(
                    f"{year} 年在当前库中论文数最多的主题为“{theme_label}”"
                    f"（{count} 篇）。这是当前库的年度构成，不是领域总体趋势。"
                ),
                status="年度结构统计",
                evidence=_unique_evidence(evidence),
                paper_ids=tuple(
                    paper_id
                    for paper_id in (theme_node.paper_ids if theme_node else ())
                    if papers.get(paper_id)
                    and papers[paper_id].year == year
                ),
                reliability="基于论文年份与主题归属",
            )
        )

    density = sorted(
        directions,
        key=lambda node: (
            -(node.evidence_count / max(1, node.paper_count)),
            -node.paper_count,
            node.label,
        ),
    )[:5]
    for node in density:
        if node.paper_count == 0:
            continue
        ratio = node.evidence_count / node.paper_count
        insights.append(
            KnowledgeFinding(
                finding_id=f"finding:density:{node.node_id}",
                category="证据密度",
                title=node.label,
                statement=(
                    f"平均每篇论文有 {ratio:.1f} 条可定位证据进入该方向。"
                    "证据密度只反映当前解析覆盖，不代表研究质量或效应大小。"
                ),
                status="证据覆盖统计",
                evidence=node.evidence,
                paper_ids=node.paper_ids,
                reliability="证据条数/论文数自动计算",
            )
        )
    return tuple(insights)


def _classify_scope(
    paper: PaperRecord,
    analysis: PaperAnalysis | None,
    evidence: tuple[EvidenceRef, ...],
) -> tuple[str, str]:
    population = analysis.population if analysis else ()
    population_text = _join_evidence(population)
    relevance = assess_topic_relevance(
        title=paper.title,
        abstract=paper.abstract,
        keywords=paper.keywords,
        population_text=population_text,
        full_text=_join_evidence(evidence),
    )
    status = {
        "相关": "范围内",
        "待审核": "待审核",
    }.get(relevance.status, "范围外")
    return status, "；".join(relevance.reasons) or INSUFFICIENT_EVIDENCE_TEXT


def _match_themes(
    paper: PaperRecord,
    evidence: tuple[EvidenceRef, ...],
    analysis: PaperAnalysis | None,
) -> dict[str, tuple[EvidenceRef, ...]]:
    metadata = _join_paper_metadata(paper)
    matches: dict[str, tuple[EvidenceRef, ...]] = {}
    for theme in THEMES:
        direct_refs = tuple(
            item
            for item in evidence
            if _contains_any(item.quote, theme.match_terms)
        )
        metadata_match = _contains_any(metadata, theme.match_terms)
        if direct_refs:
            matches[theme.theme_id] = _unique_evidence(direct_refs)
            continue
        if not metadata_match or analysis is None:
            continue
        # 题名或摘要完成主题归类后，仍使用论文自身的结构化原文作为证据，
        # 不根据关键词相似性虚构论文之间的关系。
        structured_refs = _analysis_evidence_matching(
            analysis,
            ("conclusions", "results", "methods", "population"),
            (),
        )
        if structured_refs:
            matches[theme.theme_id] = structured_refs
    return matches


def _match_direction_themes(
    paper: KnowledgeMapPaper,
    analysis: PaperAnalysis | None,
) -> dict[str, tuple[EvidenceRef, ...]]:
    """题名或文本命中具体研究方向时，用方向所属主题做保守归类。"""

    matches: dict[str, tuple[EvidenceRef, ...]] = {}
    evidence_pool = _paper_analysis_evidence(paper, analysis)
    for direction in RESEARCH_DIRECTIONS:
        refs = _direction_evidence_for_paper(
            paper,
            analysis,
            direction.terms,
            evidence_pool=evidence_pool,
        )
        if refs:
            matches.setdefault(direction.theme_id, refs)
    return {
        theme_id: _unique_evidence(refs)
        for theme_id, refs in matches.items()
    }


def _match_metadata_themes(
    paper: PaperRecord,
    metadata_evidence: EvidenceRef,
) -> dict[str, tuple[EvidenceRef, ...]]:
    """最后仅在题名、关键词或摘要明确命中主题词时使用题录证据。"""

    text = " ".join((paper.title, paper.abstract, *paper.keywords))
    return {
        theme.theme_id: (metadata_evidence,)
        for theme in THEMES
        if _contains_any(text, theme.match_terms)
    }


def _metadata_evidence(paper: PaperRecord) -> EvidenceRef | None:
    title = paper.title.strip()
    if not title:
        return None
    keyword_text = "、".join(
        keyword for keyword in paper.keywords if keyword.strip()
    )
    quote = f"题名：{title}"
    if keyword_text:
        quote += f"；关键词：{keyword_text}"
    return EvidenceRef(
        chunk_id=f"metadata:{paper.paper_id}",
        document_id=paper.paper_id,
        document_title=paper.title,
        file_name="论文题录",
        page_number=1,
        quote=quote,
        source_type="元数据",
        section_hint="元数据",
    )


def _build_content_nodes(
    *,
    theme: KnowledgeThemeDefinition,
    paper_ids: tuple[str, ...],
    analyses: Mapping[str, PaperAnalysis | None],
    debate: DebateAnalysis | None,
    scope_ids: set[str],
) -> tuple[KnowledgeMapNode, ...]:
    result: list[KnowledgeMapNode] = []
    for content in CONTENT_ITEMS:
        if content.kind == "controversies":
            continue
        refs_by_paper: dict[str, list[EvidenceRef]] = defaultdict(list)
        for paper_id in paper_ids:
            analysis = analyses.get(paper_id)
            if analysis is None:
                continue
            refs_by_paper[paper_id].extend(
                _analysis_evidence_matching(
                    analysis,
                    content.analysis_fields,
                    content.match_terms,
                )
            )
        node = _make_content_node(
            theme_id=theme.theme_id,
            kind=content.kind,
            label=content.label,
            refs_by_paper=refs_by_paper,
        )
        if node is not None:
            result.append(node)

    controversy = _controversy_node(
        theme_id=theme.theme_id,
        paper_ids=tuple(paper_id for paper_id in paper_ids if paper_id in scope_ids),
        debate=debate,
    )
    if controversy is not None:
        result.append(controversy)
    return tuple(result)


def _make_content_node(
    *,
    theme_id: str,
    kind: str,
    label: str,
    refs_by_paper: Mapping[str, Iterable[EvidenceRef]],
) -> KnowledgeMapNode | None:
    paper_ids = tuple(paper_id for paper_id, refs in refs_by_paper.items() if refs)
    evidence = _flatten(refs_by_paper.values())
    if not paper_ids or not evidence:
        return None
    node_id = f"content:{theme_id.split(':', 1)[-1]}:{kind}"
    return KnowledgeMapNode(
        node_id=node_id,
        label=label,
        node_type="研究内容",
        level=3,
        description=f"{len(paper_ids)} 篇论文提供可定位到页码的{label}证据。",
        parent_ids=(theme_id,),
        paper_ids=paper_ids,
        evidence=evidence,
        content_kind=kind,
    )


def _controversy_node(
    *,
    theme_id: str,
    paper_ids: tuple[str, ...],
    debate: DebateAnalysis | None,
) -> KnowledgeMapNode | None:
    if debate is None or not _has_controversy(debate):
        return None
    refs_by_paper: dict[str, list[EvidenceRef]] = defaultdict(list)
    related_claims: list[str] = []
    for viewpoint in debate.viewpoints:
        for evidence in viewpoint.evidence:
            if evidence.document_id not in paper_ids:
                continue
            refs_by_paper[evidence.document_id].append(evidence)
            if viewpoint.claim not in related_claims:
                related_claims.append(viewpoint.claim)
    if not refs_by_paper:
        return None
    return KnowledgeMapNode(
        node_id=f"content:{theme_id.split(':', 1)[-1]}:controversies",
        label="学术争议",
        node_type="研究内容",
        level=3,
        description=(
            f"当前证据涉及 {len(refs_by_paper)} 篇论文的不同观点："
            + "；".join(related_claims[:3])
        ),
        parent_ids=(theme_id,),
        paper_ids=tuple(refs_by_paper),
        evidence=_flatten(refs_by_paper.values()),
        content_kind="controversies",
    )


def _build_theme_relations(
    theme_evidence: Mapping[str, Mapping[str, tuple[EvidenceRef, ...]]],
) -> tuple[KnowledgeMapEdge, ...]:
    edges: list[KnowledgeMapEdge] = []
    for left_index, left in enumerate(THEMES):
        left_papers = set(theme_evidence.get(left.theme_id, {}))
        for right in THEMES[left_index + 1 :]:
            right_papers = set(theme_evidence.get(right.theme_id, {}))
            shared = tuple(sorted(left_papers & right_papers))
            if not shared:
                continue
            evidence = _flatten(
                (
                    theme_evidence[left.theme_id][paper_id]
                    for paper_id in shared
                )
            )
            edges.append(
                KnowledgeMapEdge(
                    edge_id=f"edge:theme:{left.theme_id}:{right.theme_id}",
                    source_id=left.theme_id,
                    target_id=right.theme_id,
                    relation=f"共享 {len(shared)} 篇范围匹配论文",
                    edge_kind="theme_relation",
                    evidence=evidence,
                    paper_ids=shared,
                )
            )
    return tuple(edges)


def _build_paper_relations(
    nodes: list[KnowledgeMapNode],
    edges: list[KnowledgeMapEdge],
    papers_by_id: Mapping[str, KnowledgeMapPaper],
) -> tuple[KnowledgeMapEdge, ...]:
    content_by_paper: dict[str, dict[str, KnowledgeMapNode]] = defaultdict(dict)
    for node in nodes:
        if node.node_type != "研究内容":
            continue
        for paper_id in node.paper_ids:
            content_by_paper[paper_id][node.content_kind] = node
    paper_ids = sorted(content_by_paper)
    relations: list[KnowledgeMapEdge] = []
    for left_index, left_id in enumerate(paper_ids):
        for right_id in paper_ids[left_index + 1 :]:
            shared_kinds = (
                set(content_by_paper[left_id])
                & set(content_by_paper[right_id])
            )
            if len(shared_kinds) < 2:
                continue
            if left_id not in papers_by_id or right_id not in papers_by_id:
                continue
            evidence: list[EvidenceRef] = []
            for kind in sorted(shared_kinds):
                evidence.extend(
                    item
                    for item in content_by_paper[left_id][kind].evidence
                    if item.document_id == left_id
                )
                evidence.extend(
                    item
                    for item in content_by_paper[right_id][kind].evidence
                    if item.document_id == right_id
                )
            relations.append(
                KnowledgeMapEdge(
                    edge_id=f"edge:paper:{left_id}:{right_id}",
                    source_id=f"paper:{left_id}",
                    target_id=f"paper:{right_id}",
                    relation=(
                        "可比较的共享研究内容："
                        + "、".join(_content_label(kind) for kind in sorted(shared_kinds))
                    ),
                    edge_kind="paper_relation",
                    evidence=_unique_evidence(evidence),
                    paper_ids=(left_id, right_id),
                )
            )
    return tuple(relations[:30])


def _build_findings(
    *,
    analyses: Mapping[str, PaperAnalysis | None],
    papers: Mapping[str, KnowledgeMapPaper],
    in_scope_ids: set[str],
    theme_evidence: Mapping[str, Mapping[str, tuple[EvidenceRef, ...]]],
    debate: DebateAnalysis | None,
    synthesis: ResearchSynthesis | None,
) -> tuple[KnowledgeFinding, ...]:
    findings: list[KnowledgeFinding] = []
    scoped_debate = _filter_debate(debate, in_scope_ids)
    consensus_items = _consensus_findings(scoped_debate)
    if consensus_items:
        findings.extend(consensus_items)
    else:
        findings.append(
            KnowledgeFinding(
                finding_id="finding:consensus:insufficient",
                category="学术共识",
                title="尚无可确认的学术共识",
                statement=(
                    "当前论文库范围内没有形成可核验的同题结果综合。"
                    "页面中的主题级证据不足状态不是对研究内容作肯定结论。"
                ),
                status="证据不足",
                evidence=(),
                paper_ids=(),
                reliability="证据不足",
            )
        )

    controversy = _controversy_findings(scoped_debate)
    if controversy:
        findings.extend(controversy)
    else:
        findings.append(
            KnowledgeFinding(
                finding_id="finding:controversy:insufficient",
                category="学术争议",
                title="尚未识别出可验证的相反结论",
                statement=(
                    "当前论文库没有在同一研究问题、同一结局范围内形成"
                    "证据充分的相反结果对照。"
                    "这不表示该领域不存在争议。"
                ),
                status="证据不足",
                evidence=(),
                paper_ids=(),
                reliability="证据不足",
            )
        )

    for paper_id in sorted(in_scope_ids):
        analysis = analyses.get(paper_id)
        paper = papers[paper_id]
        if analysis is None:
            continue
        for index, evidence in enumerate(analysis.limitations, start=1):
            findings.append(
                KnowledgeFinding(
                    finding_id=f"finding:limitation:{paper_id}:{index}",
                    category="研究局限",
                    title=f"{paper.title}：研究局限原文",
                    statement=evidence.quote,
                    status="有原文证据",
                    evidence=(evidence,),
                    paper_ids=(paper_id,),
                    reliability="仅反映该论文明确陈述的局限",
                )
            )

    for theme in THEMES:
        refs_by_paper = theme_evidence.get(theme.theme_id, {})
        count = len(refs_by_paper)
        evidence = _flatten(refs_by_paper.values())
        if count == 0:
            findings.append(
                KnowledgeFinding(
                    finding_id=f"finding:gap:{theme.theme_id}",
                    category="研究空白",
                    title=f"{theme.label}：当前库覆盖不足",
                    statement=(
                        "在本次检索范围与当前论文库中，未发现同时满足青少年"
                        "研究对象条件且可定位原文证据的论文。不能据此断定整个"
                        "学术领域没有相关研究，需要扩大公开数据库检索与人工审核。"
                    ),
                    status="证据不足",
                    evidence=(),
                    paper_ids=(),
                    reliability="覆盖缺口，不是领域空白结论",
                    limitations=("当前库检索范围有限。",),
                )
            )
        elif count < 3:
            findings.append(
                KnowledgeFinding(
                    finding_id=f"finding:gap:{theme.theme_id}:limited",
                    category="研究空白",
                    title=f"{theme.label}：直接证据仅 {count} 篇",
                    statement=(
                        f"当前库在该主题仅纳入 {count} 篇范围匹配论文，"
                        "不足以判断研究结论是否稳定，也不足以推断整个人群。"
                    ),
                    status="待补充证据",
                    evidence=evidence,
                    paper_ids=tuple(refs_by_paper),
                    reliability="低至中等，受当前库覆盖限制",
                    limitations=("样本论文数量不足以进行稳定综合。",),
                )
            )

    supported_gaps = tuple(
        item
        for item in findings
        if item.category == "研究空白" and item.is_supported
    )
    for item in supported_gaps[:4]:
        findings.append(
            KnowledgeFinding(
                finding_id=f"finding:innovation:{item.finding_id}",
                category="创新方向",
                title=f"围绕“{item.title.split('：', 1)[0]}”补充验证",
                statement=(
                    "可优先补充更大年龄跨度、明确运动剂量、长期追踪和可重复"
                    "测量方案的研究设计。具体假设必须在新增证据后形成，当前"
                    "不宣称已有研究能够支持因果结论。"
                ),
                status="候选方向，待验证",
                evidence=item.evidence,
                paper_ids=item.paper_ids,
                reliability="基于当前库覆盖缺口提出",
                limitations=item.limitations,
            )
        )

    if synthesis is not None and synthesis.research_gaps:
        findings.append(
            KnowledgeFinding(
                finding_id="finding:synthesis:gaps",
                category="研究空白",
                title="既有知识综合结果的范围提示",
                statement=(
                    "；".join(synthesis.research_gaps[:3])
                    + "。该结果来自现有 Agent 综合记录，仍需回到原文证据核验。"
                ),
                status="待人工核验",
                evidence=tuple(
                    item
                    for paper_id in in_scope_ids
                    for item in (
                        analyses[paper_id].limitations
                        if analyses.get(paper_id)
                        else ()
                    )[:1]
                ),
                paper_ids=tuple(sorted(in_scope_ids)),
                reliability="现有综合记录，需人工核验",
            )
        )
    return tuple(findings)


def _consensus_findings(
    debate: DebateAnalysis | None,
) -> tuple[KnowledgeFinding, ...]:
    if debate is None:
        return ()
    if debate.question_syntheses:
        findings: list[KnowledgeFinding] = []
        for synthesis in debate.question_syntheses:
            if synthesis.status not in {
                "初步共识",
                "有一致研究发现",
            }:
                continue
            if (
                not synthesis.supporting_paper_ids
                or not synthesis.representative_evidence
            ):
                continue
            findings.append(
                KnowledgeFinding(
                    finding_id=f"finding:consensus:{synthesis.question_id}",
                    category="学术共识",
                    title=synthesis.label,
                    statement=f"{synthesis.question} {synthesis.conclusion}",
                    status=synthesis.status,
                    evidence=synthesis.representative_evidence,
                    paper_ids=synthesis.supporting_paper_ids,
                    reliability=(
                        f"{synthesis.evidence_certainty}；"
                        f"{synthesis.certainty_reason}"
                    ),
                    limitations=synthesis.limitations,
                    analysis_time=synthesis.analysis_time,
                )
            )
        return tuple(findings)
    findings: list[KnowledgeFinding] = []
    for index, viewpoint in enumerate(debate.viewpoints, start=1):
        paper_ids = tuple(dict.fromkeys(viewpoint.supporting_papers))
        if len(paper_ids) < 2 or not viewpoint.evidence:
            continue
        findings.append(
            KnowledgeFinding(
                finding_id=f"finding:consensus:{index}",
                category="学术共识线索",
                title="多篇论文呈现一致方向（待系统综述核验）",
                statement=viewpoint.claim,
                status="待系统综述核验",
                evidence=viewpoint.evidence,
                paper_ids=paper_ids,
                reliability=debate.evidence_strength,
                limitations=debate.limitations,
            )
        )
    return tuple(findings)


def _controversy_findings(
    debate: DebateAnalysis | None,
) -> tuple[KnowledgeFinding, ...]:
    if debate is None or not _has_controversy(debate):
        return ()
    if debate.conflicts:
        return tuple(
            KnowledgeFinding(
                finding_id=f"finding:controversy:{conflict.conflict_id}",
                category="学术争议",
                title=RESEARCH_QUESTION_BY_ID.get(
                    conflict.question_id,
                    conflict.question,
                ).label
                if conflict.question_id in RESEARCH_QUESTION_BY_ID
                else conflict.question,
                statement=(
                    f"{conflict.question} "
                    f"观点A：{conflict.viewpoint_a} "
                    f"观点B：{conflict.viewpoint_b} "
                    f"{conflict.current_conclusion}"
                ),
                status="同一研究问题内发现结果方向差异",
                evidence=tuple((*conflict.evidence_a, *conflict.evidence_b)),
                paper_ids=tuple((*conflict.paper_ids_a, *conflict.paper_ids_b)),
                reliability=conflict.evidence_certainty,
                limitations=tuple(
                    (*conflict.divergence_reasons, *conflict.unresolved_questions)
                ),
                analysis_time=conflict.analysis_time,
            )
            for conflict in debate.conflicts
        )
    scoped_viewpoints = tuple(
        viewpoint
        for viewpoint in debate.viewpoints
        if viewpoint.supporting_papers and viewpoint.evidence
    )
    if len(scoped_viewpoints) < 2:
        return ()
    statements = "；".join(
        f"观点{index}（{viewpoint.stance}）：{viewpoint.claim}"
        for index, viewpoint in enumerate(scoped_viewpoints, start=1)
    )
    evidence = tuple(
        item
        for viewpoint in scoped_viewpoints
        for item in viewpoint.evidence
    )
    paper_ids = tuple(
        dict.fromkeys(
            paper_id
            for viewpoint in scoped_viewpoints
            for paper_id in viewpoint.supporting_papers
        )
    )
    return (
        KnowledgeFinding(
            finding_id="finding:controversy:evidence",
            category="学术争议",
            title=debate.topic,
            statement=statements,
            status="发现方向差异，尚不能判定谁对谁错",
            evidence=evidence,
            paper_ids=paper_ids,
            reliability=debate.evidence_strength,
            limitations=tuple(debate.possible_reasons) + tuple(debate.limitations),
        ),
    )


def _filter_debate(
    debate: DebateAnalysis | None,
    in_scope_ids: set[str],
) -> DebateAnalysis | None:
    if debate is None:
        return None
    viewpoints: list[Viewpoint] = []
    for viewpoint in debate.viewpoints:
        evidence = tuple(
            item for item in viewpoint.evidence if item.document_id in in_scope_ids
        )
        support = tuple(
            paper_id
            for paper_id in viewpoint.supporting_papers
            if paper_id in in_scope_ids
        )
        if not evidence or not support:
            continue
        viewpoints.append(
            Viewpoint(
                claim=viewpoint.claim,
                stance=viewpoint.stance,
                supporting_papers=support,
                evidence=evidence,
                claim_origin=viewpoint.claim_origin,
                stance_origin=viewpoint.stance_origin,
            )
        )
    question_syntheses: list[QuestionSynthesis] = []
    for synthesis in debate.question_syntheses:
        supporting = tuple(
            paper_id
            for paper_id in synthesis.supporting_paper_ids
            if paper_id in in_scope_ids
        )
        opposing = tuple(
            paper_id
            for paper_id in synthesis.opposing_paper_ids
            if paper_id in in_scope_ids
        )
        inconclusive = tuple(
            paper_id
            for paper_id in synthesis.inconclusive_paper_ids
            if paper_id in in_scope_ids
        )
        evidence = tuple(
            item
            for item in synthesis.representative_evidence
            if item.document_id in in_scope_ids
        )
        if not (supporting or opposing or inconclusive or evidence):
            continue
        question_syntheses.append(
            QuestionSynthesis(
                question_id=synthesis.question_id,
                label=synthesis.label,
                question=synthesis.question,
                conclusion=synthesis.conclusion,
                status=synthesis.status,
                evidence_certainty=synthesis.evidence_certainty,
                certainty_reason=synthesis.certainty_reason,
                supporting_paper_ids=supporting,
                opposing_paper_ids=opposing,
                inconclusive_paper_ids=inconclusive,
                representative_evidence=evidence,
                outcome_measures=synthesis.outcome_measures,
                limitations=synthesis.limitations,
                analysis_time=synthesis.analysis_time,
            )
        )
    conflicts: list[ResearchConflict] = []
    for conflict in debate.conflicts:
        paper_ids_a = tuple(
            item for item in conflict.paper_ids_a if item in in_scope_ids
        )
        paper_ids_b = tuple(
            item for item in conflict.paper_ids_b if item in in_scope_ids
        )
        evidence_a = tuple(
            item for item in conflict.evidence_a if item.document_id in in_scope_ids
        )
        evidence_b = tuple(
            item for item in conflict.evidence_b if item.document_id in in_scope_ids
        )
        if not (paper_ids_a and paper_ids_b and evidence_a and evidence_b):
            continue
        conflicts.append(
            ResearchConflict(
                conflict_id=conflict.conflict_id,
                question_id=conflict.question_id,
                question=conflict.question,
                viewpoint_a=conflict.viewpoint_a,
                paper_ids_a=paper_ids_a,
                evidence_a=evidence_a,
                viewpoint_b=conflict.viewpoint_b,
                paper_ids_b=paper_ids_b,
                evidence_b=evidence_b,
                divergence_reasons=conflict.divergence_reasons,
                current_conclusion=conflict.current_conclusion,
                unresolved_questions=conflict.unresolved_questions,
                evidence_certainty=conflict.evidence_certainty,
                analysis_time=conflict.analysis_time,
            )
        )
    if not viewpoints and not question_syntheses and not conflicts:
        return None
    return DebateAnalysis(
        topic=debate.topic,
        viewpoints=tuple(viewpoints),
        possible_reasons=debate.possible_reasons,
        limitations=debate.limitations,
        evidence_strength=debate.evidence_strength,
        research_gaps=debate.research_gaps,
        consensus_summary=debate.consensus_summary,
        disagreement_summary=debate.disagreement_summary,
        analysis_mode=debate.analysis_mode,
        considered_paper_ids=tuple(
            paper_id
            for paper_id in debate.considered_paper_ids
            if paper_id in in_scope_ids
        ),
        evidence_covered_paper_ids=tuple(
            paper_id
            for paper_id in debate.evidence_covered_paper_ids
            if paper_id in in_scope_ids
        ),
        coverage_note=debate.coverage_note,
        question_syntheses=tuple(question_syntheses),
        conflicts=tuple(conflicts),
        analysis_time=debate.analysis_time,
        engine_version=debate.engine_version,
    )


def _has_controversy(debate: DebateAnalysis) -> bool:
    if debate.conflicts:
        return True
    if debate.question_syntheses:
        # 新版问题级综合已明确区分正向、负向和未达统计显著。
        # 没有 conflicts 时不得再从跨主题观点中反推争议。
        return False
    meaningful_stances = {
        viewpoint.stance
        for viewpoint in debate.viewpoints
        if viewpoint.stance
        in {"支持", "不支持", "负向结果"}
    }
    return "支持" in meaningful_stances and bool(
        meaningful_stances & {"不支持", "负向结果"}
    )


def _method_distribution(
    in_scope_ids: set[str],
    analyses: Mapping[str, PaperAnalysis | None],
) -> tuple[tuple[str, int], ...]:
    counts = Counter()
    for paper_id in in_scope_ids:
        analysis = analyses.get(paper_id)
        if analysis is None:
            counts["尚未结构化阅读"] += 1
            continue
        method_text = _join_evidence(analysis.methods).casefold()
        matched = [
            label
            for label, terms in _METHOD_CATEGORIES
            if any(term.casefold() in method_text for term in terms)
        ]
        if not matched:
            counts[
                "有方法原文，但未命中预设方法类别"
                if analysis.methods
                else "方法证据不足"
            ] += 1
            continue
        for label in matched:
            counts[label] += 1
    return tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _analysis_evidence_matching(
    analysis: PaperAnalysis | None,
    field_names: Iterable[str],
    match_terms: Iterable[str],
) -> tuple[EvidenceRef, ...]:
    """从结构化阅读结果中选取满足文字条件的原文证据。"""

    if analysis is None:
        return ()
    terms = tuple(match_terms)
    evidence: list[EvidenceRef] = []
    for field_name in field_names:
        for item in analysis.field_evidence(field_name):
            if terms and not _contains_any(item.quote, terms):
                continue
            evidence.append(item)
    return _unique_evidence(evidence)


def _theme_statistic(
    *,
    theme: TopicThemeDefinition,
    theme_refs: Mapping[str, tuple[EvidenceRef, ...]],
    papers: Mapping[str, KnowledgeMapPaper],
    debate: DebateAnalysis | None,
    direction_count: int = 0,
    variable_count: int = 0,
) -> ThemeStatistic:
    """由主题下真实论文的分析字段生成可回溯摘要。"""

    theme_paper_ids = tuple(theme_refs)
    theme_papers = tuple(
        papers[paper_id]
        for paper_id in theme_paper_ids
        if paper_id in papers
    )
    method_evidence = _flatten(paper.methods for paper in theme_papers)
    population_evidence = _flatten(paper.population for paper in theme_papers)
    measure_evidence = _flatten(paper.measures for paper in theme_papers)
    conclusion_evidence = _flatten(paper.conclusions for paper in theme_papers)
    limitation_evidence = _flatten(paper.limitations for paper in theme_papers)

    controversy_evidence: tuple[EvidenceRef, ...] = ()
    controversy_claims: list[str] = []
    if debate is not None and _has_controversy(debate):
        selected_ids = set(theme_paper_ids)
        controversy_evidence = _unique_evidence(
            item
            for viewpoint in debate.viewpoints
            if any(
                paper_id in selected_ids
                for paper_id in viewpoint.supporting_papers
            )
            for item in viewpoint.evidence
            if item.document_id in selected_ids
        )
        if controversy_evidence:
            controversy_claims = [
                viewpoint.claim
                for viewpoint in debate.viewpoints
                if viewpoint.supporting_papers
                and viewpoint.evidence
                and any(
                    paper_id in selected_ids
                    for paper_id in viewpoint.supporting_papers
                )
            ]

    representative_ids = tuple(
        paper.paper_id
        for paper in sorted(
            theme_papers,
            key=lambda item: (
                -len(item.conclusions),
                -len(item.methods),
                -len(item.evidence),
                item.title,
            ),
        )[:3]
    )
    source_evidence = _flatten(
        (
            theme_refs.get(paper_id, ())
            for paper_id in theme_paper_ids
        )
    ) + method_evidence + population_evidence + measure_evidence
    source_evidence += conclusion_evidence + limitation_evidence + controversy_evidence

    controversy_summary = INSUFFICIENT_EVIDENCE_TEXT
    if controversy_evidence:
        controversy_summary = "；".join(controversy_claims[:2])
        controversy_summary = (
            f"当前库存在方向差异，尚不能判定：{controversy_summary}"
        )

    return ThemeStatistic(
        theme_id=theme.theme_id,
        label=theme.label,
        paper_count=len(theme_refs),
        evidence_count=len(
            {
                item.chunk_id
                for refs in theme_refs.values()
                for item in refs
            }
        ),
        method_summary=_evidence_preview(method_evidence),
        population_summary=_evidence_preview(population_evidence),
        measure_summary=_evidence_preview(measure_evidence),
        conclusion_summary=_evidence_preview(conclusion_evidence),
        controversy_summary=controversy_summary,
        limitation_summary=_evidence_preview(limitation_evidence),
        representative_paper_ids=representative_ids,
        source_evidence=_unique_evidence(source_evidence),
        direction_count=direction_count,
        variable_count=variable_count,
    )


def _evidence_preview(
    evidence: Iterable[EvidenceRef],
    *,
    limit: int = 2,
    max_chars: int = 180,
) -> str:
    """把真实原文证据整理为短摘要，不添加证据之外的结论。"""

    snippets: list[str] = []
    for item in _unique_evidence(evidence):
        quote = " ".join(item.quote.split())
        if len(quote) > max_chars:
            quote = quote[:max_chars].rstrip() + "…"
        snippets.append(f"{quote}（第 {item.page_number} 页）")
        if len(snippets) >= limit:
            break
    return "；".join(snippets) or INSUFFICIENT_EVIDENCE_TEXT


def _evidence_for_papers(
    paper_ids: Iterable[str],
    theme_evidence: Mapping[str, Mapping[str, tuple[EvidenceRef, ...]]],
) -> tuple[EvidenceRef, ...]:
    evidence: list[EvidenceRef] = []
    selected = set(paper_ids)
    for theme_refs in theme_evidence.values():
        for paper_id, refs in theme_refs.items():
            if paper_id in selected:
                evidence.extend(refs)
    return _unique_evidence(evidence)


def _join_evidence(evidence: Iterable[EvidenceRef]) -> str:
    return " ".join(item.quote for item in evidence)


def _join_paper_metadata(paper: PaperRecord) -> str:
    return " ".join(
        (
            paper.title,
            paper.abstract,
            paper.journal,
            " ".join(paper.keywords),
        )
    )


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    normalized = text.casefold()
    return any(term.casefold() in normalized for term in terms if term)


def _flatten(
    values: Iterable[Iterable[EvidenceRef]],
) -> tuple[EvidenceRef, ...]:
    return _unique_evidence(item for group in values for item in group)


def _unique_evidence(
    evidence: Iterable[EvidenceRef],
) -> tuple[EvidenceRef, ...]:
    """按 chunk ID 去重，同时保留数据库返回的原文顺序。"""

    unique: dict[str, EvidenceRef] = {}
    for item in evidence:
        unique.setdefault(item.chunk_id, item)
    return tuple(unique.values())


def _content_label(kind: str) -> str:
    content = CONTENT_BY_KIND.get(kind)
    return content.label if content is not None else kind


def _deduplicate_edges(
    edges: Iterable[KnowledgeMapEdge],
) -> tuple[KnowledgeMapEdge, ...]:
    unique: dict[str, KnowledgeMapEdge] = {}
    for edge in edges:
        unique.setdefault(edge.edge_id, edge)
    return tuple(unique.values())


__all__ = [
    "CORE_DIRECTION",
    "CORE_DIRECTION_ID",
    "KNOWLEDGE_MAP_ENGINE_VERSION",
    "KnowledgeFinding",
    "KnowledgeGraphStatistics",
    "KnowledgeMap",
    "KnowledgeMapEdge",
    "KnowledgeMapNode",
    "KnowledgeMapPaper",
    "KnowledgeThemeDefinition",
    "RETRIEVAL_SCOPE_NOTE",
    "THEMES",
    "THEME_BY_ID",
    "ThemeStatistic",
    "build_knowledge_map",
    "build_markdown_report",
]
