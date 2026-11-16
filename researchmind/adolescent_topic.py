"""青少年体质健康专题的可解释规则。

本模块只提供本地文本规则，不调用大模型。相关度判断用于决定论文是否进入
专题核心库，或进入人工审核；判断结果不是学术质量评价，也不能替代人工核验。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


CORE_RESEARCH_DIRECTION = "青少年体质健康与体育锻炼干预研究"
CORE_DOMAIN_NAME = "青少年体质健康与体育锻炼干预"
CORE_DOMAIN_DESCRIPTION = (
    "围绕青少年体质健康评价、体育锻炼干预、运动处方、心理与运动行为、"
    "干预效果评价及影响因素与作用机制建立的可追溯专题论文库。"
)
CORE_DOMAIN_KEYWORDS = (
    "青少年体质健康",
    "体育锻炼干预",
    "运动处方",
    "运动强度",
    "身体素质",
    "青少年肥胖",
)
TOPIC_RULE_MODE = "本地规则"
INSUFFICIENT_EVIDENCE_TEXT = "证据不足，暂不能判断"


@dataclass(frozen=True)
class TopicThemeDefinition:
    """专题第二层研究主题及其可审计的匹配词。"""

    theme_id: str
    order: int
    label: str
    description: str
    match_terms: tuple[str, ...]


@dataclass(frozen=True)
class TopicContentDefinition:
    """专题第三层研究内容及允许使用的原文证据字段。"""

    kind: str
    order: int
    label: str
    analysis_fields: tuple[str, ...]
    match_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class TopicRelevance:
    """一条可展示、可复核的专题相关度判断。"""

    status: str
    score: float
    reasons: tuple[str, ...]
    matched_theme_ids: tuple[str, ...]

    @property
    def is_core_relevant(self) -> bool:
        return self.status == "相关"

    @property
    def display_status(self) -> str:
        return f"{self.status}（{TOPIC_RULE_MODE}，匹配度 {self.score:.0%}）"


THEMES: tuple[TopicThemeDefinition, ...] = (
    TopicThemeDefinition(
        theme_id="theme:health-assessment",
        order=1,
        label="青少年体质健康评价",
        description="青少年体质健康水平、评价指标、测量方法及人群差异。",
        match_terms=(
            "体质健康",
            "健康评价",
            "健康水平",
            "身体素质",
            "心肺适能",
            "physical fitness",
            "health-related fitness",
            "fitness assessment",
            "health status",
            "body composition",
            "cardiorespiratory fitness",
        ),
    ),
    TopicThemeDefinition(
        theme_id="theme:exercise-intervention",
        order=2,
        label="体育锻炼干预",
        description="不同体育锻炼方式、运动项目及干预实施策略。",
        match_terms=(
            "体育锻炼干预",
            "运动干预",
            "体育干预",
            "有氧运动",
            "抗阻训练",
            "力量训练",
            "球类运动",
            "exercise intervention",
            "physical activity intervention",
            "exercise training",
            "aerobic exercise",
            "resistance training",
            "sports intervention",
        ),
    ),
    TopicThemeDefinition(
        theme_id="theme:exercise-prescription",
        order=3,
        label="运动处方与干预方案",
        description="运动处方、运动剂量、强度、频率、周期和课程化方案。",
        match_terms=(
            "运动处方",
            "干预方案",
            "运动强度",
            "运动频率",
            "干预周期",
            "运动剂量",
            "exercise prescription",
            "exercise program",
            "exercise intensity",
            "exercise frequency",
            "intervention duration",
            "exercise dose",
            "fitt",
        ),
    ),
    TopicThemeDefinition(
        theme_id="theme:psych-behavior",
        order=4,
        label="心理与运动行为影响",
        description="体育锻炼与青少年心理健康、运动动机、行为及自我认知的关系。",
        match_terms=(
            "心理健康",
            "运动行为",
            "身体活动行为",
            "运动动机",
            "自我效能",
            "主观幸福感",
            "心理影响",
            "mental health",
            "physical activity behavior",
            "exercise motivation",
            "self-efficacy",
            "well-being",
            "psychological",
        ),
    ),
    TopicThemeDefinition(
        theme_id="theme:outcome-evaluation",
        order=5,
        label="干预效果与评价指标",
        description="运动干预效果、结局指标、研究设计和统计评价。",
        match_terms=(
            "干预效果",
            "效果评价",
            "评价指标",
            "结局指标",
            "测量指标",
            "体适能指标",
            "intervention effect",
            "outcome measure",
            "outcome assessment",
            "evaluation index",
            "measurement",
            "randomized controlled",
        ),
    ),
    TopicThemeDefinition(
        theme_id="theme:mechanism",
        order=6,
        label="影响因素与作用机制",
        description="影响体育锻炼效果和体质健康差异的个体、家庭、学校及机制因素。",
        match_terms=(
            "影响因素",
            "作用机制",
            "中介作用",
            "调节作用",
            "相关性",
            "风险因素",
            "influencing factor",
            "mechanism",
            "mediator",
            "moderator",
            "association",
            "determinant",
            "risk factor",
        ),
    ),
)

THEME_BY_ID = {theme.theme_id: theme for theme in THEMES}

CONTENT_ITEMS: tuple[TopicContentDefinition, ...] = (
    TopicContentDefinition(
        kind="population",
        order=1,
        label="研究对象",
        analysis_fields=("population",),
    ),
    TopicContentDefinition(
        kind="methods",
        order=2,
        label="研究方法",
        analysis_fields=("methods",),
    ),
    TopicContentDefinition(
        kind="intervention_period",
        order=3,
        label="干预周期",
        analysis_fields=("methods", "results"),
        match_terms=(
            "干预周期",
            "持续",
            "周",
            "月",
            "week",
            "weeks",
            "month",
            "months",
            "duration",
            "program length",
        ),
    ),
    TopicContentDefinition(
        kind="exercise_intensity",
        order=4,
        label="运动强度",
        analysis_fields=("methods", "results"),
        match_terms=(
            "运动强度",
            "低强度",
            "中等强度",
            "高强度",
            "心率",
            "最大摄氧量",
            "intensity",
            "moderate",
            "vigorous",
            "heart rate",
            "vo2",
        ),
    ),
    TopicContentDefinition(
        kind="exercise_frequency",
        order=5,
        label="运动频率",
        analysis_fields=("methods", "results"),
        match_terms=(
            "运动频率",
            "每周",
            "次/周",
            "频率",
            "frequency",
            "per week",
            "times per week",
            "sessions",
        ),
    ),
    TopicContentDefinition(
        kind="measures",
        order=6,
        label="评价指标",
        analysis_fields=("methods", "results"),
        match_terms=(
            "评价指标",
            "测量指标",
            "结局指标",
            "测试指标",
            "问卷",
            "量表",
            "体脂率",
            "体质指数",
            "bmi",
            "body fat",
            "outcome",
            "measurement",
            "questionnaire",
            "test",
        ),
    ),
    TopicContentDefinition(
        kind="results",
        order=7,
        label="研究结果",
        analysis_fields=("results",),
    ),
    TopicContentDefinition(
        kind="controversies",
        order=8,
        label="学术争议",
        analysis_fields=(),
    ),
)

CONTENT_BY_KIND = {item.kind: item for item in CONTENT_ITEMS}

TEEN_MARKERS = (
    "青少年",
    "中学生",
    "初中生",
    "高中生",
    "儿童青少年",
    "学龄儿童",
    "青春期",
    "adolescent",
    "teenager",
    "youth",
    "school-aged",
    "school student",
    "children and adolescents",
)
COLLEGE_MARKERS = (
    "大学生",
    "本科生",
    "研究生",
    "高校学生",
    "高职学生",
    "college student",
    "university student",
    "undergraduate",
    "postgraduate",
    "higher education",
)
ADULT_MARKERS = (
    "成年人",
    "老年人",
    "older adult",
    "middle-aged adult",
    "adult population",
)
CORE_TOPIC_MARKERS = (
    "体质健康",
    "体育锻炼",
    "身体活动",
    "运动干预",
    "运动处方",
    "physical fitness",
    "physical activity",
    "exercise intervention",
    "exercise prescription",
    "sport intervention",
)


def assess_topic_relevance(
    *,
    title: str,
    abstract: str = "",
    keywords: Iterable[str] = (),
    population_text: str = "",
    full_text: str = "",
) -> TopicRelevance:
    """按可解释规则判断论文是否适合进入青少年专题核心库。"""

    metadata = " ".join((title, abstract, *tuple(keywords)))
    combined = f"{metadata} {population_text} {full_text}"
    teen_in_population = contains_any(population_text, TEEN_MARKERS)
    teen_in_metadata = contains_any(metadata, TEEN_MARKERS)
    college_in_population = contains_any(population_text, COLLEGE_MARKERS)
    college_in_metadata = contains_any(metadata, COLLEGE_MARKERS)
    adult_in_population = contains_any(population_text, ADULT_MARKERS)
    adult_in_metadata = contains_any(metadata, ADULT_MARKERS)
    has_topic_marker = contains_any(metadata, CORE_TOPIC_MARKERS)
    has_teen = teen_in_population or teen_in_metadata
    has_college = college_in_population or college_in_metadata
    has_adult = adult_in_population or adult_in_metadata
    matched_theme_ids = match_theme_ids(combined)

    reasons: list[str] = []
    if has_teen:
        reasons.append("题名、摘要或研究对象原文包含青少年样本线索。")
    if has_topic_marker:
        reasons.append("题名或摘要涉及体质健康、体育锻炼或运动干预。")
    if matched_theme_ids:
        reasons.append(
            "命中研究主题："
            + "、".join(THEME_BY_ID[item].label for item in matched_theme_ids)
        )

    if has_college and has_teen:
        reasons.append("同时出现青少年与大学生线索，样本边界不清，需要人工审核。")
        return TopicRelevance("待审核", 0.50, tuple(reasons), matched_theme_ids)
    if has_college:
        reasons.append("题名、摘要或研究对象原文显示为大学生或高校学生。")
        return TopicRelevance("不相关", 0.05, tuple(reasons), matched_theme_ids)
    if has_adult and not has_teen:
        reasons.append("题名、摘要或研究对象原文显示为成年人样本。")
        return TopicRelevance("不相关", 0.10, tuple(reasons), matched_theme_ids)
    if not has_teen:
        reasons.append("未找到可核验的青少年研究对象线索。")
        return TopicRelevance("待审核", 0.20, tuple(reasons), matched_theme_ids)
    if not has_topic_marker and not matched_theme_ids:
        reasons.append("未找到体质健康或体育锻炼干预主题线索。")
        return TopicRelevance("待审核", 0.35, tuple(reasons), matched_theme_ids)

    score = 0.62
    score += min(0.18, len(matched_theme_ids) * 0.035)
    if teen_in_population:
        score += 0.10
    if has_topic_marker:
        score += 0.06
    reasons.append("符合专题研究对象与主题范围，可进入核心库并继续核验全文。")
    return TopicRelevance(
        "相关",
        min(score, 0.98),
        tuple(reasons),
        matched_theme_ids,
    )


def match_theme_ids(text: str) -> tuple[str, ...]:
    """返回文本命中的研究主题，不把元数据命中当成证据节点。"""

    return tuple(
        theme.theme_id
        for theme in THEMES
        if contains_any(text, theme.match_terms)
    )


def contains_any(text: str, terms: Iterable[str]) -> bool:
    """大小写不敏感地检查任一关键词是否出现。"""

    normalized = str(text or "").casefold()
    return any(term.casefold() in normalized for term in terms if term)


__all__ = [
    "ADULT_MARKERS",
    "COLLEGE_MARKERS",
    "CONTENT_BY_KIND",
    "CONTENT_ITEMS",
    "CORE_DOMAIN_DESCRIPTION",
    "CORE_DOMAIN_KEYWORDS",
    "CORE_DOMAIN_NAME",
    "CORE_RESEARCH_DIRECTION",
    "CORE_TOPIC_MARKERS",
    "INSUFFICIENT_EVIDENCE_TEXT",
    "TEEN_MARKERS",
    "THEME_BY_ID",
    "THEMES",
    "TOPIC_RULE_MODE",
    "TopicContentDefinition",
    "TopicRelevance",
    "TopicThemeDefinition",
    "assess_topic_relevance",
    "contains_any",
    "match_theme_ids",
]
