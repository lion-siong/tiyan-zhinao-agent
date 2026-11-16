"""体育科研证据清洗、主题定义与自动化质量初筛。

这些规则只负责把可核验文本从论文中筛出来，并给出保守的自动提示。它们不是
正式的系统综述、偏倚风险评价或 GRADE 专家评级。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


INSUFFICIENT_EVIDENCE_TEXT = "证据不足，暂不能判断"
AUTO_GRADE_DISCLAIMER = "自动质量初筛，非正式 GRADE 评价"
ANALYSIS_ENGINE_VERSION = "evidence-synthesis-v5"

BOILERPLATE_MARKERS = (
    "creative commons",
    "all rights reserved",
    "terms and conditions",
    "article is licensed",
    "distribution or reproduction",
    "academic practice",
    "credit line",
    "unless otherwise stated",
    "creativecommons",
    "licenses/by",
    "license, which permits",
    "copyright",
    "©",
    "correspondence",
    "corresponding author",
    "author contributions",
    "authors’ contributions",
    "authors' contributions",
    "conflict of interest",
    "competing interests",
    "acknowledg",
    "funding",
    "data availability",
    "supplementary material",
    "reviewed by",
    "edited by",
    "received:",
    "accepted:",
    "published:",
    "how to cite",
    "citation:",
)

METHOD_MARKERS = (
    "method",
    "methods",
    "participants",
    "randomized",
    "randomised",
    "controlled trial",
    "cross-sectional",
    "systematic review",
    "meta-analysis",
    "questionnaire",
    "measured",
    "assessed",
    "研究设计",
    "研究方法",
    "研究对象",
    "随机对照",
    "横断面",
    "问卷调查",
    "测量",
)

RESULT_MARKERS = (
    "result",
    "results",
    "we found",
    "found that",
    "showed that",
    "significant",
    "no significant",
    "not significant",
    "improved",
    "increased",
    "decreased",
    "reduced",
    "associated with",
    "effect size",
    "95% ci",
    "p =",
    "p<",
    "p <",
    "结果表明",
    "结果显示",
    "结果发现",
    "研究发现",
    "显著",
    "无显著",
    "改善",
    "提高",
    "降低",
)

CONCLUSION_MARKERS = (
    "conclusion",
    "conclusions",
    "in conclusion",
    "we conclude",
    "findings suggest",
    "研究结论",
    "结论与建议",
    "综上所述",
)

LIMITATION_MARKERS = (
    "limitation",
    "limitations",
    "future studies",
    "further research",
    "future research",
    "研究局限",
    "局限性",
    "未来研究",
)


@dataclass(frozen=True)
class ResearchQuestionDefinition:
    """一个可独立形成共识或争议的研究问题。"""

    question_id: str
    label: str
    question: str
    exercise_terms: tuple[str, ...]
    focus_terms: tuple[str, ...]
    benefit_terms: tuple[str, ...]
    adverse_terms: tuple[str, ...]
    direct_outcome_terms: tuple[str, ...] = ()


RESEARCH_QUESTIONS: tuple[ResearchQuestionDefinition, ...] = (
    ResearchQuestionDefinition(
        question_id="question:cardiorespiratory",
        label="体育锻炼与心肺耐力",
        question="体育锻炼是否改善儿童青少年的心肺耐力或心肺适能？",
        exercise_terms=(
            "体育锻炼",
            "身体活动",
            "运动",
            "exercise",
            "physical activity",
            "training",
        ),
        focus_terms=(
            "心肺耐力",
            "心肺适能",
            "最大摄氧量",
            "有氧能力",
            "cardiorespiratory",
            "cardiorespiratory fitness",
            "vo2",
            "vo₂",
            "aerobic capacity",
        ),
        benefit_terms=(
            "improved cardiorespiratory",
            "increased cardiorespiratory",
            "increased vo2",
            "higher vo2",
            "improved aerobic capacity",
            "改善心肺",
            "提高心肺",
            "提高最大摄氧量",
            "改善有氧能力",
        ),
        adverse_terms=(
            "decreased cardiorespiratory",
            "lower cardiorespiratory",
            "reduced vo2",
            "worsened cardiorespiratory",
            "降低心肺",
            "心肺适能下降",
            "最大摄氧量下降",
        ),
    ),
    ResearchQuestionDefinition(
        question_id="question:body-composition",
        label="体育锻炼与身体成分",
        question="体育锻炼是否改善儿童青少年的身体成分或肥胖相关指标？",
        exercise_terms=(
            "体育锻炼",
            "身体活动",
            "运动",
            "exercise",
            "physical activity",
            "training",
        ),
        focus_terms=(
            "身体成分",
            "体脂",
            "脂肪量",
            "瘦体重",
            "bmi",
            "肥胖",
            "body composition",
            "body fat",
            "fat mass",
            "lean mass",
            "adiposity",
            "obesity",
        ),
        benefit_terms=(
            "reduced body fat",
            "decreased body fat",
            "lower body fat",
            "reduced bmi",
            "decreased bmi",
            "lower bmi",
            "reduced fat mass",
            "decreased fat mass",
            "increased lean mass",
            "improved body composition",
            "reduced obesity",
            "improved adiposity",
            "降低体脂",
            "体脂率下降",
            "降低 bmi",
            "降低肥胖",
            "改善身体成分",
            "增加瘦体重",
        ),
        adverse_terms=(
            "increased body fat",
            "higher body fat",
            "increased bmi",
            "higher bmi",
            "increased fat mass",
            "worsened body composition",
            "体脂增加",
            "体脂率升高",
            "bmi 升高",
            "身体成分恶化",
        ),
    ),
    ResearchQuestionDefinition(
        question_id="question:muscular-strength",
        label="体育锻炼与肌肉力量",
        question="体育锻炼是否改善儿童青少年的肌肉力量、爆发力或速度表现？",
        exercise_terms=(
            "体育锻炼",
            "身体活动",
            "运动训练",
            "力量训练",
            "exercise",
            "physical activity",
            "training",
        ),
        focus_terms=(
            "肌肉力量",
            "力量",
            "爆发力",
            "速度",
            "跳跃",
            "muscle strength",
            "strength",
            "power",
            "jump",
            "sprint",
        ),
        benefit_terms=(
            "increased strength",
            "improved strength",
            "greater strength",
            "improved power",
            "increased power",
            "improved jump",
            "improved sprint",
            "greater gains",
            "提高力量",
            "增强力量",
            "力量提高",
            "提高爆发力",
            "改善跳跃",
            "改善速度",
        ),
        adverse_terms=(
            "decreased strength",
            "lower strength",
            "reduced strength",
            "impaired power",
            "worsened jump",
            "力量下降",
            "力量降低",
            "爆发力下降",
        ),
    ),
    ResearchQuestionDefinition(
        question_id="question:physical-fitness",
        label="体育锻炼与身体素质",
        question="体育锻炼是否改善儿童青少年的总体体质健康或身体素质？",
        exercise_terms=(
            "体育锻炼",
            "身体活动",
            "运动",
            "exercise",
            "physical activity",
            "physical education",
            "training",
        ),
        focus_terms=(
            "体质健康",
            "身体素质",
            "体适能",
            "physical fitness",
            "health-related fitness",
            "fitness",
        ),
        benefit_terms=(
            "improved physical fitness",
            "increased physical fitness",
            "improved fitness",
            "greater fitness",
            "improved physical education",
            "改善体质",
            "提高体质",
            "改善身体素质",
            "提高身体素质",
            "体适能提高",
        ),
        adverse_terms=(
            "decreased physical fitness",
            "lower physical fitness",
            "reduced fitness",
            "worsened fitness",
            "体质健康下降",
            "身体素质下降",
            "体适能下降",
        ),
    ),
    ResearchQuestionDefinition(
        question_id="question:mental-health",
        label="体育锻炼与心理健康",
        question="体育锻炼是否改善儿童青少年的心理健康、抑郁或焦虑结局？",
        exercise_terms=(
            "体育锻炼",
            "身体活动",
            "运动",
            "exercise",
            "physical activity",
            "training",
        ),
        focus_terms=(
            "心理健康",
            "抑郁",
            "焦虑",
            "主观幸福感",
            "自尊",
            "mental health",
            "depression",
            "anxiety",
            "well-being",
            "wellbeing",
            "self-esteem",
        ),
        benefit_terms=(
            "reduced depression",
            "decreased depression",
            "lower depression",
            "reduced anxiety",
            "decreased anxiety",
            "lower anxiety",
            "improved mental health",
            "improved well-being",
            "improved wellbeing",
            "improved self-esteem",
            "beneficial effects",
            "缓解抑郁",
            "降低抑郁",
            "减轻焦虑",
            "降低焦虑",
            "改善心理健康",
            "提高幸福感",
            "提升自尊",
        ),
        adverse_terms=(
            "increased depression",
            "higher depression",
            "increased anxiety",
            "higher anxiety",
            "worsened mental health",
            "lower well-being",
            "抑郁加重",
            "焦虑加重",
            "心理健康恶化",
            "幸福感下降",
        ),
    ),
    ResearchQuestionDefinition(
        question_id="question:frequency",
        label="运动频率与健康效果",
        question="更高的运动频率是否与儿童青少年更好的体质健康或心理结局相关？",
        exercise_terms=(
            "体育锻炼",
            "身体活动",
            "运动",
            "exercise",
            "physical activity",
            "training",
        ),
        focus_terms=(
            "运动频率",
            "每周次数",
            "每周",
            "次/周",
            "frequency",
            "sessions per week",
            "times per week",
            "days per week",
        ),
        benefit_terms=(
            "higher frequency",
            "greater frequency",
            "more frequent",
            "dose-response",
            "greater improvement",
            "更高频率",
            "频率越高",
            "更多次数",
            "剂量反应",
            "改善更明显",
        ),
        adverse_terms=(
            "lower frequency",
            "less frequent",
            "no dose-response",
            "频率较低",
            "频率更低",
            "无剂量反应",
        ),
    ),
    ResearchQuestionDefinition(
        question_id="question:intensity",
        label="运动强度与健康效果",
        question="更高运动强度是否比低至中等强度带来更大的体质健康或心理改善？",
        exercise_terms=(
            "体育锻炼",
            "身体活动",
            "运动",
            "exercise",
            "physical activity",
            "training",
        ),
        focus_terms=(
            "运动强度",
            "高强度",
            "中等强度",
            "mvpa",
            "vigorous",
            "moderate-to-vigorous",
            "high-intensity",
            "exercise intensity",
        ),
        benefit_terms=(
            "higher intensity",
            "greater intensity",
            "vigorous intensity",
            "high-intensity training improved",
            "greater improvement",
            "更高强度",
            "强度较高",
            "高强度更有效",
            "改善更明显",
        ),
        adverse_terms=(
            "lower intensity",
            "moderate intensity was superior",
            "no additional benefit",
            "强度较低",
            "中等强度更有效",
            "没有额外获益",
        ),
    ),
)

RESEARCH_QUESTION_BY_ID = {
    question.question_id: question for question in RESEARCH_QUESTIONS
}


def contains_any(text: str, terms: Iterable[str]) -> bool:
    normalized = _normalize(text)
    return any(_normalize(term) in normalized for term in terms if term)


def evidence_quality_flags(
    text: str,
    *,
    section_hint: str = "",
    source_type: str = "全文",
) -> tuple[bool, str, str]:
    """返回 ``(是否研究证据, 段落类型, 排除原因)``。"""

    normalized = _normalize(text)
    if not normalized:
        return False, "非研究内容", "空文本"

    boilerplate_hits = tuple(
        marker for marker in BOILERPLATE_MARKERS if marker in normalized
    )
    result_hits = tuple(marker for marker in RESULT_MARKERS if marker in normalized)
    method_hits = tuple(marker for marker in METHOD_MARKERS if marker in normalized)
    conclusion_hits = tuple(
        marker for marker in CONCLUSION_MARKERS if marker in normalized
    )
    limitation_hits = tuple(
        marker for marker in LIMITATION_MARKERS if marker in normalized
    )
    research_score = (
        min(5, len(result_hits) * 2)
        + min(3, len(method_hits))
        + min(3, len(conclusion_hits) * 2)
        + min(2, len(limitation_hits))
    )
    boilerplate_score = len(boilerplate_hits) * 2

    if _mostly_reference_list(normalized):
        return False, "参考文献", "片段以参考文献条目为主"
    if any(
        marker
        in normalized
        for marker in (
            "creative commons",
            "all rights reserved",
            "article is licensed",
            "distribution or reproduction",
            "academic practice",
            "credit line",
            "unless otherwise stated",
            "creativecommons",
            "licenses/by",
        )
    ):
        return False, "版权/元信息", "版权或许可声明"
    if boilerplate_score >= 4 and research_score < 5:
        return False, "版权/元信息", "版权、资助、单位或期刊元信息"
    if re.search(r"\b(?:received|accepted|published)\s*:", normalized) and not (
        result_hits or conclusion_hits
    ):
        return False, "版权/元信息", "投稿日期等期刊元信息"

    hint = _normalize(section_hint)
    if hint in {"result", "结果"} or result_hits:
        section = "结果"
    elif hint in {"conclusion", "结论"} or conclusion_hits:
        section = "结论"
    elif hint in {"limitation", "局限"} or limitation_hits:
        section = "局限"
    elif hint in {"method", "方法"} or method_hits:
        section = "方法"
    elif hint in {"目的"}:
        section = "目的"
    elif source_type == "摘要":
        section = "摘要"
    else:
        section = "其他正文"

    if source_type == "摘要":
        if len(normalized) >= 80 and research_score > 0:
            return True, section, ""
        return False, "非研究内容", "摘要缺少研究结果或方法信息"
    if research_score <= 0:
        return False, "非研究内容", "没有研究目的、方法、结果或结论信号"
    return True, section, ""


def classify_result_direction(
    text: str,
    question: ResearchQuestionDefinition,
) -> tuple[str, tuple[str, ...]]:
    """按具体研究问题判断保守结果方向，混合结果不强行归类。"""

    normalized = _normalize(text)
    positive_hits = tuple(
        marker for marker in question.benefit_terms if _normalize(marker) in normalized
    )
    adverse_hits = tuple(
        marker for marker in question.adverse_terms if _normalize(marker) in normalized
    )
    null_hits = tuple(
        marker
        for marker in (
            "no significant",
            "not significant",
            "no difference",
            "did not differ",
            "without significant",
            "无显著",
            "不显著",
            "没有显著",
            "无差异",
        )
        if marker in normalized
    )
    effect_terms = positive_hits + adverse_hits + null_hits
    if positive_hits and adverse_hits:
        return "混合结果", effect_terms
    if null_hits and (positive_hits or adverse_hits):
        return "混合结果", effect_terms
    if null_hits:
        return "无显著关系", effect_terms
    if adverse_hits:
        return "负向结果", effect_terms
    if positive_hits:
        return "正向结果", effect_terms
    return "待评估", ()


def infer_study_design(title: str, *texts: str) -> str:
    text = _normalize(" ".join((title, *texts)))
    rules = (
        (("umbrella review", "meta-meta-analysis"), "伞状综述/元-元分析"),
        (("network meta-analysis", "network meta analysis"), "网络 Meta 分析"),
        (("meta-analysis", "meta analysis", "元分析"), "Meta 分析"),
        (("systematic review", "系统综述"), "系统综述"),
        (
            ("randomized controlled trial", "randomised controlled trial", "随机对照"),
            "随机对照试验",
        ),
        (("cohort", "队列研究"), "队列研究"),
        (("cross-sectional", "横断面"), "横断面研究"),
        (("qualitative evidence synthesis", "质性证据综合"), "质性证据综合"),
        (("narrative synthesis", "叙述性综合"), "叙述性系统综述"),
    )
    for terms, label in rules:
        if any(term in text for term in terms):
            return label
    return "设计未明确"


def extract_sample_size(*texts: str) -> int | None:
    """从原文中用保守模式提取明确的参与者人数。"""

    text = " ".join(texts)
    patterns = (
        r"(?:n\s*=\s*|total of |sample of |included |participants were )"
        r"(\d{2,7})",
        r"(\d{2,7})\s+(?:children|adolescents|students|participants|youth)",
        r"(?:样本|研究对象|参与者)[^\d]{0,24}(\d{2,7})",
    )
    values: list[int] = []
    for pattern in patterns:
        values.extend(int(match) for match in re.findall(pattern, text, re.I))
    values = [value for value in values if 10 <= value <= 5_000_000]
    return max(values) if values else None


def preliminary_certainty(
    *,
    status: str,
    designs: Iterable[str],
    sample_sizes: Iterable[int | None],
    full_text_paper_count: int,
    abstract_only_paper_count: int,
) -> tuple[str, str]:
    """生成保守的自动确定性初筛及理由，不冒充正式 GRADE。"""

    design_list = tuple(designs)
    sizes = tuple(size for size in sample_sizes if size is not None)
    design_points = sum(
        {
            "伞状综述/元-元分析": 3.0,
            "网络 Meta 分析": 3.0,
            "Meta 分析": 2.5,
            "系统综述": 2.0,
            "质性证据综合": 1.5,
            "叙述性系统综述": 1.5,
            "随机对照试验": 2.0,
            "队列研究": 1.0,
            "横断面研究": 0.5,
            "设计未明确": 0.0,
        }.get(design, 0.0)
        for design in design_list
    )
    strongest = max(
        (
            {
                "伞状综述/元-元分析": 3.0,
                "网络 Meta 分析": 3.0,
                "Meta 分析": 2.5,
                "系统综述": 2.0,
                "质性证据综合": 1.5,
                "叙述性系统综述": 1.5,
                "随机对照试验": 2.0,
                "队列研究": 1.0,
                "横断面研究": 0.5,
                "设计未明确": 0.0,
            }.get(design, 0.0)
            for design in design_list
        ),
        default=0.0,
    )
    score = strongest + min(1.5, design_points / 3.0)
    if status in {
        "初步共识",
        "有一致研究发现",
        "一致未观察到显著差异",
    }:
        score += 1.0
    elif status.startswith("存在争议"):
        score -= 0.8
    if any(size >= 1000 for size in sizes):
        score += 0.8
    elif sizes and max(sizes) < 100:
        score -= 0.8
    score += min(1.0, full_text_paper_count * 0.15)
    score -= min(1.0, abstract_only_paper_count * 0.2)

    reasons = [
        "研究设计：" + ("、".join(dict.fromkeys(design_list)) or "未明确"),
        (
            f"可提取样本量 {len(sizes)} 篇"
            if sizes
            else "未从可定位原文提取到明确样本量"
        ),
        f"全文证据 {full_text_paper_count} 篇，摘要级证据 {abstract_only_paper_count} 篇",
    ]
    if status.startswith("存在争议"):
        reasons.append(
            "同一研究问题检测到正向与负向结果的待审核候选；"
            "未达统计显著不归入反方"
        )
    if full_text_paper_count < 2 or abstract_only_paper_count:
        reasons.append("全文覆盖率有限，摘要信息不能替代全文核验")
    reasons.append("偏倚风险和发表偏倚尚未完成人工评价")
    reason_text = "；".join(reasons) + f"。{AUTO_GRADE_DISCLAIMER}。"

    if status == "证据不足":
        return "无法评估", reason_text
    if score >= 7.0:
        return "高", reason_text
    if score >= 5.0:
        return "中等", reason_text
    if score >= 2.5:
        return "低", reason_text
    return "极低", reason_text


def _mostly_reference_list(normalized: str) -> bool:
    reference_headings = normalized.count("references")
    numeric_citations = len(re.findall(r"\[\d{1,3}\]", normalized))
    year_citations = len(
        re.findall(r"\b(?:19|20)\d{2}\b[;:,]\s*[A-Za-z]", normalized)
    )
    doi_citations = normalized.count("doi:")
    if reference_headings and len(normalized) < 160:
        return True
    return (
        numeric_citations >= 4
        or year_citations >= 4
        or (doi_citations >= 3 and len(normalized) < 1000)
    )


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()
