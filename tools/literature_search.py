"""文献检索策略生成工具。

本模块只帮助组织关键词和数据库检索式，不联网返回文献，也不生成虚构引用。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SearchStrategy:
    """可复制的文献检索策略。"""

    topic: str
    chinese_query: str
    english_query_hint: str
    databases: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["databases"] = list(self.databases)
        result["notes"] = list(self.notes)
        return result


def build_search_strategy(
    topic: str,
    population: str = "",
    method_or_intervention: str = "",
) -> SearchStrategy:
    """根据研究主题生成数据库检索表达式草稿。

    英文检索式只提供结构提示，真实英文同义词需要研究者根据数据库词表补充。
    """

    clean_topic = _clean_text(topic)
    if not clean_topic:
        raise ValueError("研究主题不能为空。")

    clean_population = _clean_text(population)
    clean_method = _clean_text(method_or_intervention)
    topic_terms = _split_terms(clean_topic)

    chinese_terms = _join_or_terms(topic_terms)
    groups = [f"（{chinese_terms}）"]
    if clean_population:
        groups.append(f"（{_join_or_terms(_split_terms(clean_population))}）")
    if clean_method:
        groups.append(f"（{_join_or_terms(_split_terms(clean_method))}）")
    chinese_query = " AND ".join(groups)

    english_hint = (
        f'主题词示例："sport" AND "physical education" AND ({clean_topic})；'
        "请从数据库主题词表中补充同义词，并分别测试标题、摘要和关键词字段。"
    )

    return SearchStrategy(
        topic=clean_topic,
        chinese_query=chinese_query,
        english_query_hint=english_hint,
        databases=(
            "中国知网 CNKI",
            "万方数据",
            "Web of Science",
            "PubMed",
            "SPORTDiscus",
        ),
        notes=(
            "先在 CNKI 或 Web of Science 记录检索日期、数据库范围和筛选条件。",
            "使用“研究对象 AND 干预或暴露 AND 结局指标”的分组方式逐步缩小范围。",
            "当前工具不返回真实文献；引用前必须打开原文核验作者、年份和结论。",
        ),
    )


def _clean_text(value: str) -> str:
    return " ".join(value.strip().split())


def _split_terms(value: str) -> list[str]:
    """按空格、逗号、分号和中文逗号拆分关键词。"""

    terms = [item.strip() for item in re.split(r"[\s,，;；]+", value) if item.strip()]
    return list(dict.fromkeys(terms))[:8]


def _join_or_terms(terms: list[str]) -> str:
    return " OR ".join(terms) if terms else ""
