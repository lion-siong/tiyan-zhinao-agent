"""基于 Function Calling 的公开文献检索与综述 Agent。

Agent 的公开检索边界是 Semantic Scholar 元数据和 arXiv 官方预印本元数据。
它不会爬取知网，也不会把外部论文自动写入用户上传 PDF 的本地 RAG 知识库。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from config import Settings, settings as default_settings
from services.llm_service import LLMConfigurationError, LLMService
from tools.arxiv_search import ArxivSearchError, arxiv_search
from tools.citation_formatter import format_gbt7714
from tools.semantic_scholar import (
    SemanticScholarError,
    SemanticScholarPaper,
    semantic_scholar_search,
)


SEMANTIC_SCHOLAR_TOOL = {
    "type": "function",
    "function": {
        "name": "semantic_scholar_search",
        "description": (
            "检索 Semantic Scholar 公开论文元数据。只返回题名、作者、"
            "年份、摘要、DOI 和开放获取链接，不下载论文全文。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "根据用户研究问题优化后的检索式，可包含中英文关键词、"
                        "研究对象、变量和结局指标。"
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "需要返回的论文条数。",
                    "minimum": 1,
                    "maximum": 20,
                },
            },
            "required": ["query", "limit"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class LiteratureResearchResult:
    """一次公开文献 Agent 运行结果。"""

    original_query: str
    optimized_query: str
    papers: tuple[SemanticScholarPaper, ...]
    review: str
    references: tuple[str, ...]
    mode: str
    source: str
    warnings: tuple[str, ...] = ()
    success: bool = True
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转换供界面或后续模块使用的普通字典。"""

        return {
            "original_query": self.original_query,
            "optimized_query": self.optimized_query,
            "papers": [paper.to_dict() for paper in self.papers],
            "review": self.review,
            "references": list(self.references),
            "mode": self.mode,
            "source": self.source,
            "warnings": list(self.warnings),
            "success": self.success,
            "message": self.message,
        }


class LiteratureResearchAgent:
    """通过模型 Function Calling 自动检索并整理公开文献。"""

    def __init__(
        self,
        config: Settings | None = None,
        llm_service: LLMService | None = None,
        search_tool: Callable[..., tuple[SemanticScholarPaper, ...]] | None = None,
        preprint_tool: Callable[..., tuple[SemanticScholarPaper, ...]] | None = None,
    ) -> None:
        self.config = config or default_settings
        self.llm_service = llm_service or LLMService(self.config)
        self.search_tool = search_tool or semantic_scholar_search
        self.preprint_tool = preprint_tool or arxiv_search

    def run(
        self,
        research_question: str,
        *,
        limit: int = 8,
    ) -> LiteratureResearchResult:
        """执行“理解问题 -> 优化检索式 -> 工具检索 -> 整理综述”。"""

        clean_question = research_question.strip()
        if not clean_question:
            return LiteratureResearchResult(
                original_query="",
                optimized_query="",
                papers=(),
                review="",
                references=(),
                mode="输入校验",
                source="",
                success=False,
                message="研究问题不能为空。",
            )
        if not 1 <= limit <= self.config.semantic_scholar_max_results:
            return LiteratureResearchResult(
                original_query=clean_question,
                optimized_query="",
                papers=(),
                review="",
                references=(),
                mode="输入校验",
                source="",
                success=False,
                message=(
                    "检索条数必须在 1 到 "
                    f"{self.config.semantic_scholar_max_results} 之间。"
                ),
            )

        if self.llm_service.is_configured:
            try:
                return self._run_with_function_calling(
                    clean_question,
                    limit=limit,
                )
            except LLMConfigurationError as exc:
                # 网络或服务商兼容性问题不应阻止公开文献检索。
                result = self._run_without_llm(
                    clean_question,
                    limit=limit,
                    extra_warnings=(
                        f"Function Calling 未完成，已切换到规则检索模式：{exc}",
                    ),
                )
                return result

        return self._run_without_llm(clean_question, limit=limit)

    def _run_with_function_calling(
        self,
        research_question: str,
        *,
        limit: int,
    ) -> LiteratureResearchResult:
        """执行标准两阶段 Function Calling 流程。"""

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "你是体育科研文献检索 Agent。先理解用户研究问题，"
                    "把研究对象、核心变量、结局指标和可能同义词整理成"
                    "适合 Semantic Scholar 的检索式，然后必须调用 "
                    "semantic_scholar_search 工具。不要直接编造检索结果。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"研究问题：{research_question}\n"
                    f"请最多检索 {limit} 条论文，并只通过工具获取真实元数据。"
                ),
            },
        ]
        assistant_message = self.llm_service.chat_messages(
            messages,
            tools=[SEMANTIC_SCHOLAR_TOOL],
            tool_choice="auto",
            temperature=0.1,
        )
        tool_calls = assistant_message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            raise LLMConfigurationError(
                "模型未返回 semantic_scholar_search 工具调用。"
            )

        selected_call = _select_semantic_scholar_call(tool_calls)
        arguments = _parse_tool_arguments(selected_call)
        optimized_query = str(arguments.get("query") or "").strip()
        if not optimized_query:
            raise LLMConfigurationError("模型返回的检索关键词为空。")
        requested_limit = _safe_limit(arguments.get("limit"), limit)

        search_warnings: list[str] = []
        papers, source = self._search_public_literature(
            optimized_query,
            limit=requested_limit,
            warnings=search_warnings,
        )
        references = tuple(format_gbt7714(paper) for paper in papers)

        if not papers:
            review = _build_empty_review(optimized_query)
            return LiteratureResearchResult(
                original_query=research_question,
                optimized_query=optimized_query,
                papers=(),
                review=review,
                references=(),
                mode="Function Calling",
                source=source,
                warnings=tuple(search_warnings),
                message="公开来源没有返回可核验论文，未生成综述或虚构引用。",
            )

        messages.append(_sanitize_assistant_message(assistant_message))
        messages.append(
            {
                "role": "tool",
                "tool_call_id": str(selected_call.get("id") or ""),
                "name": "semantic_scholar_search",
                "content": json.dumps(
                    [paper.to_dict() for paper in papers],
                    ensure_ascii=False,
                ),
            }
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "请只依据上面的工具结果写中文文献综述和研究现状。"
                    "每个关键判断使用 [P1]、[P2] 标注论文，不得补造论文、"
                    "DOI、期刊、页码或实验数据。摘要不足时明确写“证据不足”。"
                    "请按“研究主题、研究现状、主要发现、局限与空白”组织。"
                ),
            }
        )

        review = ""
        review_warning = ""
        try:
            review_message = self.llm_service.chat_messages(
                messages,
                temperature=0.2,
            )
            review = _message_text(review_message)
        except LLMConfigurationError as exc:
            review_warning = f"综述生成调用失败，已使用规则整理：{exc}"

        if not review:
            review = _build_rule_based_review(papers, optimized_query)
            review_warning = review_warning or (
                "模型未返回综述内容，已使用规则整理。"
            )
        if review_warning:
            search_warnings.append(review_warning)

        return LiteratureResearchResult(
            original_query=research_question,
            optimized_query=optimized_query,
            papers=papers,
            review=review,
            references=references,
            mode=(
                "大模型综述（公开元数据）"
                if not review_warning
                else "Function Calling + 规则整理"
            ),
            source=source,
            warnings=tuple(search_warnings),
            message=(
                "综述只依据 Semantic Scholar/arXiv 返回的元数据和摘要，"
                "不能替代阅读论文全文。"
            ),
        )

    def _run_without_llm(
        self,
        research_question: str,
        *,
        limit: int,
        extra_warnings: tuple[str, ...] = (),
    ) -> LiteratureResearchResult:
        """无模型密钥时保留真实检索，并明确使用规则整理。"""

        optimized_query = _rule_based_query(research_question)
        warnings = list(extra_warnings)
        papers, source = self._search_public_literature(
            optimized_query,
            limit=limit,
            warnings=warnings,
        )
        references = tuple(format_gbt7714(paper) for paper in papers)
        if not papers:
            return LiteratureResearchResult(
                original_query=research_question,
                optimized_query=optimized_query,
                papers=(),
                review=_build_empty_review(optimized_query),
                references=(),
                mode="规则检索",
                source=source,
                warnings=tuple(warnings),
                message=(
                    "未获得可核验元数据；本轮不生成综述或引用。"
                ),
            )

        warnings.append(
            "当前为规则整理模式，只复述检索到的元数据和摘要，"
            "不会生成大模型式综合分析。"
        )
        return LiteratureResearchResult(
            original_query=research_question,
            optimized_query=optimized_query,
            papers=papers,
            review=_build_rule_based_review(papers, optimized_query),
            references=references,
            mode="规则整理（非大模型综述）",
            source=source,
            warnings=tuple(warnings),
            message=(
                "检索结果来自公开学术来源；摘要可能由来源缺失或截断，"
                "正式引用前必须打开原文核验。"
            ),
        )

    def _search_public_literature(
        self,
        query: str,
        *,
        limit: int,
        warnings: list[str],
    ) -> tuple[tuple[SemanticScholarPaper, ...], str]:
        """优先 Semantic Scholar，失败或无结果时回退 arXiv。"""

        try:
            papers = self.search_tool(query, limit)
            if papers:
                return papers, "Semantic Scholar"
            warnings.append(
                "Semantic Scholar 未返回匹配论文，已尝试 arXiv 公共预印本。"
            )
        except (SemanticScholarError, ValueError, OSError) as exc:
            warnings.append(
                f"Semantic Scholar 检索失败，已尝试 arXiv：{exc}"
            )

        try:
            papers = self.preprint_tool(query, limit)
            if papers:
                return papers, "arXiv"
            warnings.append("arXiv 也未返回匹配预印本。")
        except (ArxivSearchError, ValueError, OSError) as exc:
            warnings.append(f"arXiv 检索失败：{exc}")
        return (), "Semantic Scholar / arXiv"


def _select_semantic_scholar_call(
    tool_calls: list[Any],
) -> dict[str, Any]:
    """找到第一个 semantic_scholar_search 调用。"""

    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if (
            isinstance(function, dict)
            and function.get("name") == "semantic_scholar_search"
        ):
            return call
    raise LLMConfigurationError(
        "工具调用中没有 semantic_scholar_search。"
    )


def _parse_tool_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    """兼容字符串和对象两种工具参数格式。"""

    function = tool_call.get("function")
    if not isinstance(function, dict):
        raise LLMConfigurationError("工具调用缺少 function 字段。")
    raw_arguments = function.get("arguments")
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if not isinstance(raw_arguments, str):
        raise LLMConfigurationError("工具参数不是有效 JSON。")
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise LLMConfigurationError(
            "模型返回的工具参数不是有效 JSON。"
        ) from exc
    if not isinstance(parsed, dict):
        raise LLMConfigurationError("工具参数必须是 JSON 对象。")
    return parsed


def _safe_limit(value: Any, maximum: int) -> int:
    """限制模型请求的返回条数，防止异常超量调用。"""

    try:
        requested = int(value)
    except (TypeError, ValueError):
        requested = maximum
    return max(1, min(requested, maximum))


def _sanitize_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    """保留 Function Calling 所需字段，避免传入无用响应字段。"""

    sanitized: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content") or "",
    }
    if message.get("tool_calls"):
        sanitized["tool_calls"] = message["tool_calls"]
    return sanitized


def _message_text(message: dict[str, Any]) -> str:
    """从字符串或多段内容中提取 assistant 文本。"""

    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict)
        ).strip()
    return str(content or "").strip()


def _rule_based_query(question: str) -> str:
    """无模型时的保守检索式清理，不臆造英文术语。"""

    removals = ("研究", "关系研究", "影响因素研究")
    normalized = " ".join(question.split())
    for token in removals:
        normalized = normalized.replace(token, " ")
    return " ".join(normalized.split()) or question


def _build_empty_review(query: str) -> str:
    """没有真实结果时只说明检索状态，不生成学术结论。"""

    return (
        "## 检索结果\n\n"
        f"本轮使用检索式“{query}”查询公开学术来源，但没有获得"
        "可核验论文元数据。\n\n"
        "因此当前证据不足以生成研究现状、共识或研究空白。"
        "请调整关键词、稍后重试，或在获得真实论文后阅读全文核验。"
    )


def _build_rule_based_review(
    papers: tuple[SemanticScholarPaper, ...],
    query: str,
) -> str:
    """无模型时仅整理真实元数据和摘要，不推断摘要外结论。"""

    years = [paper.year for paper in papers if paper.year.isdigit()]
    venues = Counter(
        paper.venue for paper in papers if paper.venue.strip()
    )
    year_range = (
        f"{min(years)}-{max(years)}"
        if years
        else "年份信息不足"
    )
    venue_text = (
        "、".join(
            f"{name}（{count} 篇）"
            for name, count in venues.most_common(3)
        )
        if venues
        else "来源未提供期刊或会议信息"
    )

    blocks = [
        "## 研究现状概览（规则整理）",
        "",
        f"- 检索式：{query}",
        f"- 纳入真实元数据：{len(papers)} 篇",
        f"- 年份范围：{year_range}",
        f"- 主要来源：{venue_text}",
        "",
        "## 逐篇摘要线索",
    ]
    for index, paper in enumerate(papers, start=1):
        citation_line = "，".join(
            item
            for item in (
                paper.year or "年份信息缺失",
                paper.venue or "来源信息缺失",
            )
            if item
        )
        abstract = paper.abstract.strip()
        abstract_text = (
            abstract[:800] + ("..." if len(abstract) > 800 else "")
            if abstract
            else "来源未提供摘要，当前不能概括该文研究结论。"
        )
        blocks.extend(
            [
                "",
                f"### [P{index}] {paper.title}",
                f"{paper.authors_text}｜{citation_line}",
                "",
                abstract_text,
            ]
        )
    blocks.extend(
        [
            "",
            "## 证据边界",
            "",
            "以上内容只是对公开元数据和摘要的逐篇整理，"
            "不是系统综述。研究设计、样本、统计结果、局限和学术争议"
            "必须回到论文全文核验，当前不据此宣称形成学术共识。",
        ]
    )
    return "\n".join(blocks)
