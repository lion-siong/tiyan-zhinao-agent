"""Semantic Scholar 公开论文元数据检索工具。

本模块只查询论文标题、作者、年份、摘要、DOI 和开放获取链接，不抓取知网等
受限来源，也不自动下载第三方论文全文。
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from config import Settings, settings as default_settings


SEMANTIC_SCHOLAR_FIELDS = (
    "paperId,title,authors,year,abstract,externalIds,openAccessPdf,"
    "url,venue,publicationTypes,publicationDate"
)


class SemanticScholarError(RuntimeError):
    """Semantic Scholar 不可用、限流或返回格式异常。"""


@dataclass(frozen=True)
class SemanticScholarPaper:
    """Semantic Scholar 返回的一条可核验论文元数据。"""

    paper_id: str
    title: str
    authors: tuple[str, ...]
    year: str
    abstract: str
    doi: str
    pdf_url: str
    url: str
    venue: str
    publication_types: tuple[str, ...]
    source: str = "Semantic Scholar"

    @property
    def authors_text(self) -> str:
        """返回适合界面展示的作者文本，不补造缺失作者。"""

        return "、".join(self.authors) if self.authors else "作者信息缺失"

    def to_dict(self) -> dict[str, Any]:
        """转换为便于模型和界面使用的普通字典。"""

        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "abstract": self.abstract,
            "doi": self.doi,
            "pdf_url": self.pdf_url,
            "url": self.url,
            "venue": self.venue,
            "publication_types": list(self.publication_types),
            "source": self.source,
        }


def semantic_scholar_search(
    query: str,
    limit: int = 10,
    *,
    config: Settings | None = None,
) -> tuple[SemanticScholarPaper, ...]:
    """检索 Semantic Scholar 公开论文元数据。

    参数：
        query: 已优化或用户输入的研究主题。
        limit: 返回论文条数，默认 10。
        config: 测试或调用方可注入的配置，常规使用全局 .env 配置。

    函数不会下载 PDF，只保留来源提供的开放获取链接。
    """

    clean_query = query.strip()
    if not clean_query:
        raise SemanticScholarError("检索主题不能为空。")

    current_config = config or default_settings
    maximum = current_config.semantic_scholar_max_results
    if not 1 <= limit <= maximum:
        raise SemanticScholarError(
            f"返回条数必须在 1 到 {maximum} 之间。"
        )

    parameters = {
        "query": clean_query,
        "limit": str(limit),
        "fields": SEMANTIC_SCHOLAR_FIELDS,
    }
    base_url = current_config.semantic_scholar_base_url.rstrip("/")
    url = f"{base_url}/paper/search?{urllib.parse.urlencode(parameters)}"
    payload = _read_json(
        url,
        timeout=current_config.semantic_scholar_timeout_seconds,
        api_key=current_config.semantic_scholar_api_key,
    )

    records = payload.get("data")
    if not isinstance(records, list):
        raise SemanticScholarError(
            "Semantic Scholar 返回结果缺少 data 列表。"
        )

    papers: list[SemanticScholarPaper] = []
    for record in records:
        paper = _parse_paper(record)
        if paper is not None:
            papers.append(paper)

    # API 偶尔返回重复记录，使用 DOI 或题名进行稳定去重。
    seen: set[str] = set()
    unique: list[SemanticScholarPaper] = []
    for paper in papers:
        key = (
            f"doi:{paper.doi.lower()}"
            if paper.doi
            else f"title:{_title_key(paper.title)}"
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(paper)
    return tuple(unique)


def _read_json(
    url: str,
    *,
    timeout: int,
    api_key: str,
) -> dict[str, Any]:
    """读取公开 JSON 接口，并把网络异常转换为清晰提示。"""

    headers = {
        "Accept": "application/json",
        "User-Agent": "ResearchMind/1.0 (university competition project)",
    }
    if api_key:
        headers["x-api-key"] = api_key
    request = urllib.request.Request(url, headers=headers, method="GET")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise SemanticScholarError(
                "Semantic Scholar 请求过于频繁，请稍后重试；"
                "可在 .env 配置 SEMANTIC_SCHOLAR_API_KEY 提升额度。"
            ) from exc
        raise SemanticScholarError(
            f"Semantic Scholar 返回 HTTP {exc.code}。"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SemanticScholarError(
            f"无法连接 Semantic Scholar：{exc}"
        ) from exc

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SemanticScholarError(
            "Semantic Scholar 返回的不是有效 JSON。"
        ) from exc
    if not isinstance(data, dict):
        raise SemanticScholarError(
            "Semantic Scholar 返回的数据格式不正确。"
        )
    return data


def _parse_paper(record: Any) -> SemanticScholarPaper | None:
    """把 API 字典映射为稳定数据对象，缺失字段保持为空。"""

    if not isinstance(record, dict):
        return None
    title = _clean_text(record.get("title"))
    if not title:
        return None

    authors: list[str] = []
    raw_authors = record.get("authors")
    if isinstance(raw_authors, list):
        for author in raw_authors:
            if not isinstance(author, dict):
                continue
            name = _clean_text(author.get("name"))
            if name:
                authors.append(name)

    external_ids = record.get("externalIds")
    doi = ""
    if isinstance(external_ids, dict):
        doi = _normalize_doi(str(external_ids.get("DOI") or ""))

    open_access_pdf = record.get("openAccessPdf")
    pdf_url = ""
    if isinstance(open_access_pdf, dict):
        pdf_url = str(open_access_pdf.get("url") or "").strip()

    raw_types = record.get("publicationTypes")
    publication_types = (
        tuple(
            str(item).strip()
            for item in raw_types
            if str(item).strip()
        )
        if isinstance(raw_types, list)
        else ()
    )

    year_value = record.get("year")
    year = str(year_value).strip() if year_value is not None else ""
    return SemanticScholarPaper(
        paper_id=str(record.get("paperId") or "").strip(),
        title=title,
        authors=tuple(authors),
        year=year,
        abstract=_clean_text(record.get("abstract")),
        doi=doi,
        pdf_url=pdf_url,
        url=str(record.get("url") or "").strip(),
        venue=_clean_text(record.get("venue")),
        publication_types=publication_types,
    )


def _clean_text(value: Any) -> str:
    """清理模型或网页展示不需要的连续空白。"""

    return " ".join(str(value or "").split()).strip()


def _normalize_doi(value: str) -> str:
    """只保留 DOI 标识本身，避免拼出不存在的新标识。"""

    return re.sub(
        r"^https?://(?:dx\.)?doi\.org/",
        "",
        value.strip(),
        flags=re.IGNORECASE,
    ).rstrip(".,;")


def _title_key(title: str) -> str:
    """生成题名去重键。"""

    return re.sub(r"\W+", "", title.lower())
