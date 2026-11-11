"""arXiv 公开预印本检索工具。

仅在 Semantic Scholar 无结果或不可用时作为公开预印本回退来源。这里只读取
官方 Atom 元数据，不绕过出版方权限。
"""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from config import Settings, settings as default_settings
from tools.semantic_scholar import SemanticScholarPaper


ATOM_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_NAMESPACE = {"arxiv": "http://arxiv.org/schemas/atom"}


class ArxivSearchError(RuntimeError):
    """arXiv 接口不可用或返回格式异常。"""


def arxiv_search(
    query: str,
    limit: int = 10,
    *,
    config: Settings | None = None,
) -> tuple[SemanticScholarPaper, ...]:
    """检索 arXiv 官方公开预印本元数据。"""

    clean_query = query.strip()
    if not clean_query:
        raise ArxivSearchError("检索主题不能为空。")
    if not 1 <= limit <= 20:
        raise ArxivSearchError("arXiv 返回条数必须在 1 到 20 之间。")

    current_config = config or default_settings
    parameters = {
        "search_query": f"all:{clean_query}",
        "start": "0",
        "max_results": str(limit),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = (
        f"{current_config.arxiv_api_url.rstrip('/')}?"
        f"{urllib.parse.urlencode(parameters)}"
    )
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/atom+xml",
            "User-Agent": "ResearchMind/1.0 (university competition project)",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=current_config.semantic_scholar_timeout_seconds,
        ) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise ArxivSearchError(
            f"arXiv 返回 HTTP {exc.code}。"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ArxivSearchError(f"无法连接 arXiv：{exc}") from exc

    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise ArxivSearchError("arXiv 返回的 Atom 内容无法解析。") from exc

    papers: list[SemanticScholarPaper] = []
    for entry in root.findall("atom:entry", ATOM_NAMESPACE):
        paper = _parse_entry(entry)
        if paper is not None:
            papers.append(paper)
    return tuple(papers)


def _parse_entry(entry: ET.Element) -> SemanticScholarPaper | None:
    """从一条 Atom entry 提取真实元数据。"""

    title = _node_text(entry, "atom:title")
    if not title:
        return None

    authors = tuple(
        name
        for author in entry.findall("atom:author", ATOM_NAMESPACE)
        if (name := _node_text(author, "atom:name"))
    )
    url = _node_text(entry, "atom:id")
    pdf_url = ""
    for link in entry.findall("atom:link", ATOM_NAMESPACE):
        if link.attrib.get("title") == "pdf":
            pdf_url = link.attrib.get("href", "").strip()
            break

    doi = _node_text(entry, "arxiv:doi", ARXIV_NAMESPACE)
    journal_ref = _node_text(entry, "arxiv:journal_ref", ARXIV_NAMESPACE)
    published = _node_text(entry, "atom:published")
    return SemanticScholarPaper(
        paper_id=url.rsplit("/", 1)[-1] if url else "",
        title=title,
        authors=authors,
        year=published[:4] if len(published) >= 4 else "",
        abstract=_node_text(entry, "atom:summary"),
        doi=doi,
        pdf_url=pdf_url,
        url=url,
        venue=journal_ref or "arXiv",
        publication_types=("preprint",),
        source="arXiv",
    )


def _node_text(
    parent: ET.Element,
    path: str,
    namespaces: dict[str, str] | None = None,
) -> str:
    """读取一个 Atom/arXiv 文本节点并清理空白。"""

    namespace_map = namespaces or ATOM_NAMESPACE
    node = parent.find(path, namespace_map)
    return " ".join(node.text.split()) if node is not None and node.text else ""
