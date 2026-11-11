"""把真实论文元数据整理为 GB/T 7714 风格引用。"""

from __future__ import annotations

from datetime import date
from typing import Protocol


class CitablePaper(Protocol):
    """引用格式化器需要的最小论文字段。"""

    title: str
    authors: tuple[str, ...]
    year: str
    venue: str
    doi: str
    pdf_url: str
    url: str
    publication_types: tuple[str, ...]


def format_gbt7714(
    paper: CitablePaper,
    *,
    accessed_date: date | None = None,
) -> str:
    """根据已有元数据生成 GB/T 7714 风格引用。

    缺失元数据会写成“信息缺失”，不会猜测期刊、卷期、页码或作者。
    该格式化器覆盖期刊、会议和电子资源三种常见类型，正式提交前仍应人工核对。
    """

    authors = _format_authors(paper.authors)
    title = paper.title.strip().rstrip(".") or "题名信息缺失"
    year = paper.year.strip() or "年份信息缺失"
    venue = paper.venue.strip()
    doi_suffix = f" DOI: {paper.doi}." if paper.doi.strip() else ""
    types = {item.lower() for item in paper.publication_types}

    if venue:
        document_type = "[C]" if _is_conference(types, venue) else "[J]"
        return (
            f"{authors}. {title}{document_type}. "
            f"{venue}, {year}.{doi_suffix}"
        ).strip()

    access_date = accessed_date or date.today()
    resource_url = paper.pdf_url.strip() or paper.url.strip()
    url_suffix = resource_url or "URL 信息缺失"
    return (
        f"{authors}. {title}[EB/OL]. "
        f"({year})[{access_date.isoformat()}]. {url_suffix}.{doi_suffix}"
    ).strip()


def _format_authors(authors: tuple[str, ...]) -> str:
    """按现有作者顺序格式化，超过三人时保留前三人和“等”。"""

    clean_authors = tuple(
        " ".join(author.split()) for author in authors if author.strip()
    )
    if not clean_authors:
        return "作者信息缺失"
    if len(clean_authors) <= 3:
        return ", ".join(clean_authors)
    return f"{', '.join(clean_authors[:3])}, 等"


def _is_conference(types: set[str], venue: str) -> bool:
    """只根据明确元数据判断会议，不从题名猜测。"""

    combined = f"{' '.join(types)} {venue}".lower()
    return any(
        marker in combined
        for marker in ("conference", "proceedings", "symposium", "会议")
    )
