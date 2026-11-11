"""合法学术元数据检索与开放 PDF 下载。

OpenAlex 和 Crossref 只提供公开元数据，不绕过付费墙。只有元数据明确给出
开放获取 PDF 地址时才允许下载；下载后仍由现有 PDF 导入器提取书目信息和正文。
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from typing import Any

from config import Settings, settings as default_settings
from researchmind.models import CandidatePaper


class LiteratureCollectionError(RuntimeError):
    """学术来源不可用、限流或返回格式异常时抛出的错误。"""


@dataclass(frozen=True)
class CollectionResult:
    """一次学术检索的结果。"""

    source: str
    query: str
    candidates: tuple[CandidatePaper, ...]
    warnings: tuple[str, ...]
    date_from: str


class LiteratureCollector:
    """从 OpenAlex 检索论文元数据，Crossref 作为失败回退。"""

    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or default_settings

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        date_from: str | None = None,
    ) -> CollectionResult:
        """检索真实论文元数据，并按 DOI/题名去重。"""

        clean_query = query.strip()
        if not clean_query:
            raise LiteratureCollectionError("检索主题不能为空。")
        if not 1 <= limit <= 25:
            raise LiteratureCollectionError("单次检索数量必须在 1 到 25 之间。")
        start_date = date_from or self.config.researchmind_date_from
        _validate_iso_date(start_date)

        warnings: list[str] = []
        try:
            candidates = _search_openalex(
                clean_query,
                limit=limit,
                date_from=start_date,
                mailto=self.config.openalex_mailto,
                timeout=self.config.llm_timeout_seconds,
            )
            source = "OpenAlex"
        except LiteratureCollectionError as exc:
            warnings.append(f"OpenAlex 检索失败，已尝试 Crossref：{exc}")
            candidates = _search_crossref(
                clean_query,
                limit=limit,
                date_from=start_date,
                timeout=self.config.llm_timeout_seconds,
            )
            source = "Crossref"

        deduplicated = _deduplicate_candidates(candidates)
        if not deduplicated:
            warnings.append(
                "来源没有返回可核验的论文元数据，未向论文库写入任何记录。"
            )
        return CollectionResult(
            source=source,
            query=clean_query,
            candidates=deduplicated,
            warnings=tuple(warnings),
            date_from=start_date,
        )

    def download_open_pdf(
        self,
        candidate: CandidatePaper,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        """下载开放获取 PDF，并验证文件头和大小。"""

        if not candidate.has_open_pdf:
            raise LiteratureCollectionError(
                "该候选论文没有可确认的开放获取 PDF 地址。"
            )
        limit = max_bytes or self.config.max_pdf_size_mb * 1024 * 1024
        request = urllib.request.Request(
            candidate.pdf_url,
            headers={
                "User-Agent": _user_agent(self.config.openalex_mailto),
                "Accept": "application/pdf",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.llm_timeout_seconds,
            ) as response:
                content_type = response.headers.get("Content-Type", "")
                data = response.read(limit + 1)
        except urllib.error.HTTPError as exc:
            raise LiteratureCollectionError(
                f"开放 PDF 下载返回 HTTP {exc.code}。"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LiteratureCollectionError(
                f"无法下载开放 PDF：{exc}"
            ) from exc

        if len(data) > limit:
            raise LiteratureCollectionError(
                f"PDF 超过 {self.config.max_pdf_size_mb} MB 限制。"
            )
        if not data.startswith(b"%PDF"):
            raise LiteratureCollectionError(
                "下载内容不是 PDF，可能被来源页面重定向或需要登录。"
            )
        if "pdf" not in content_type.lower() and not candidate.pdf_url.lower().endswith(
            ".pdf"
        ):
            raise LiteratureCollectionError(
                "来源返回的文件类型无法确认是 PDF。"
            )
        return data


def _search_openalex(
    query: str,
    *,
    limit: int,
    date_from: str,
    mailto: str,
    timeout: int,
) -> tuple[CandidatePaper, ...]:
    end_date = date.today().isoformat()
    parameters = {
        "search": query,
        "filter": (
            f"from_publication_date:{date_from},"
            f"to_publication_date:{end_date}"
        ),
        "sort": "relevance_score:desc",
        "per-page": str(limit),
    }
    if mailto:
        parameters["mailto"] = mailto
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(parameters)
    payload = _read_json(url, timeout=timeout, mailto=mailto)
    results = payload.get("results")
    if not isinstance(results, list):
        raise LiteratureCollectionError("OpenAlex 返回结果缺少 results 列表。")

    candidates: list[CandidatePaper] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("display_name") or item.get("title") or "").strip()
        if not title:
            continue
        doi = _normalize_doi(str(item.get("doi") or ""))
        year = str(item.get("publication_year") or "").strip()
        authors = _openalex_authors(item)
        journal, pdf_url = _openalex_location(item)
        abstract = _reconstruct_abstract(item.get("abstract_inverted_index"))
        keywords = tuple(
            str(concept.get("display_name", "")).strip()
            for concept in item.get("concepts", [])
            if isinstance(concept, dict)
            and str(concept.get("display_name", "")).strip()
        )[:8]
        open_access = item.get("open_access")
        is_open_access = bool(
            isinstance(open_access, dict) and open_access.get("is_oa")
        )
        if not pdf_url:
            is_open_access = False
        candidate_id = _candidate_id(doi, title)
        candidates.append(
            CandidatePaper(
                candidate_id=candidate_id,
                title=title,
                authors=authors,
                year=year,
                journal=journal,
                doi=doi,
                source_url=(
                    f"https://doi.org/{doi}"
                    if doi
                    else str(item.get("id") or "https://openalex.org")
                ),
                abstract=abstract,
                keywords=keywords,
                source="OpenAlex",
                pdf_url=pdf_url,
                is_open_access=is_open_access,
            )
        )
    return tuple(candidates)


def _search_crossref(
    query: str,
    *,
    limit: int,
    date_from: str,
    timeout: int,
) -> tuple[CandidatePaper, ...]:
    parameters = {
        "query": query,
        "filter": f"from-pub-date:{date_from},type:journal-article",
        "rows": str(limit),
        "select": (
            "DOI,title,author,published,container-title,abstract,"
            "URL,link,subject,is-referenced-by-count"
        ),
    }
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(parameters)
    payload = _read_json(url, timeout=timeout, mailto="")
    message = payload.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("items"), list):
        raise LiteratureCollectionError("Crossref 返回结果格式不正确。")

    candidates: list[CandidatePaper] = []
    for item in message["items"]:
        if not isinstance(item, dict):
            continue
        titles = item.get("title")
        title = (
            str(titles[0]).strip()
            if isinstance(titles, list) and titles
            else ""
        )
        if not title:
            continue
        doi = _normalize_doi(str(item.get("DOI") or ""))
        year = _crossref_year(item)
        authors = tuple(
            " ".join(
                part
                for part in (
                    str(author.get("given", "")).strip(),
                    str(author.get("family", "")).strip(),
                )
                if part
            )
            for author in item.get("author", [])
            if isinstance(author, dict)
            and (
                str(author.get("given", "")).strip()
                or str(author.get("family", "")).strip()
            )
        )
        journals = item.get("container-title")
        journal = (
            str(journals[0]).strip()
            if isinstance(journals, list) and journals
            else "待核对"
        )
        pdf_url = ""
        for link in item.get("link", []):
            if not isinstance(link, dict):
                continue
            content_type = str(link.get("content-type", "")).lower()
            link_url = str(link.get("URL", "")).strip()
            if link_url and ("pdf" in content_type or link_url.lower().endswith(".pdf")):
                pdf_url = link_url
                break
        subject = item.get("subject")
        keywords = tuple(
            str(value).strip()
            for value in subject
            if str(value).strip()
        )[:8] if isinstance(subject, list) else ()
        candidates.append(
            CandidatePaper(
                candidate_id=_candidate_id(doi, title),
                title=title,
                authors=authors,
                year=year,
                journal=journal,
                doi=doi,
                source_url=str(item.get("URL") or ""),
                abstract=_strip_markup(str(item.get("abstract") or "")),
                keywords=keywords,
                source="Crossref",
                pdf_url=pdf_url,
                # Crossref 的 link 字段只表示内容链接，并不能证明开放许可。
                is_open_access=False,
            )
        )
    return tuple(candidates)


def _read_json(url: str, *, timeout: int, mailto: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": _user_agent(mailto),
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise LiteratureCollectionError(
            f"学术元数据接口返回 HTTP {exc.code}。"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LiteratureCollectionError(
            f"无法连接学术元数据接口：{exc}"
        ) from exc
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise LiteratureCollectionError(
            "学术元数据接口返回的不是 JSON。"
        ) from exc
    if not isinstance(data, dict):
        raise LiteratureCollectionError("学术元数据接口返回格式不正确。")
    return data


def _openalex_authors(item: dict[str, Any]) -> tuple[str, ...]:
    authors: list[str] = []
    for authorship in item.get("authorships", []):
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author")
        if isinstance(author, dict):
            name = str(author.get("display_name", "")).strip()
            if name:
                authors.append(name)
    return tuple(authors)


def _openalex_location(item: dict[str, Any]) -> tuple[str, str]:
    for key in ("best_oa_location", "primary_location"):
        location = item.get(key)
        if not isinstance(location, dict):
            continue
        pdf_url = str(location.get("pdf_url") or "").strip()
        source = location.get("source")
        journal = (
            str(source.get("display_name", "")).strip()
            if isinstance(source, dict)
            else ""
        )
        if pdf_url or journal:
            return journal or "待核对", pdf_url
    return "待核对", ""


def _reconstruct_abstract(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    positioned: list[tuple[int, str]] = []
    for word, positions in value.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int):
                positioned.append((position, str(word)))
    positioned.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positioned).strip()


def _crossref_year(item: dict[str, Any]) -> str:
    published = item.get("published")
    if isinstance(published, dict):
        parts = published.get("date-parts")
        if (
            isinstance(parts, list)
            and parts
            and isinstance(parts[0], list)
            and parts[0]
        ):
            return str(parts[0][0])
    return "待核对"


def _deduplicate_candidates(
    candidates: tuple[CandidatePaper, ...],
) -> tuple[CandidatePaper, ...]:
    seen: set[str] = set()
    output: list[CandidatePaper] = []
    for candidate in candidates:
        key = (
            f"doi:{candidate.doi.lower()}"
            if candidate.doi
            else f"title:{re.sub(r'\W+', '', candidate.title.lower())}"
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
    return tuple(output)


def _normalize_doi(value: str) -> str:
    return re.sub(
        r"^https?://(?:dx\.)?doi\.org/",
        "",
        value.strip(),
        flags=re.IGNORECASE,
    ).rstrip(".,;")


def _candidate_id(doi: str, title: str) -> str:
    import hashlib

    key = doi.lower() if doi else re.sub(r"\W+", "", title.lower())
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"candidate-{digest}"


def _strip_markup(value: str) -> str:
    try:
        root = ET.fromstring(f"<root>{value}</root>")
        return " ".join("".join(root.itertext()).split())
    except ET.ParseError:
        return re.sub(r"<[^>]+>", " ", value).strip()


def _validate_iso_date(value: str) -> None:
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise LiteratureCollectionError(
            "检索起始日期必须是 YYYY-MM-DD 格式。"
        ) from exc


def _user_agent(mailto: str) -> str:
    contact = f" ({mailto})" if mailto else ""
    return f"ResearchMind/1.0{contact}"
