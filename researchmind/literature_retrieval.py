"""多数据源文献检索编排层。

本模块在现有 Semantic Scholar / arXiv / OpenAlex / Crossref 链路基础上做增量
升级，统一接入五类公开学术数据源：

    Semantic Scholar → arXiv → OpenAlex → Crossref → PubMed

设计原则（对应竞赛验收要求）：

* 统一输出 :class:`researchmind.models.CandidatePaper`，供现有论文库导入链路直接消费；
* 单源失败被隔离，不会中断整体检索任务；
* 识别 429/403/500/502/503/504，429 优先遵守 ``Retry-After``，其余按指数退避加随机抖动重试；
* 进程内关键词缓存，避免重复请求相同检索式；
* 每个数据源记录状态、返回数量、耗时、失败原因与重试建议；
* 统一做标题非空校验、DOI 格式校验、字段完整度分级与多级去重；
* 不伪造论文、DOI、作者、期刊或引用次数；未取到全文时只标记题录/摘要。

本模块只读取公开元数据，不绕过付费墙，不通过换 IP / 代理轮换规避限流。
"""

from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Iterable, Sequence

from config import Settings, settings as default_settings
from researchmind.models import CandidatePaper

# 可重试的 HTTP 状态码：限流、封禁与服务端临时故障。
RETRYABLE_STATUS_CODES = frozenset({429, 403, 500, 502, 503, 504})
# 不重试的确定性错误（参数错误、未授权、不存在等）。
NON_RETRYABLE_STATUS_CODES = frozenset({400, 401, 404, 410, 422})

SOURCE_ORDER = ("Semantic Scholar", "arXiv", "OpenAlex", "Crossref", "PubMed")

# 体育科研领域常见中英同义词，用于主题扩展；仅用于补充检索式与相关性提示，
# 不会替换原始检索词，也不会据此编造论文。
_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("体育锻炼", "身体活动", "运动干预", "physical exercise", "physical activity", "exercise intervention"),
    ("体质健康", "体能", "身体素质", "physical fitness", "physical health", "sport performance"),
    ("大学生", "高校学生", "college students", "undergraduate", "university students"),
    ("青少年", "儿童", "adolescent", "youth", "children"),
    ("运动生理学", "有氧能力", "心肺耐力", "exercise physiology", "aerobic capacity", "cardiorespiratory fitness"),
    ("运动表现", "竞技表现", "训练效果", "athletic performance", "training effect", "sport performance"),
)


class RetrievalSourceError(RuntimeError):
    """单个数据源检索失败时抛出的结构化错误。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.retryable = retryable


@dataclass(frozen=True)
class SourceReport:
    """单个数据源本次检索的状态与统计。"""

    source: str
    status: str  # success / empty / failed / disabled
    requested: int = 0
    returned: int = 0
    elapsed_ms: int = 0
    http_status: int | None = None
    error: str = ""
    retryable: bool = False
    attempts: int = 0
    suggestion: str = ""

    @property
    def ok(self) -> bool:
        return self.status in {"success", "empty"}


@dataclass(frozen=True)
class RetrievalReport:
    """一次多源检索的汇总结果与可核验统计。"""

    query: str
    expanded_queries: tuple[str, ...]
    candidates: tuple[CandidatePaper, ...]
    sources: tuple[SourceReport, ...]
    total_raw: int
    total_unique: int
    verified_doi_count: int
    incomplete_count: int
    unverifiable_count: int
    abstract_count: int
    fulltext_count: int
    all_failed: bool
    warnings: tuple[str, ...]

    def source_report(self, name: str) -> SourceReport | None:
        for report in self.sources:
            if report.source == name:
                return report
        return None


# --------------------------------------------------------------------------- #
# 主题扩展与规范化
# --------------------------------------------------------------------------- #


def expand_query(query: str) -> tuple[str, ...]:
    """返回主检索词与命中的同义词扩展词。

    结果去重、保序；主检索词始终放在首位。
    """

    clean = query.strip()
    if not clean:
        return ()
    expanded: list[str] = [clean]
    lowered = clean.lower()
    for group in _SYNONYM_GROUPS:
        if any(token in clean or token.lower() in lowered for token in group):
            for token in group:
                if token not in expanded:
                    expanded.append(token)
    return tuple(expanded)


def normalize_doi(value: str) -> str:
    """规范化 DOI（去协议前缀、去首尾标点、转小写），仅用于本地校验与去重。"""

    text = re.sub(
        r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)",
        "",
        str(value or "").strip(),
        flags=re.IGNORECASE,
    )
    return text.rstrip(".,;，；。").strip().casefold()


def normalize_title(value: str) -> str:
    """规范化题名：忽略全半角、大小写、空白与标点差异。"""

    text = re.sub(r"\s+", " ", str(value or "").strip().casefold())
    return "".join(ch for ch in text if ch.isalnum())


_DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$")


def is_valid_doi(doi: str) -> bool:
    """校验 DOI 格式是否符合 ``10.后缀/标识`` 的基本形态。"""

    normalized = normalize_doi(doi)
    if not normalized:
        return False
    return bool(_DOI_PATTERN.match(normalized))


# --------------------------------------------------------------------------- #
# 本地缓存（进程内，带 TTL）
# --------------------------------------------------------------------------- #


class _RetrievalCache:
    """按 ``源 + 检索式 + 条数 + 日期范围`` 缓存候选结果，避免重复请求。"""

    def __init__(self, ttl_seconds: int) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[float, tuple[CandidatePaper, ...]]] = {}

    def get(self, key: str) -> tuple[CandidatePaper, ...] | None:
        if self._ttl <= 0:
            return None
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, payload = entry
        if time.time() > expires_at:
            self._store.pop(key, None)
            return None
        return payload

    def set(self, key: str, payload: tuple[CandidatePaper, ...]) -> None:
        if self._ttl <= 0:
            return
        self._store[key] = (time.time() + self._ttl, payload)

    def clear(self) -> None:
        self._store.clear()


# --------------------------------------------------------------------------- #
# 韧性 HTTP 客户端（供 PubMed 等直连接口使用）
# --------------------------------------------------------------------------- #


class _Throttle:
    """限制相邻请求的最小间隔，避免短时间集中打满数据源。"""

    def __init__(self, interval_seconds: float) -> None:
        self._interval = max(interval_seconds, 0.0)
        self._last_at = 0.0

    def wait(self) -> None:
        if self._interval <= 0:
            return
        now = time.time()
        gap = now - self._last_at
        if gap < self._interval:
            time.sleep(self._interval - gap)
        self._last_at = time.time()


def _retry_after_seconds(
    headers: Any, fallback: float, *, cap: float
) -> float:
    raw = headers.get("Retry-After") if headers is not None else None
    try:
        delay = float(raw) if raw else fallback
    except (TypeError, ValueError):
        delay = fallback
    return min(max(delay, 0.2), cap)


def _backoff_delay(attempt: int, base: float, cap: float, jitter: float) -> float:
    """指数退避 base * 2**attempt，再叠加 0~jitter 随机抖动，封顶 cap。"""

    exponential = base * (2 ** max(attempt, 0))
    delay = min(exponential + random.uniform(0.0, jitter), cap)
    return delay


def http_get_json(
    url: str,
    *,
    headers: dict[str, str],
    config: Settings,
    throttle: _Throttle,
) -> dict[str, Any]:
    """带错误码识别、退避抖动与重试的 JSON GET。

    抛 :class:`RetrievalSourceError`，由编排层隔离到单条源报告。
    """

    request = urllib.request.Request(url, headers=headers, method="GET")
    last_error: RetrievalSourceError | None = None

    for attempt in range(config.retrieval_max_retries + 1):
        throttle.wait()
        started = time.time()
        try:
            with urllib.request.urlopen(
                request, timeout=config.retrieval_timeout_seconds
            ) as response:
                body = response.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            if not isinstance(data, dict):
                raise RetrievalSourceError("接口返回格式不正确。")
            return data
        except urllib.error.HTTPError as exc:
            retryable = exc.code in RETRYABLE_STATUS_CODES
            retry_after = (
                _retry_after_seconds(
                    exc.headers,
                    config.retrieval_base_backoff_seconds,
                    cap=config.retrieval_max_backoff_seconds,
                )
                if retryable
                else None
            )
            last_error = RetrievalSourceError(
                f"HTTP {exc.code}",
                status_code=exc.code,
                retry_after=retry_after,
                retryable=retryable,
            )
            if retryable and attempt < config.retrieval_max_retries:
                time.sleep(retry_after if retry_after else _backoff_delay(
                    attempt,
                    config.retrieval_base_backoff_seconds,
                    config.retrieval_max_backoff_seconds,
                    config.retrieval_base_backoff_seconds,
                ))
                continue
            raise last_error from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = RetrievalSourceError(
                f"网络或解析失败：{exc}", retryable=True
            )
            if attempt < config.retrieval_max_retries:
                time.sleep(_backoff_delay(
                    attempt,
                    config.retrieval_base_backoff_seconds,
                    config.retrieval_max_backoff_seconds,
                    config.retrieval_base_backoff_seconds,
                ))
                continue
            raise last_error from exc

    raise last_error or RetrievalSourceError("未知检索失败。")


# --------------------------------------------------------------------------- #
# 真实性校验与分级
# --------------------------------------------------------------------------- #


def quality_tier(candidate: CandidatePaper) -> str:
    """根据字段完整度与 DOI 可核验性分级。

    * ``verified``：DOI 格式有效，且作者、年份齐全；
    * ``incomplete``：题录可读，但缺 DOI 或缺作者/年份等关键元数据；
    * ``unverifiable``：无 DOI 且题名残缺或年份缺失，无法可靠核验。
    """

    title_ok = bool(candidate.title.strip())
    has_doi = is_valid_doi(candidate.doi)
    has_authors = bool(candidate.authors)
    has_year = bool(candidate.year.strip()) and candidate.year.strip() != "待核对"

    if not title_ok:
        return "unverifiable"
    if has_doi and has_authors and has_year:
        return "verified"
    if not has_doi and (not has_year or len(candidate.title.strip()) < 6):
        return "unverifiable"
    return "incomplete"


def _dedup_key(candidate: CandidatePaper) -> str:
    """DOI 优先去重；无 DOI 时用规范化题名 + 年份辅助去重。"""

    doi = normalize_doi(candidate.doi)
    if doi:
        return f"doi:{doi}"
    return f"title:{normalize_title(candidate.title)}|year:{candidate.year.strip()}"


def _merge_preferred(
    primary: CandidatePaper, other: CandidatePaper
) -> CandidatePaper:
    """合并跨源重复记录：用字段更完整的一条补齐缺失项。"""

    def pick(a: str, b: str) -> str:
        return a.strip() or b.strip()

    authors = primary.authors or other.authors
    keywords = tuple(dict.fromkeys((*primary.keywords, *other.keywords)))
    return CandidatePaper(
        candidate_id=primary.candidate_id,
        title=pick(primary.title, other.title),
        authors=authors,
        year=pick(primary.year, other.year),
        journal=pick(primary.journal, other.journal),
        doi=pick(primary.doi, other.doi),
        source_url=pick(primary.source_url, other.source_url),
        abstract=pick(primary.abstract, other.abstract),
        keywords=keywords[:8],
        source=primary.source,
        pdf_url=pick(primary.pdf_url, other.pdf_url),
        is_open_access=primary.is_open_access or other.is_open_access,
    )


def deduplicate(candidates: Iterable[CandidatePaper]) -> tuple[CandidatePaper, ...]:
    """跨源多级去重合并，保留字段更完整的记录。"""

    merged: dict[str, CandidatePaper] = {}
    order: list[str] = []
    for candidate in candidates:
        if not candidate.title.strip():
            continue
        key = _dedup_key(candidate)
        if key not in merged:
            merged[key] = candidate
            order.append(key)
            continue
        existing = merged[key]
        # 优先保留信息更全的一条作为主体。
        if _completeness(candidate) > _completeness(existing):
            merged[key] = _merge_preferred(candidate, existing)
        else:
            merged[key] = _merge_preferred(existing, candidate)
    return tuple(merged[key] for key in order)


def _completeness(candidate: CandidatePaper) -> int:
    score = 0
    if candidate.doi.strip():
        score += 2
    if candidate.authors:
        score += 1
    if candidate.year.strip() and candidate.year.strip() != "待核对":
        score += 1
    if candidate.journal.strip() and candidate.journal.strip() != "待核对":
        score += 1
    if candidate.abstract.strip():
        score += 1
    return score


def _candidate_id(doi: str, title: str) -> str:
    import hashlib

    key = normalize_doi(doi) or normalize_title(title)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"candidate-{digest}"


# --------------------------------------------------------------------------- #
# 数据源适配器：统一签名 (query, limit, date_from, date_to, config) -> 候选元组
# --------------------------------------------------------------------------- #


def _adapt_semantic_scholar(
    query: str,
    limit: int,
    date_from: str,
    date_to: str,
    config: Settings,
) -> tuple[CandidatePaper, ...]:
    from tools import semantic_scholar as ss_tool

    papers = ss_tool.semantic_scholar_search(query, limit, config=config)
    return tuple(
        CandidatePaper(
            candidate_id=_candidate_id(paper.doi, paper.title),
            title=paper.title,
            authors=paper.authors,
            year=str(paper.year or ""),
            journal=paper.venue or "待核对",
            doi=normalize_doi(paper.doi),
            source_url=paper.url,
            abstract=paper.abstract,
            keywords=tuple(paper.publication_types or ()),
            source="Semantic Scholar",
            pdf_url=paper.pdf_url,
            is_open_access=False,
        )
        for paper in papers
    )


def _adapt_arxiv(
    query: str,
    limit: int,
    date_from: str,
    date_to: str,
    config: Settings,
) -> tuple[CandidatePaper, ...]:
    from tools import arxiv_search as arxiv_tool

    papers = arxiv_tool.arxiv_search(query, limit, config=config)
    return tuple(
        CandidatePaper(
            candidate_id=_candidate_id(paper.doi, paper.title),
            title=paper.title,
            authors=paper.authors,
            year=str(paper.year or ""),
            journal="arXiv 预印本",
            doi=normalize_doi(paper.doi),
            source_url=paper.url,
            abstract=paper.abstract,
            keywords=tuple(paper.publication_types or ()),
            source="arXiv",
            pdf_url=paper.pdf_url,
            is_open_access=False,
        )
        for paper in papers
    )


def _adapt_openalex(
    query: str,
    limit: int,
    date_from: str,
    date_to: str,
    config: Settings,
) -> tuple[CandidatePaper, ...]:
    from researchmind import collectors

    return collectors._search_openalex(
        query,
        limit=limit,
        date_from=date_from,
        date_to=date_to,
        mailto=config.openalex_mailto,
        timeout=config.retrieval_timeout_seconds,
    )


def _adapt_crossref(
    query: str,
    limit: int,
    date_from: str,
    date_to: str,
    config: Settings,
) -> tuple[CandidatePaper, ...]:
    from researchmind import collectors

    return collectors._search_crossref(
        query,
        limit=limit,
        date_from=date_from,
        date_to=date_to,
        timeout=config.retrieval_timeout_seconds,
    )


def _adapt_pubmed(
    query: str,
    limit: int,
    date_from: str,
    date_to: str,
    config: Settings,
) -> tuple[CandidatePaper, ...]:
    """通过 NCBI E-utilities 检索 PubMed 题录。

    esummary 仅提供题录信息，不含摘要正文；此处不二次抓取摘要，
    避免把题录误标为摘要或全文。PubMed 是英文医学库，中文主题会
    先映射到领域英文同义词再检索，避免中文词被退化为近期泛文献。
    """

    base = config.pubmed_base_url.rstrip("/")
    common: dict[str, str] = {
        "db": "pubmed",
        "retmode": "json",
        "tool": "researchmind",
    }
    if config.openalex_mailto:
        common["email"] = config.openalex_mailto
    if config.pubmed_api_key:
        common["api_key"] = config.pubmed_api_key

    english_terms = [
        token for token in expand_query(query) if re.search(r"[A-Za-z]", token)
    ]
    pubmed_term = " OR ".join(english_terms) if english_terms else query

    esearch_params = {
        **common,
        "term": pubmed_term,
        "retmax": str(limit),
        "datetype": "pdat",
        "mindate": date_from,
        "maxdate": date_to,
    }
    esearch_url = (
        base + "/esearch.fcgi?" + urllib.parse.urlencode(esearch_params)
    )
    esearch = http_get_json(
        esearch_url,
        headers={"Accept": "application/json"},
        config=config,
        throttle=_PUBMED_THROTTLE,
    )
    ids = esearch.get("esearchresult", {}).get("idlist")
    if not isinstance(ids, list) or not ids:
        return ()

    esummary_params = {**common, "id": ",".join(str(uid) for uid in ids)}
    esummary_url = (
        base + "/esummary.fcgi?" + urllib.parse.urlencode(esummary_params)
    )
    esummary = http_get_json(
        esummary_url,
        headers={"Accept": "application/json"},
        config=config,
        throttle=_PUBMED_THROTTLE,
    )
    result = esummary.get("result")
    if not isinstance(result, dict):
        return ()

    candidates: list[CandidatePaper] = []
    for uid in ids:
        item = result.get(str(uid))
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        authors = tuple(
            str(author.get("name", "")).strip()
            for author in item.get("authors", [])
            if isinstance(author, dict) and str(author.get("name", "")).strip()
        )
        pubdate = str(item.get("pubdate") or "")
        year = pubdate[:4] if len(pubdate) >= 4 and pubdate[:4].isdigit() else "待核对"
        journal = str(
            item.get("fulljournalname") or item.get("source") or "待核对"
        ).strip()
        doi = ""
        for article in item.get("articleids", []):
            if isinstance(article, dict) and article.get("idtype") == "doi":
                doi = normalize_doi(str(article.get("value") or ""))
                break
        keywords = tuple(
            str(keyword).strip()
            for keyword in item.get("keywords", [])
            if str(keyword).strip()
        )[:8]
        candidates.append(
            CandidatePaper(
                candidate_id=_candidate_id(doi, title),
                title=title,
                authors=authors,
                year=year,
                journal=journal,
                doi=doi,
                source_url=f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                abstract="",
                keywords=keywords,
                source="PubMed",
                pdf_url="",
                is_open_access=False,
            )
        )
    return tuple(candidates)


_PUBMED_THROTTLE = _Throttle(0.0)

_ADAPTERS: dict[str, Callable[..., tuple[CandidatePaper, ...]]] = {
    "Semantic Scholar": _adapt_semantic_scholar,
    "arXiv": _adapt_arxiv,
    "OpenAlex": _adapt_openalex,
    "Crossref": _adapt_crossref,
    "PubMed": _adapt_pubmed,
}

# 这些工具内部已自带限流/网络重试，编排层不再叠加外层重试，避免超时被放大。
_INNER_RETRY_SOURCES = frozenset({"Semantic Scholar", "arXiv"})


def _suggestion(status_code: int | None, error: str) -> str:
    if status_code == 429:
        return "请求过于频繁，已按 Retry-After 退避；可稍后重试，或在 .env 配置对应 API Key 提升额度。"
    if status_code == 403:
        return "访问被拒绝（匿名额度受限或来源不开放），已自动切换其他数据源。"
    if status_code in {500, 502, 503, 504}:
        return "数据源服务端临时故障，已退避重试；可稍后再次检索。"
    if "timed out" in error.lower() or "timeout" in error.lower():
        return "当前网络连接该来源超时，已自动切换其他数据源。"
    if "无法连接" in error or "网络" in error:
        return "当前网络无法连接该来源，已自动切换其他数据源。"
    return "该数据源本次不可用，已由其他来源接管；可稍后重试。"


class MultiSourceRetriever:
    """按固定顺序调度多源检索，单源失败隔离并汇总统一结果。"""

    def __init__(
        self,
        config: Settings | None = None,
        *,
        cache: _RetrievalCache | None = None,
        adapters: dict[str, Callable[..., tuple[CandidatePaper, ...]]]
        | None = None,
        source_order: Sequence[str] | None = None,
    ) -> None:
        self.config = config or default_settings
        self.cache = cache or _RetrievalCache(
            self.config.retrieval_cache_ttl_seconds
        )
        self._adapters = dict(_ADAPTERS)
        if adapters:
            self._adapters.update(adapters)
        self._order = tuple(source_order or SOURCE_ORDER)
        _PUBMED_THROTTLE._interval = max(
            self.config.retrieval_request_interval_seconds, 0.0
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> RetrievalReport:
        clean_query = query.strip()
        if not clean_query:
            return RetrievalReport(
                query="",
                expanded_queries=(),
                candidates=(),
                sources=(),
                total_raw=0,
                total_unique=0,
                verified_doi_count=0,
                incomplete_count=0,
                unverifiable_count=0,
                abstract_count=0,
                fulltext_count=0,
                all_failed=True,
                warnings=("检索主题不能为空。",),
            )
        if not 1 <= limit <= self.config.semantic_scholar_max_results:
            limit = min(max(limit, 1), self.config.semantic_scholar_max_results)

        start_date = date_from or self.config.researchmind_date_from
        end_date = date_to or date.today().isoformat()
        expanded = expand_query(clean_query)

        raw: list[CandidatePaper] = []
        reports: list[SourceReport] = []
        warnings: list[str] = []

        for source in self._order:
            adapter = self._adapters.get(source)
            if adapter is None:
                continue
            cache_key = f"{source}|{clean_query}|{limit}|{start_date}|{end_date}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                raw.extend(cached)
                reports.append(
                    SourceReport(
                        source=source,
                        status="success" if cached else "empty",
                        requested=limit,
                        returned=len(cached),
                        elapsed_ms=0,
                        attempts=1,
                        suggestion="缓存命中，未重复请求。" if cached else "",
                    )
                )
                continue

            candidates, report = self._invoke_source(
                source, adapter, clean_query, limit, start_date, end_date
            )
            reports.append(report)
            if report.ok:
                self.cache.set(cache_key, candidates)
                raw.extend(candidates)
            else:
                warnings.append(f"{source}：{report.error or report.suggestion}")

        unique = deduplicate(raw)
        tiers = [quality_tier(candidate) for candidate in unique]
        verified_doi_count = sum(is_valid_doi(c.doi) for c in unique)
        abstract_count = sum(bool(c.abstract.strip()) for c in unique)
        fulltext_count = sum(c.has_open_pdf for c in unique)

        return RetrievalReport(
            query=clean_query,
            expanded_queries=expanded,
            candidates=unique,
            sources=tuple(reports),
            total_raw=len(raw),
            total_unique=len(unique),
            verified_doi_count=verified_doi_count,
            incomplete_count=tiers.count("incomplete"),
            unverifiable_count=tiers.count("unverifiable"),
            abstract_count=abstract_count,
            fulltext_count=fulltext_count,
            all_failed=bool(reports) and all(
                report.status == "failed" for report in reports
            ),
            warnings=tuple(warnings),
        )

    def _invoke_source(
        self,
        source: str,
        adapter: Callable[..., tuple[CandidatePaper, ...]],
        query: str,
        limit: int,
        date_from: str,
        date_to: str,
    ) -> tuple[tuple[CandidatePaper, ...], SourceReport]:
        started = time.time()
        attempts = 0
        last_error = ""
        retryable = False
        http_status: int | None = None
        max_retries = (
            0 if source in _INNER_RETRY_SOURCES else self.config.retrieval_max_retries
        )

        for attempt in range(max_retries + 1):
            attempts = attempt + 1
            try:
                candidates = adapter(
                    query, limit, date_from, date_to, self.config
                )
                elapsed = int((time.time() - started) * 1000)
                status = "success" if candidates else "empty"
                suggestion = (
                    ""
                    if candidates
                    else "该来源未返回匹配论文。"
                )
                return candidates, SourceReport(
                    source=source,
                    status=status,
                    requested=limit,
                    returned=len(candidates),
                    elapsed_ms=elapsed,
                    attempts=attempts,
                    suggestion=suggestion,
                )
            except RetrievalSourceError as exc:
                last_error = str(exc)
                http_status = exc.status_code
                retryable = exc.retryable
                if exc.retryable and attempt < max_retries:
                    delay = exc.retry_after or _backoff_delay(
                        attempt,
                        self.config.retrieval_base_backoff_seconds,
                        self.config.retrieval_max_backoff_seconds,
                        self.config.retrieval_base_backoff_seconds,
                    )
                    time.sleep(delay)
                    continue
                break
            except Exception as exc:  # 单源异常隔离，不中断整体检索
                last_error = str(exc)
                retryable = True
                if attempt < max_retries:
                    time.sleep(_backoff_delay(
                        attempt,
                        self.config.retrieval_base_backoff_seconds,
                        self.config.retrieval_max_backoff_seconds,
                        self.config.retrieval_base_backoff_seconds,
                    ))
                    continue
                break

        elapsed = int((time.time() - started) * 1000)
        return (), SourceReport(
            source=source,
            status="failed",
            requested=limit,
            returned=0,
            elapsed_ms=elapsed,
            http_status=http_status,
            error=last_error,
            retryable=retryable,
            attempts=attempts,
            suggestion=_suggestion(http_status, last_error),
        )
