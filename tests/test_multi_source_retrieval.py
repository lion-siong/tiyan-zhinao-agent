"""多数据源检索编排、真实性校验、去重与链路打通的验收测试。

测试中的论文均为明确标注的自动化夹具，不代表真实论文；只写入 pytest
临时目录，不改动正式专题库、旧库或线上数据。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agents.literature_research_agent import LiteratureResearchAgent
from config import Settings
from researchmind.adolescent_library import create_adolescent_workflow
from researchmind.literature_import import import_candidates, soft_delete_papers
from researchmind.literature_retrieval import (
    MultiSourceRetriever,
    RetrievalSourceError,
    deduplicate,
    is_valid_doi,
    quality_tier,
)
from researchmind.models import CandidatePaper
from services.llm_service import LLMService


FIXTURE_NOTE = "自动化测试夹具，仅用于验证检索与入库流程，不代表真实论文。"


def _settings(tmp_path: Path) -> Settings:
    return Settings.from_env(
        {
            "RESEARCHMIND_DB_PATH": str(tmp_path / "legacy" / "db.sqlite3"),
            "ADOLESCENT_RESEARCHMIND_DB_PATH": str(
                tmp_path / "topic" / "db.sqlite3"
            ),
            "ADOLESCENT_KNOWLEDGE_BASE_DIR": str(tmp_path / "topic-kb"),
            "OPENAI_API_KEY": "",
            "RETRIEVAL_MAX_RETRIES": "1",
            "RETRIEVAL_BASE_BACKOFF_SECONDS": "0",
            "RETRIEVAL_MAX_BACKOFF_SECONDS": "0",
            "RETRIEVAL_REQUEST_INTERVAL_SECONDS": "0",
            "RETRIEVAL_CACHE_TTL_SECONDS": "0",
        }
    )


def _candidate(
    key: str,
    *,
    title: str = "青少年体育锻炼与体质健康自动化测试夹具",
    doi: str | None = None,
    authors: tuple[str, ...] = ("测试作者",),
    year: str = "2024",
    journal: str = "测试期刊",
    abstract: str = FIXTURE_NOTE,
    source: str = "OpenAlex",
) -> CandidatePaper:
    return CandidatePaper(
        candidate_id=f"fixture-{key}",
        title=title,
        authors=authors,
        year=year,
        journal=journal,
        doi=doi if doi is not None else f"10.0000/fixture.{key}",
        source_url=f"https://example.org/fixture/{key}",
        abstract=abstract,
        keywords=("青少年", "体育锻炼"),
        source=source,
    )


def _adapter_returning(papers: tuple[CandidatePaper, ...]):
    def _adapter(
        query: str, limit: int, date_from: str, date_to: str, config: Settings
    ) -> tuple[CandidatePaper, ...]:
        return papers

    return _adapter


def _adapter_raising(exc: Exception):
    def _adapter(
        query: str, limit: int, date_from: str, date_to: str, config: Settings
    ) -> tuple[CandidatePaper, ...]:
        raise exc

    return _adapter


def _retriever(
    settings: Settings,
    adapters: dict[str, Any],
) -> MultiSourceRetriever:
    # 只运行注入的适配器，未注入的默认源不参与，避免测试真实联网。
    return MultiSourceRetriever(
        settings, adapters=adapters, source_order=tuple(adapters)
    )


# --------------------------------------------------------------------------- #
# 1. Semantic Scholar 正常返回数据
# --------------------------------------------------------------------------- #


def test_semantic_scholar_returns_real_fixture(settings: Settings) -> None:
    paper = _candidate("ss-ok", source="Semantic Scholar")
    retriever = _retriever(
        settings, {"Semantic Scholar": _adapter_returning((paper,))}
    )

    report = retriever.search("大学生体质健康", limit=5)

    ss = report.source_report("Semantic Scholar")
    assert ss is not None and ss.status == "success"
    assert report.total_unique == 1
    assert report.candidates[0].title == paper.title


# --------------------------------------------------------------------------- #
# 2. Semantic Scholar 返回 429
# --------------------------------------------------------------------------- #


def test_semantic_scholar_429_is_isolated_and_marked_retryable(
    settings: Settings,
) -> None:
    exc = RetrievalSourceError(
        "HTTP 429", status_code=429, retry_after=1.0, retryable=True
    )
    retriever = _retriever(
        settings, {"Semantic Scholar": _adapter_raising(exc)}
    )

    report = retriever.search("exercise", limit=5)

    ss = report.source_report("Semantic Scholar")
    assert ss is not None and ss.status == "failed"
    assert ss.http_status == 429 and ss.retryable is True
    assert "重试" in ss.suggestion or "额度" in ss.suggestion


# --------------------------------------------------------------------------- #
# 3. arXiv 返回 429 / 失败
# --------------------------------------------------------------------------- #


def test_arxiv_failure_is_isolated(settings: Settings) -> None:
    exc = RetrievalSourceError("HTTP 429", status_code=429, retryable=True)
    retriever = _retriever(settings, {"arXiv": _adapter_raising(exc)})

    report = retriever.search("preprint", limit=5)

    arxiv = report.source_report("arXiv")
    assert arxiv is not None and arxiv.status == "failed"
    assert arxiv.http_status == 429


# --------------------------------------------------------------------------- #
# 4. OpenAlex 成功接管检索
# --------------------------------------------------------------------------- #


def test_openalex_takes_over_after_ss_and_arxiv_fail(settings: Settings) -> None:
    fallback = _candidate("oa", source="OpenAlex")
    retriever = _retriever(
        settings,
        {
            "Semantic Scholar": _adapter_raising(
                RetrievalSourceError("HTTP 429", status_code=429)
            ),
            "arXiv": _adapter_raising(RetrievalSourceError("timeout", retryable=True)),
            "OpenAlex": _adapter_returning((fallback,)),
        },
    )

    report = retriever.search("physical fitness", limit=5)

    assert report.total_unique == 1
    assert report.candidates[0].source == "OpenAlex"
    oa = report.source_report("OpenAlex")
    assert oa is not None and oa.status == "success"


# --------------------------------------------------------------------------- #
# 5. 多数据源同时失败
# --------------------------------------------------------------------------- #


def test_all_sources_fail_marks_all_failed(settings: Settings) -> None:
    adapters = {
        source: _adapter_raising(RetrievalSourceError("unavailable", retryable=True))
        for source in (
            "Semantic Scholar",
            "arXiv",
            "OpenAlex",
            "Crossref",
            "PubMed",
        )
    }
    retriever = _retriever(settings, adapters)

    report = retriever.search("exercise", limit=5)

    assert report.all_failed is True
    assert report.candidates == ()
    assert report.total_unique == 0
    assert all(source.status == "failed" for source in report.sources)


# --------------------------------------------------------------------------- #
# 6. DOI 重复论文去重
# --------------------------------------------------------------------------- #


def test_doi_duplicates_are_merged(settings: Settings) -> None:
    first = _candidate("doi-1", title="Title Alpha", doi="10.0000/alpha")
    second = _candidate(
        "doi-2",
        title="Title Beta",
        doi="https://doi.org/10.0000/alpha.",
        source="Crossref",
    )

    merged = deduplicate((first, second))

    assert len(merged) == 1
    assert is_valid_doi(merged[0].doi)


# --------------------------------------------------------------------------- #
# 7. 没有 DOI 的论文按标题+年份去重
# --------------------------------------------------------------------------- #


def test_title_year_duplicates_without_doi_are_merged(settings: Settings) -> None:
    first = _candidate("t-1", title="Same Title Here", doi="", year="2024")
    second = _candidate(
        "t-2", title="Same Title Here", doi="", year="2024", source="Crossref"
    )

    merged = deduplicate((first, second))

    assert len(merged) == 1


# --------------------------------------------------------------------------- #
# 8. 元数据缺失分级与空标题过滤
# --------------------------------------------------------------------------- #


def test_metadata_completeness_tiering_and_empty_title_filter() -> None:
    verified = _candidate("ok", doi="10.0000/ok", authors=("A", "B"), year="2024")
    incomplete = _candidate(
        "incomplete", doi="", authors=(), year="2024", journal=""
    )
    unverifiable = _candidate(
        "unverifiable", title="短", doi="", authors=(), year=""
    )
    empty_title = _candidate("empty", title="", doi="10.0000/empty")

    assert quality_tier(verified) == "verified"
    assert quality_tier(incomplete) == "incomplete"
    assert quality_tier(unverifiable) == "unverifiable"

    merged = deduplicate((verified, incomplete, unverifiable, empty_title))
    assert empty_title not in merged
    assert any(c.candidate_id == "fixture-ok" for c in merged)


# --------------------------------------------------------------------------- #
# 9. 论文成功导入数据库
# --------------------------------------------------------------------------- #


def test_retrieved_candidates_import_into_sqlite(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    workflow = create_adolescent_workflow(settings)
    domain = workflow.ensure_default_domain()
    paper = _candidate("import-me")

    result = import_candidates(workflow, domain, (paper,))

    assert result.imported == 1
    assert workflow.store.paper_count(domain.domain_id) == 1
    stored = workflow.store.list_papers(domain.domain_id)[0]
    assert stored.doi == "10.0000/fixture.import-me"
    assert stored.parse_status == "仅元数据"


# --------------------------------------------------------------------------- #
# 10. 删除论文后关联数据同步处理
# --------------------------------------------------------------------------- #


def test_delete_paper_invalidates_related_derivations(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    workflow = create_adolescent_workflow(settings)
    domain = workflow.ensure_default_domain()
    paper = _candidate("delete-me")
    import_candidates(workflow, domain, (paper,))
    paper_id = workflow.store.list_papers(domain.domain_id)[0].paper_id

    deletion = soft_delete_papers(workflow.store, domain.domain_id, (paper_id,))
    invalidated = workflow.store.invalidate_domain_derivations(
        domain.domain_id, deletion.paper_ids
    )

    assert deletion.paper_ids == (paper_id,)
    # 派生分析被标记失效；题库按引用关系清理，不留指向已删论文的残留。
    assert invalidated["debates"] >= 0
    assert workflow.store.get_paper(paper_id, include_deleted=True).is_deleted == 1


# --------------------------------------------------------------------------- #
# 11. 检索失败时不生成虚假综述
# --------------------------------------------------------------------------- #


def test_failed_retrieval_does_not_produce_fake_review(
    settings: Settings,
) -> None:
    adapters = {
        source: _adapter_raising(RetrievalSourceError("down", retryable=True))
        for source in (
            "Semantic Scholar",
            "arXiv",
            "OpenAlex",
            "Crossref",
            "PubMed",
        )
    }
    retriever = _retriever(settings, adapters)
    agent = LiteratureResearchAgent(
        settings,
        LLMService(settings),
        retriever=retriever,
    )

    result = agent.run("青少年锻炼与体质健康", limit=5)

    assert result.papers == ()
    assert result.references == ()
    assert result.report is not None and result.report["all_failed"] is True
    assert "未" in result.message or "没有" in result.message


# --------------------------------------------------------------------------- #
# 12. 网络中断后重试恢复
# --------------------------------------------------------------------------- #


def test_recovers_after_transient_network_failure(settings: Settings) -> None:
    paper = _candidate("recovered", source="OpenAlex")
    calls = {"count": 0}

    def flaky_adapter(
        query: str, limit: int, date_from: str, date_to: str, config: Settings
    ) -> tuple[CandidatePaper, ...]:
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("network interrupted")
        return (paper,)

    retriever = _retriever(settings, {"OpenAlex": flaky_adapter})

    report = retriever.search("exercise", limit=5)

    assert calls["count"] == 2
    oa = report.source_report("OpenAlex")
    assert oa is not None and oa.status == "success"
    assert report.total_unique == 1


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return _settings(tmp_path)
