"""公开文献 Agent 与用户 PDF 本地 RAG 的自动化测试。"""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from agents.literature_research_agent import (
    LiteratureResearchAgent,
    _parse_tool_arguments,
)
from config import Settings
from services.llm_service import LLMService
from services.user_paper_rag import (
    USER_PAPER_COLLECTION_NAME,
    UserPaperRAGError,
    UserPaperRAGService,
)
from services.vector_service import VectorStoreError
from tools.arxiv_search import arxiv_search
from tools.citation_formatter import format_gbt7714
from tools.semantic_scholar import (
    SemanticScholarError,
    SemanticScholarPaper,
    semantic_scholar_search,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _paper(
    *,
    title: str = "Exercise behavior and physical fitness",
    authors: tuple[str, ...] = ("Alice Smith", "Bob Lee"),
    year: str = "2025",
    venue: str = "Journal of Sports Research",
    doi: str = "10.1234/example",
) -> SemanticScholarPaper:
    return SemanticScholarPaper(
        paper_id="paper-1",
        title=title,
        authors=authors,
        year=year,
        abstract="A real abstract supplied by the public source.",
        doi=doi,
        pdf_url="https://example.org/open.pdf",
        url="https://example.org/paper",
        venue=venue,
        publication_types=("JournalArticle",),
    )


class _Response:
    """模拟 urllib 的上下文响应对象。"""

    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_semantic_scholar_search_maps_and_deduplicates_real_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = {
        "paperId": "abc",
        "title": "Exercise behavior and physical fitness",
        "authors": [{"name": "Alice Smith"}, {"name": "Bob Lee"}],
        "year": 2025,
        "abstract": "A source-provided abstract.",
        "externalIds": {"DOI": "10.1234/example"},
        "openAccessPdf": {"url": "https://example.org/open.pdf"},
        "url": "https://example.org/paper",
        "venue": "Journal of Sports Research",
        "publicationTypes": ["JournalArticle"],
    }
    payload = {"data": [record, dict(record, paperId="duplicate")]}
    monkeypatch.setattr(
        "tools.semantic_scholar.urllib.request.urlopen",
        lambda request, timeout: _Response(json.dumps(payload)),
    )
    config = Settings.from_env(
        {
            "SEMANTIC_SCHOLAR_BASE_URL": "https://example.org/graph/v1",
            "SEMANTIC_SCHOLAR_MAX_RESULTS": "20",
        }
    )

    results = semantic_scholar_search("exercise fitness", 5, config=config)

    assert len(results) == 1
    assert results[0].title == record["title"]
    assert results[0].authors == ("Alice Smith", "Bob Lee")
    assert results[0].year == "2025"
    assert results[0].doi == "10.1234/example"
    assert results[0].pdf_url == "https://example.org/open.pdf"


def test_semantic_scholar_search_handles_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_rate_limit(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            429,
            "Too Many Requests",
            hdrs=None,
            fp=io.BytesIO(b""),
        )

    monkeypatch.setattr(
        "tools.semantic_scholar.urllib.request.urlopen",
        raise_rate_limit,
    )
    with pytest.raises(SemanticScholarError, match="请求过于频繁"):
        semantic_scholar_search("exercise", 5)


def test_semantic_scholar_search_handles_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_timeout(request, timeout):
        raise TimeoutError("test timeout")

    monkeypatch.setattr(
        "tools.semantic_scholar.urllib.request.urlopen",
        raise_timeout,
    )
    with pytest.raises(SemanticScholarError, match="无法连接"):
        semantic_scholar_search("exercise", 5)


def test_semantic_scholar_rejects_invalid_arguments() -> None:
    with pytest.raises(SemanticScholarError, match="不能为空"):
        semantic_scholar_search("   ", 5)
    with pytest.raises(SemanticScholarError, match="返回条数"):
        semantic_scholar_search("exercise", 0)


def test_arxiv_search_parses_official_atom_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atom = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>https://arxiv.org/abs/2501.00001v1</id>
        <published>2025-01-01T00:00:00Z</published>
        <title>Open preprint title</title>
        <summary>Public preprint abstract.</summary>
        <author><name>Carol Wang</name></author>
        <link title="pdf" href="https://arxiv.org/pdf/2501.00001" />
        <arxiv:doi>10.1234/preprint</arxiv:doi>
        <arxiv:journal_ref>Example Journal</arxiv:journal_ref>
      </entry>
    </feed>
    """
    monkeypatch.setattr(
        "tools.arxiv_search.urllib.request.urlopen",
        lambda request, timeout: _Response(atom),
    )

    results = arxiv_search("open preprint", 1)

    assert len(results) == 1
    assert results[0].source == "arXiv"
    assert results[0].year == "2025"
    assert results[0].pdf_url == "https://arxiv.org/pdf/2501.00001"


def test_gbt7714_uses_only_existing_metadata() -> None:
    journal = _paper()
    no_venue = _paper(
        title="Open repository report",
        venue="",
        doi="",
        authors=("A Author",),
    )
    no_author = _paper(authors=(), doi="")

    journal_reference = format_gbt7714(journal)
    no_venue_reference = format_gbt7714(no_venue)
    no_author_reference = format_gbt7714(no_author)

    assert "[J]" in journal_reference
    assert "Journal of Sports Research" in journal_reference
    assert "[EB/OL]" in no_venue_reference
    assert "https://example.org/open.pdf" in no_venue_reference
    assert no_author_reference.startswith("作者信息缺失")
    assert "卷" not in journal_reference
    assert "页码" not in journal_reference


class _FunctionCallingLLM:
    """按两阶段顺序返回工具调用和最终综述。"""

    is_configured = True

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat_messages(self, messages, **kwargs):
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "kwargs": kwargs,
            }
        )
        if len(self.calls) == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "semantic_scholar_search",
                            "arguments": json.dumps(
                                {
                                    "query": (
                                        "college students physical exercise "
                                        "physical fitness"
                                    ),
                                    "limit": 3,
                                }
                            ),
                        },
                    }
                ],
            }
        return {
            "role": "assistant",
            "content": "研究现状：现有元数据显示该主题受到关注 [P1]。",
        }


def test_agent_optimizes_query_then_calls_tool_and_writes_review() -> None:
    llm = _FunctionCallingLLM()
    captured: dict[str, Any] = {}

    def fake_search(query: str, limit: int):
        captured["query"] = query
        captured["limit"] = limit
        return (_paper(),)

    agent = LiteratureResearchAgent(
        Settings.from_env({"OPENAI_API_KEY": "test-key"}),
        llm_service=llm,
        search_tool=fake_search,
        preprint_tool=lambda query, limit: (),
    )
    result = agent.run(
        "大学生体育锻炼与体质健康的关系研究",
        limit=5,
    )

    assert result.success is True
    assert captured == {
        "query": "college students physical exercise physical fitness",
        "limit": 3,
    }
    assert len(llm.calls) == 2
    assert any(
        message.get("role") == "tool"
        for message in llm.calls[1]["messages"]
    )
    assert result.mode == "大模型综述（公开元数据）"
    assert result.references


def test_tool_argument_parser_accepts_dict_and_rejects_invalid_json() -> None:
    assert _parse_tool_arguments(
        {"function": {"arguments": {"query": "exercise", "limit": 3}}}
    ) == {"query": "exercise", "limit": 3}
    with pytest.raises(Exception, match="有效 JSON"):
        _parse_tool_arguments(
            {"function": {"arguments": "{not-json"}}
        )


def test_agent_without_key_marks_rule_mode_and_does_not_simulate_llm() -> None:
    agent = LiteratureResearchAgent(
        Settings.from_env({"OPENAI_API_KEY": ""}),
        llm_service=LLMService(
            Settings.from_env({"OPENAI_API_KEY": ""})
        ),
        search_tool=lambda query, limit: (_paper(),),
        preprint_tool=lambda query, limit: (),
    )

    result = agent.run("体育锻炼与体质健康研究", limit=3)

    assert result.success is True
    assert result.mode == "规则整理（非大模型综述）"
    assert "规则整理" in result.review
    assert any("规则整理" in warning for warning in result.warnings)


def test_agent_falls_back_to_arxiv_when_semantic_scholar_fails() -> None:
    arxiv_paper = _paper(title="Arxiv result", doi="")
    agent = LiteratureResearchAgent(
        Settings.from_env({"OPENAI_API_KEY": ""}),
        llm_service=LLMService(
            Settings.from_env({"OPENAI_API_KEY": ""})
        ),
        search_tool=lambda query, limit: (_ for _ in ()).throw(
            SemanticScholarError("temporary failure")
        ),
        preprint_tool=lambda query, limit: (arxiv_paper,),
    )

    result = agent.run("exercise fitness", limit=3)

    assert result.success is True
    assert result.source == "arXiv"
    assert result.papers[0].title == "Arxiv result"


def test_agent_falls_back_to_openalex_when_ss_and_arxiv_unavailable() -> None:
    from dataclasses import replace

    fallback_paper = replace(
        _paper(title="OpenAlex result", doi="10.1234/oa"),
        source="OpenAlex",
    )
    agent = LiteratureResearchAgent(
        Settings.from_env({"OPENAI_API_KEY": ""}),
        llm_service=LLMService(
            Settings.from_env({"OPENAI_API_KEY": ""})
        ),
        search_tool=lambda query, limit: (_ for _ in ()).throw(
            SemanticScholarError("429 rate limited")
        ),
        preprint_tool=lambda query, limit: (),
        fallback_tool=lambda query, limit: (fallback_paper,),
    )

    result = agent.run("exercise fitness", limit=3)

    assert result.success is True
    assert result.source == "OpenAlex"
    assert result.papers[0].title == "OpenAlex result"


def test_arxiv_uses_independent_timeout_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["timeout"] = timeout
        return _Response("<feed xmlns='http://www.w3.org/2005/Atom'></feed>")

    monkeypatch.setattr(
        "tools.arxiv_search.urllib.request.urlopen", fake_urlopen
    )
    config = Settings.from_env({"ARXIV_TIMEOUT_SECONDS": "7"})

    arxiv_search("open preprint", 1, config=config)

    assert captured["timeout"] == 7


class _FixedUserPaperEncoder:
    """用户论文 Chroma 测试使用的确定性编码器。"""

    def encode_documents(self, texts):
        return [self._encode(text) for text in texts]

    def encode_query(self, text: str):
        return self._encode(text)

    @staticmethod
    def _encode(text: str) -> list[float]:
        normalized = text.lower()
        if "aerobic" in normalized or "oxygen" in normalized:
            return [1.0, 0.0, 0.0]
        if "strength" in normalized:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def _pdf_bytes(text: str) -> bytes:
    fitz = pytest.importorskip("pymupdf")
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    payload = document.tobytes()
    document.close()
    return payload


def _user_service(tmp_path: Path) -> UserPaperRAGService:
    config = Settings.from_env(
        {
            "OPENAI_API_KEY": "",
            "EMBEDDING_MODEL": "fake-user-e5",
        }
    )
    return UserPaperRAGService(
        config,
        base_dir=tmp_path / "user_papers",
        llm_service=LLMService(config),
        encoder=_FixedUserPaperEncoder(),
    )


def test_user_pdf_rag_persists_isolated_chroma_and_page_evidence(
    tmp_path: Path,
) -> None:
    service = _user_service(tmp_path)
    pdf = _pdf_bytes(
        "Aerobic fitness is measured by maximum oxygen uptake in college students."
    )

    result = service.ingest_pdf("aerobic-study.pdf", pdf)
    records = service.list_papers()
    answer = service.answer(
        "What measures aerobic fitness?",
        (result.record.paper_key,),
        top_k=3,
    )

    assert result.created is True
    assert result.record.chunk_count >= 1
    assert result.record.stored_path.is_file()
    assert len(records) == 1
    assert service.vector_store.collection_name == USER_PAPER_COLLECTION_NAME
    assert service.vector_store.path == (tmp_path / "user_papers" / "chroma")
    assert answer.success is True
    assert answer.evidence
    assert answer.evidence[0].chunk.document_name == "aerobic-study.pdf"
    assert answer.evidence[0].chunk.page_number == 1
    assert "oxygen" in answer.evidence[0].chunk.text.lower()
    service.close()


def test_user_pdf_rag_deduplicates_content_and_separates_same_name(
    tmp_path: Path,
) -> None:
    service = _user_service(tmp_path)
    aerobic_pdf = _pdf_bytes("Aerobic oxygen uptake study.")
    strength_pdf = _pdf_bytes("Strength training study.")

    first = service.ingest_pdf("same-name.pdf", aerobic_pdf)
    duplicate = service.ingest_pdf("renamed-copy.pdf", aerobic_pdf)
    second = service.ingest_pdf("same-name.pdf", strength_pdf)

    assert first.created is True
    assert duplicate.created is False
    assert duplicate.record.paper_key == first.record.paper_key
    assert second.created is True
    assert second.record.paper_key != first.record.paper_key
    assert len(service.list_papers()) == 2
    service.close()


def test_user_pdf_rag_refuses_unsupported_or_empty_files(
    tmp_path: Path,
) -> None:
    service = _user_service(tmp_path)

    with pytest.raises(UserPaperRAGError, match="只接受 PDF"):
        service.ingest_pdf("paper.txt", b"text")
    with pytest.raises(UserPaperRAGError, match="空文件"):
        service.ingest_pdf("paper.pdf", b"")
    service.close()


def test_user_pdf_rag_refuses_answer_without_evidence(
    tmp_path: Path,
) -> None:
    service = _user_service(tmp_path)
    result = service.ingest_pdf(
        "fitness.pdf",
        _pdf_bytes("Aerobic oxygen uptake in college students."),
    )

    answer = service.answer(
        "quantum chip manufacturing",
        (result.record.paper_key,),
    )

    assert answer.mode == "证据不足"
    assert answer.evidence == ()
    assert "不足以支持结论" in answer.answer
    service.close()


class _FailingUserVectorStore:
    """模拟向量模型不可用，验证 BM25 回退。"""

    collection_name = USER_PAPER_COLLECTION_NAME

    def search(self, question: str, *, top_k: int):
        raise VectorStoreError("fake vector model unavailable")


def test_user_pdf_rag_falls_back_to_bm25_when_vector_fails(
    tmp_path: Path,
) -> None:
    config = Settings.from_env({"OPENAI_API_KEY": ""})
    service = UserPaperRAGService(
        config,
        base_dir=tmp_path / "user_papers",
        llm_service=LLMService(config),
        vector_store=_FailingUserVectorStore(),  # type: ignore[arg-type]
    )
    result = service.ingest_pdf(
        "fitness.pdf",
        _pdf_bytes("Aerobic oxygen uptake in college students."),
        sync_vector=False,
    )

    answer = service.answer(
        "oxygen uptake",
        (result.record.paper_key,),
        retrieval_mode="semantic",
    )

    assert answer.success is True
    assert answer.evidence
    assert "回退 BM25" in answer.message


def test_user_pdf_rag_delete_removes_stored_files(tmp_path: Path) -> None:
    service = _user_service(tmp_path)
    result = service.ingest_pdf(
        "delete-me.pdf",
        _pdf_bytes("Aerobic oxygen uptake in college students."),
    )

    service.remove_paper(result.record.paper_key)

    assert service.list_papers() == ()
    assert not result.record.stored_path.exists()
    service.close()


def test_literature_research_page_loads_without_network_request() -> None:
    from streamlit.testing.v1 import AppTest

    app_path = PROJECT_ROOT / "app.py"
    test_app = AppTest.from_file(app_path, default_timeout=60).run()
    test_app.session_state["navigation"] = "智能文献研究"
    test_app.run()

    assert not test_app.exception
    assert any(
        "OpenAlex" in item.value and "PubMed" in item.value
        for item in test_app.info
    )


def test_literature_research_page_switches_to_user_pdf_rag() -> None:
    from streamlit.testing.v1 import AppTest

    app_path = PROJECT_ROOT / "app.py"
    test_app = AppTest.from_file(app_path, default_timeout=60).run()
    test_app.session_state["navigation"] = "智能文献研究"
    test_app.run()
    test_app.segmented_control(
        key="literature_research_workspace"
    ).set_value("个人论文 RAG").run()

    assert not test_app.exception
    assert any(
        "只读取你主动上传的 PDF" in item.value
        for item in test_app.info
    )
