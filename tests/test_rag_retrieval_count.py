"""检索数量边界与 RAG 失败处理测试。"""

from __future__ import annotations

import math

import pytest

from config import Settings
from services.llm_service import LLMService
from services.rag_service import (
    DEFAULT_RETRIEVAL_COUNT,
    EvidenceRAGService,
    HybridEvidenceIndex,
    LocalEvidenceIndex,
    normalize_retrieval_count,
)
from services.vector_service import VectorSearchResult
from tools.document_chunker import EvidenceChunk


def _evidence_chunks(count: int = 12) -> tuple[EvidenceChunk, ...]:
    return tuple(
        EvidenceChunk(
            f"chunk-{index}",
            "retrieval-count.pdf",
            index + 1,
            "体育训练干预与体质健康检索测试。",
        )
        for index in range(count)
    )


class _TopKVectorStore:
    def __init__(self, chunks: tuple[EvidenceChunk, ...]) -> None:
        self.chunks = chunks
        self.requested_top_k: int | None = None

    def search(self, question: str, *, top_k: int):
        self.requested_top_k = top_k
        return tuple(
            VectorSearchResult(chunk.chunk_id, 1.0 - index / 100)
            for index, chunk in enumerate(self.chunks[:top_k])
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (1, 1),
        (5, 5),
        (10, 10),
        (0, DEFAULT_RETRIEVAL_COUNT),
        (-3, DEFAULT_RETRIEVAL_COUNT),
        (11, DEFAULT_RETRIEVAL_COUNT),
        (None, DEFAULT_RETRIEVAL_COUNT),
        ("", DEFAULT_RETRIEVAL_COUNT),
        (" 5 ", 5),
        ("not-a-number", DEFAULT_RETRIEVAL_COUNT),
        (3.5, DEFAULT_RETRIEVAL_COUNT),
        (math.nan, DEFAULT_RETRIEVAL_COUNT),
        (True, DEFAULT_RETRIEVAL_COUNT),
    ),
)
def test_normalize_retrieval_count_handles_invalid_values(
    value: object,
    expected: int,
) -> None:
    assert normalize_retrieval_count(value) == expected


@pytest.mark.parametrize("mode", ("hybrid", "semantic", "bm25"))
@pytest.mark.parametrize("top_k", (1, 5, 10))
def test_hybrid_retrieval_keeps_internal_candidate_count_within_limit(
    mode: str,
    top_k: int,
) -> None:
    chunks = _evidence_chunks()
    vector_store = _TopKVectorStore(chunks)
    index = HybridEvidenceIndex(chunks, vector_store=vector_store)

    results = index.search(
        "体育训练干预与体质健康",
        top_k=top_k,
        mode=mode,
    )

    assert 0 < len(results) <= top_k
    if mode != "bm25":
        assert vector_store.requested_top_k is not None
        assert 1 <= vector_store.requested_top_k <= 10


@pytest.mark.parametrize("invalid_top_k", (0, -1, 11))
def test_low_level_index_keeps_strict_range_validation(
    invalid_top_k: int,
) -> None:
    index = LocalEvidenceIndex(_evidence_chunks())

    with pytest.raises(ValueError, match="检索数量必须在 1 到 10 之间"):
        index.search("体育训练", top_k=invalid_top_k)


@pytest.mark.parametrize(
    ("value", "expected_count"),
    (
        (1, 1),
        (5, 5),
        (10, 10),
        (0, DEFAULT_RETRIEVAL_COUNT),
        (11, DEFAULT_RETRIEVAL_COUNT),
        (None, DEFAULT_RETRIEVAL_COUNT),
        ("5", 5),
    ),
)
def test_rag_answer_normalizes_retrieval_count(
    value: object,
    expected_count: int,
) -> None:
    service = EvidenceRAGService(
        LocalEvidenceIndex(_evidence_chunks()),
        LLMService(Settings.from_env({"OPENAI_API_KEY": ""})),
    )

    answer = service.answer_question(
        "体育训练干预与体质健康",
        top_k=value,  # type: ignore[arg-type]
    )

    assert answer.success is True
    assert len(answer.evidence) == expected_count


def test_rag_answer_prompts_to_import_when_library_is_empty() -> None:
    service = EvidenceRAGService(LocalEvidenceIndex(()))

    answer = service.answer_question("体育训练如何影响体质健康？")

    assert answer.success is False
    assert answer.mode == "证据库为空"
    assert "请先导入论文" in answer.message


class _BrokenEvidenceIndex(LocalEvidenceIndex):
    def search(self, question: str, *, top_k: int = 5):
        raise RuntimeError("索引文件损坏")


def test_rag_answer_returns_specific_retrieval_error() -> None:
    service = EvidenceRAGService(_BrokenEvidenceIndex(_evidence_chunks()))

    answer = service.answer_question("体育训练如何影响体质健康？")

    assert answer.success is False
    assert answer.mode == "检索失败"
    assert "索引文件损坏" in answer.message
