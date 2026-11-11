"""基于本地 PDF 证据的关键词、语义与混合检索回答服务。"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from config import settings
from services.llm_service import LLMService
from services.prompt_service import build_rag_system_prompt
from services.vector_service import (
    EvidenceVectorStore,
    VectorStoreError,
    VectorStoreStatus,
)
from tools.document_chunker import EvidenceChunk


@dataclass(frozen=True)
class RetrievedEvidence:
    """检索结果及匹配分数。"""

    chunk: EvidenceChunk
    score: float
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    retrieval_sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceAnswer:
    """基于原文证据的一次回答。"""

    answer: str
    mode: str
    evidence: tuple[RetrievedEvidence, ...] = ()
    success: bool = True
    message: str = ""


class LocalEvidenceIndex:
    """使用 BM25 对证据片段进行本地关键词检索。"""

    def __init__(self, chunks: Iterable[EvidenceChunk]) -> None:
        self.chunks = tuple(chunks)
        self._tokenized_chunks = [
            _tokenize(chunk.text) for chunk in self.chunks
        ]
        self._term_frequencies = [
            Counter(tokens) for tokens in self._tokenized_chunks
        ]
        self._document_frequencies: Counter[str] = Counter()
        for tokens in self._tokenized_chunks:
            self._document_frequencies.update(set(tokens))
        self._average_length = (
            sum(len(tokens) for tokens in self._tokenized_chunks)
            / len(self._tokenized_chunks)
            if self._tokenized_chunks
            else 0.0
        )

    def search(
        self,
        question: str,
        *,
        top_k: int = 5,
    ) -> tuple[RetrievedEvidence, ...]:
        """返回与问题词法相关度最高的原文片段。"""

        if not 1 <= top_k <= 10:
            raise ValueError("检索数量必须在 1 到 10 之间。")
        if not self.chunks:
            return ()

        query_terms = tuple(dict.fromkeys(_tokenize(question)))
        if not query_terms:
            return ()

        scored: list[RetrievedEvidence] = []
        total_chunks = len(self.chunks)
        k1 = 1.5
        b = 0.75

        for chunk, tokens, frequencies in zip(
            self.chunks,
            self._tokenized_chunks,
            self._term_frequencies,
            strict=True,
        ):
            score = 0.0
            document_length = len(tokens)
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                document_frequency = self._document_frequencies[term]
                inverse_document_frequency = math.log(
                    1 + (total_chunks - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                length_ratio = (
                    document_length / self._average_length
                    if self._average_length
                    else 0.0
                )
                denominator = frequency + k1 * (1 - b + b * length_ratio)
                score += inverse_document_frequency * (
                    frequency * (k1 + 1) / denominator
                )

            if score > 0:
                scored.append(RetrievedEvidence(chunk=chunk, score=score))

        scored.sort(key=lambda item: item.score, reverse=True)
        return tuple(scored[:top_k])


class HybridEvidenceIndex:
    """组合 BM25 与 Chroma 语义结果的本地混合检索索引。"""

    def __init__(
        self,
        chunks: Iterable[EvidenceChunk],
        vector_store: EvidenceVectorStore | None = None,
    ) -> None:
        self.chunks = tuple(chunks)
        self.lexical_index = LocalEvidenceIndex(self.chunks)
        self.vector_store = vector_store
        self.last_warning = ""

    def status(self) -> VectorStoreStatus | None:
        """返回当前向量库与这批文献的同步状态。"""

        if self.vector_store is None:
            return None
        return self.vector_store.status(expected_chunks=len(self.chunks))

    def search(
        self,
        question: str,
        *,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> tuple[RetrievedEvidence, ...]:
        """按指定模式检索；语义不可用时自动退回 BM25。"""

        if not 1 <= top_k <= 10:
            raise ValueError("检索数量必须在 1 到 10 之间。")
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"hybrid", "semantic", "bm25"}:
            raise ValueError("检索模式必须是 hybrid、semantic 或 bm25。")
        self.last_warning = ""
        if not self.chunks:
            return ()

        if normalized_mode == "bm25" or self.vector_store is None:
            if normalized_mode != "bm25" and self.vector_store is None:
                self.last_warning = (
                    "当前证据库未连接持久化向量库，已使用 BM25 关键词检索。"
                )
            return self.lexical_index.search(question, top_k=top_k)

        candidate_limit = min(max(top_k * 4, 10), len(self.chunks))
        lexical_results = self.lexical_index.search(
            question,
            top_k=candidate_limit,
        )
        try:
            semantic_results = self.vector_store.search(
                question,
                top_k=candidate_limit,
            )
        except (VectorStoreError, ValueError) as exc:
            self.last_warning = (
                f"语义向量检索不可用，已自动回退 BM25：{exc}"
            )
            return self.lexical_index.search(question, top_k=top_k)

        chunks_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        semantic_by_id = {
            result.chunk_id: result.similarity
            for result in semantic_results
            if result.chunk_id in chunks_by_id
        }

        if normalized_mode == "semantic":
            ranked = sorted(
                semantic_by_id.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            return tuple(
                RetrievedEvidence(
                    chunk=chunks_by_id[chunk_id],
                    score=similarity,
                    semantic_score=similarity,
                    retrieval_sources=("语义",),
                )
                for chunk_id, similarity in ranked[:top_k]
            )

        lexical_by_id = {
            item.chunk.chunk_id: item.score for item in lexical_results
        }
        maximum_bm25 = max(lexical_by_id.values(), default=0.0)
        candidate_ids = set(lexical_by_id) | set(semantic_by_id)
        fused: list[RetrievedEvidence] = []
        for chunk_id in candidate_ids:
            bm25_score = lexical_by_id.get(chunk_id, 0.0)
            semantic_score = semantic_by_id.get(chunk_id, 0.0)
            normalized_bm25 = (
                bm25_score / maximum_bm25 if maximum_bm25 else 0.0
            )
            sources = []
            if chunk_id in lexical_by_id:
                sources.append("BM25")
            if chunk_id in semantic_by_id:
                sources.append("语义")
            fused.append(
                RetrievedEvidence(
                    chunk=chunks_by_id[chunk_id],
                    score=0.45 * normalized_bm25 + 0.55 * semantic_score,
                    bm25_score=bm25_score,
                    semantic_score=semantic_score,
                    retrieval_sources=tuple(sources),
                )
            )
        fused.sort(
            key=lambda item: (
                item.score,
                item.semantic_score,
                item.bm25_score,
            ),
            reverse=True,
        )
        return tuple(fused[:top_k])


class EvidenceRAGService:
    """把本地检索结果交给原文摘录或已配置的大模型。"""

    def __init__(
        self,
        index: LocalEvidenceIndex,
        llm_service: LLMService | None = None,
    ) -> None:
        self.index = index
        self.llm_service = llm_service or LLMService()

    def answer_question(
        self,
        question: str,
        *,
        top_k: int = 5,
        use_llm: bool = False,
        retrieval_mode: str | None = None,
    ) -> EvidenceAnswer:
        """检索证据并回答；任何路径都不会在证据不足时强行生成结论。"""

        clean_question = question.strip()
        if not clean_question:
            return EvidenceAnswer(
                answer="",
                mode="输入校验",
                success=False,
                message="问题不能为空。",
            )

        retrieval_warning = ""
        if isinstance(self.index, HybridEvidenceIndex):
            evidence = self.index.search(
                clean_question,
                top_k=top_k,
                mode=retrieval_mode or settings.rag_retrieval_mode,
            )
            retrieval_warning = self.index.last_warning
        else:
            evidence = self.index.search(clean_question, top_k=top_k)
        if not evidence:
            return EvidenceAnswer(
                answer=(
                    "当前文献不足以支持结论。"
                    "请上传包含相关内容的 PDF，或调整问题中的关键词。"
                ),
                mode="证据不足",
                message=retrieval_warning,
            )

        if use_llm:
            return self._answer_with_llm(
                clean_question,
                evidence,
                retrieval_warning=retrieval_warning,
            )
        message = "未调用真实大模型，仅展示检索到的原文片段。"
        if retrieval_warning:
            message = f"{retrieval_warning} {message}"
        return EvidenceAnswer(
            answer=_build_extractive_answer(evidence),
            mode="证据摘录（模拟）",
            evidence=evidence,
            message=message,
        )

    def _answer_with_llm(
        self,
        question: str,
        evidence: tuple[RetrievedEvidence, ...],
        *,
        retrieval_warning: str = "",
    ) -> EvidenceAnswer:
        """调用真实大模型，并把证据片段作为唯一事实来源。"""

        if not self.llm_service.is_configured:
            return EvidenceAnswer(
                answer="",
                mode="配置缺失",
                evidence=evidence,
                success=False,
                message=(
                    "未配置 OPENAI_API_KEY，无法调用真实模型。"
                    "你仍可使用“本地证据摘录”模式。"
                ),
            )

        context = _format_evidence_context(evidence)
        response = self.llm_service.answer(
            question,
            allow_simulation=False,
            system_prompt=build_rag_system_prompt(context),
        )
        message = response.message
        if retrieval_warning:
            message = f"{retrieval_warning} {message}".strip()
        return EvidenceAnswer(
            answer=response.answer,
            mode=response.mode,
            evidence=evidence,
            success=response.success,
            message=message,
        )


def _format_evidence_context(
    evidence: Sequence[RetrievedEvidence],
) -> str:
    blocks = []
    for index, item in enumerate(evidence, start=1):
        score_detail = (
            f"BM25：{item.bm25_score:.3f}，语义：{item.semantic_score:.3f}"
            if item.bm25_score or item.semantic_score
            else f"匹配分数：{item.score:.3f}"
        )
        blocks.append(
            f"[证据 {index}] {item.chunk.citation}\n"
            f"{score_detail}\n"
            f"原文：\n{item.chunk.text}"
        )
    return "\n\n".join(blocks)


def _build_extractive_answer(
    evidence: Sequence[RetrievedEvidence],
) -> str:
    """不调用模型时，只提取原文并明确标注模拟属性。"""

    blocks = [
        "> **模拟内容提示：** 当前未调用真实大模型，"
        "以下内容只是检索到的原文摘录，没有进行模型综合。"
    ]
    for index, item in enumerate(evidence, start=1):
        excerpt = item.chunk.text.strip()
        blocks.append(
            f"### 证据 {index}：{item.chunk.citation}\n"
            f"> {excerpt.replace(chr(10), chr(10) + '> ')}"
        )
    blocks.append(
        "**回答边界：** 以上是本地检索命中的原文，"
        "需结合上下文判断其是否真正回答了研究问题。"
    )
    return "\n\n".join(blocks)


def _tokenize(text: str) -> list[str]:
    """生成适合中英文混合文本的轻量检索词。

    中文使用双字组合，英文与数字按连续字符切分。单个汉字过于宽泛，容易造成
    “量”“的”等偶然命中，因此不纳入索引；短中文词组同时保留完整形式。
    """

    normalized = text.lower()
    tokens = re.findall(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", normalized)
    for run in re.findall(r"[\u4e00-\u9fff]+", normalized):
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
        if len(run) <= 6:
            tokens.append(run)
    return tokens
