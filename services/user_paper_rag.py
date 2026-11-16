"""用户主动上传 PDF 的隔离式本地 RAG 服务。

本模块只处理用户上传的 PDF，不接收 Semantic Scholar、arXiv 或其他外部
检索结果。论文按文件内容的 SHA-256 隔离保存，Chroma 使用独立集合，
不会污染项目内置文献库或公开文献元数据。
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config import Settings, settings as default_settings
from services.llm_service import LLMService
from services.rag_service import (
    EvidenceAnswer,
    EvidenceRAGService,
    HybridEvidenceIndex,
)
from services.vector_service import (
    EmbeddingEncoder,
    EvidenceVectorStore,
    VectorStoreError,
    VectorSyncResult,
)
from tools.document_chunker import EvidenceChunk, chunk_pdf_pages
from tools.pdf_reader import extract_pdf_text


USER_PAPER_COLLECTION_NAME = "user_uploaded_papers"
USER_PAPER_SCHEMA_VERSION = 1


class UserPaperRAGError(RuntimeError):
    """用户论文无法导入、删除或查询时抛出的可展示错误。"""


@dataclass(frozen=True)
class UserPaperRecord:
    """本地用户论文的一篇可检索记录。"""

    paper_key: str
    file_name: str
    title: str
    page_count: int
    chunk_count: int
    stored_path: Path
    created_at: str


@dataclass(frozen=True)
class UserPaperIngestResult:
    """一次用户 PDF 导入结果。"""

    record: UserPaperRecord
    created: bool
    vector_sync: VectorSyncResult | None = None
    warning: str = ""


class UserPaperRAGService:
    """管理用户 PDF、持久化 Chroma 索引并基于原文回答。"""

    def __init__(
        self,
        config: Settings | None = None,
        *,
        base_dir: Path | str | None = None,
        llm_service: LLMService | None = None,
        vector_store: EvidenceVectorStore | None = None,
        encoder: EmbeddingEncoder | None = None,
    ) -> None:
        self.config = config or default_settings
        self.root = Path(
            base_dir or self.config.data_dir / "user_papers"
        ).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.llm_service = llm_service or LLMService(self.config)
        self.vector_store = vector_store or EvidenceVectorStore(
            self.root,
            model_name=self.config.embedding_model,
            collection_name=USER_PAPER_COLLECTION_NAME,
            cache_dir=self.config.embedding_cache_dir,
            encoder=encoder,
        )

    def ingest_pdf(
        self,
        file_name: str,
        file_bytes: bytes,
        *,
        sync_vector: bool = True,
    ) -> UserPaperIngestResult:
        """解析并保存一篇用户主动上传的 PDF。

        相同内容的文件不会重复导入。向量模型不可用时仍保存原文与分页
        证据，后续查询会自动回退到 BM25，而不是让整个上传操作失败。
        """

        clean_name = Path(file_name).name.strip()
        if not clean_name or Path(clean_name).suffix.lower() != ".pdf":
            raise UserPaperRAGError("只接受 PDF 文件。")
        if not file_bytes:
            raise UserPaperRAGError(f"{clean_name} 是空文件，无法导入。")

        max_bytes = self.config.max_pdf_size_mb * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise UserPaperRAGError(
                f"{clean_name} 超过 {self.config.max_pdf_size_mb} MB 限制。"
            )

        paper_key = hashlib.sha256(file_bytes).hexdigest()
        existing = self._load_record(paper_key)
        if existing is not None:
            return UserPaperIngestResult(
                record=existing,
                created=False,
                warning="相同 PDF 已存在，未重复导入。",
            )

        extraction = extract_pdf_text(
            file_bytes,
            max_pages=self.config.max_pdf_pages,
            max_chars=self.config.max_pdf_chars,
        )
        if not extraction.success:
            raise UserPaperRAGError(
                f"{clean_name} 无法解析：{extraction.message}"
            )
        if not extraction.pages:
            raise UserPaperRAGError(
                f"{clean_name} 没有可检索文本，可能是扫描版 PDF。"
            )

        chunks = _namespace_chunks(
            paper_key,
            chunk_pdf_pages(
                clean_name,
                extraction.pages,
                max_chars=self.config.rag_chunk_chars,
                overlap_chars=self.config.rag_chunk_overlap,
            ),
        )
        if not chunks:
            raise UserPaperRAGError(
                f"{clean_name} 未生成有效证据片段。"
            )

        created_at = datetime.now(UTC).isoformat(timespec="seconds")
        record = UserPaperRecord(
            paper_key=paper_key,
            file_name=clean_name,
            title=Path(clean_name).stem,
            page_count=extraction.page_count,
            chunk_count=len(chunks),
            stored_path=self.root / paper_key / "source.pdf",
            created_at=created_at,
        )
        self._persist_paper(record, file_bytes, chunks)

        vector_sync: VectorSyncResult | None = None
        warning = ""
        if sync_vector:
            try:
                vector_sync = self.sync_index()
            except (VectorStoreError, ValueError, OSError) as exc:
                warning = (
                    "PDF 已保存，但语义索引暂未建立；当前可回退 BM25 "
                    f"关键词检索：{exc}"
                )

        if extraction.truncated:
            truncated_notice = (
                "PDF 内容较长，已按系统页数或字符上限截取部分文本。"
            )
            warning = f"{warning} {truncated_notice}".strip()
        return UserPaperIngestResult(
            record=record,
            created=True,
            vector_sync=vector_sync,
            warning=warning,
        )

    def list_papers(self) -> tuple[UserPaperRecord, ...]:
        """列出本机已保存的用户论文，不加载向量模型。"""

        records: list[UserPaperRecord] = []
        for metadata_path in self.root.glob("*/metadata.json"):
            try:
                record = _record_from_metadata(
                    metadata_path,
                    self.root,
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            records.append(record)
        records.sort(key=lambda item: (item.created_at, item.file_name))
        return tuple(records)

    def remove_paper(self, paper_key: str) -> VectorSyncResult | None:
        """删除一篇用户论文并同步清理它的全部向量。"""

        safe_key = _validate_paper_key(paper_key)
        paper_dir = self.root / safe_key
        if not (paper_dir / "metadata.json").is_file():
            raise UserPaperRAGError("未找到对应的用户论文。")
        shutil.rmtree(paper_dir)
        try:
            return self.sync_index()
        except (VectorStoreError, ValueError, OSError) as exc:
            raise UserPaperRAGError(
                f"论文文件已删除，但向量索引清理失败：{exc}"
            ) from exc

    def sync_index(self) -> VectorSyncResult:
        """把当前全部用户论文证据同步到独立 Chroma 集合。"""

        all_chunks = tuple(
            chunk
            for record in self.list_papers()
            for chunk in self.load_chunks(record.paper_key)
        )
        return self.vector_store.sync_chunks(all_chunks)

    def load_chunks(self, paper_key: str) -> tuple[EvidenceChunk, ...]:
        """读取一篇用户论文持久化的分页证据。"""

        safe_key = _validate_paper_key(paper_key)
        chunks_path = self.root / safe_key / "chunks.json"
        try:
            payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise UserPaperRAGError("论文证据索引不存在。") from exc
        except json.JSONDecodeError as exc:
            raise UserPaperRAGError("论文证据索引已损坏。") from exc

        raw_chunks = payload.get("chunks") if isinstance(payload, dict) else None
        if not isinstance(raw_chunks, list):
            raise UserPaperRAGError("论文证据索引格式不正确。")

        chunks: list[EvidenceChunk] = []
        for item in raw_chunks:
            if not isinstance(item, dict):
                continue
            chunk_id = str(item.get("chunk_id") or "").strip()
            document_name = str(item.get("document_name") or "").strip()
            text = str(item.get("text") or "").strip()
            try:
                page_number = int(item.get("page_number"))
            except (TypeError, ValueError):
                continue
            if not chunk_id or not document_name or not text or page_number < 1:
                continue
            chunks.append(
                EvidenceChunk(
                    chunk_id=chunk_id,
                    document_name=document_name,
                    page_number=page_number,
                    text=text,
                )
            )
        return tuple(chunks)

    def answer(
        self,
        question: str,
        paper_keys: Sequence[str],
        *,
        top_k: int = 5,
        use_llm: bool = False,
        retrieval_mode: str = "hybrid",
    ) -> EvidenceAnswer:
        """只在选中的用户论文中检索并回答，证据不足时拒绝下结论。"""

        selected_keys = tuple(dict.fromkeys(key.strip() for key in paper_keys))
        selected_keys = tuple(key for key in selected_keys if key)
        if not selected_keys:
            return EvidenceAnswer(
                answer="",
                mode="输入校验",
                success=False,
                message="请至少选择一篇用户论文。",
            )

        available_keys = {
            record.paper_key for record in self.list_papers()
        }
        unknown_keys = set(selected_keys) - available_keys
        if unknown_keys:
            return EvidenceAnswer(
                answer="",
                mode="输入校验",
                success=False,
                message="部分已选论文不存在，请刷新论文列表。",
            )

        chunks = tuple(
            chunk
            for key in selected_keys
            for chunk in self.load_chunks(key)
        )
        index = HybridEvidenceIndex(
            chunks,
            vector_store=self.vector_store,
        )
        service = EvidenceRAGService(index, self.llm_service)
        return service.answer_question(
            question,
            top_k=top_k,
            use_llm=use_llm,
            retrieval_mode=retrieval_mode,
        )

    def close(self) -> None:
        """释放 Chroma 文件句柄。"""

        self.vector_store.close()

    def _persist_paper(
        self,
        record: UserPaperRecord,
        file_bytes: bytes,
        chunks: tuple[EvidenceChunk, ...],
    ) -> None:
        """先写临时目录再原子改名，避免留下半篇损坏论文。"""

        paper_dir = self.root / record.paper_key
        if paper_dir.exists():
            raise UserPaperRAGError("该 PDF 的本地目录已经存在，请重试。")

        temporary_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{record.paper_key[:8]}-",
                dir=self.root,
            )
        )
        try:
            (temporary_dir / "source.pdf").write_bytes(file_bytes)
            _write_json(
                temporary_dir / "chunks.json",
                {
                    "schema_version": USER_PAPER_SCHEMA_VERSION,
                    "paper_key": record.paper_key,
                    "chunks": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "document_name": chunk.document_name,
                            "page_number": chunk.page_number,
                            "text": chunk.text,
                        }
                        for chunk in chunks
                    ],
                },
            )
            _write_json(
                temporary_dir / "metadata.json",
                {
                    "schema_version": USER_PAPER_SCHEMA_VERSION,
                    "paper_key": record.paper_key,
                    "file_name": record.file_name,
                    "title": record.title,
                    "page_count": record.page_count,
                    "chunk_count": record.chunk_count,
                    "created_at": record.created_at,
                    "storage": "用户主动上传，本地隔离保存",
                },
            )
            # Windows 上目录 rename 可能触发 WinError 5；shutil.move
            # 对同一磁盘内的临时目录更稳定，并保持完整目录原子落位语义。
            shutil.move(str(temporary_dir), str(paper_dir))
        except Exception:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise

    def _load_record(self, paper_key: str) -> UserPaperRecord | None:
        """读取指定内容哈希对应的论文元数据。"""

        safe_key = _validate_paper_key(paper_key)
        metadata_path = self.root / safe_key / "metadata.json"
        if not metadata_path.is_file():
            return None
        return _record_from_metadata(metadata_path, self.root)


def _namespace_chunks(
    paper_key: str,
    chunks: Sequence[EvidenceChunk],
) -> tuple[EvidenceChunk, ...]:
    """给片段 ID 加内容哈希前缀，避免不同论文的同名片段冲突。"""

    return tuple(
        EvidenceChunk(
            chunk_id=f"{paper_key}-{chunk.chunk_id}",
            document_name=chunk.document_name,
            page_number=chunk.page_number,
            text=chunk.text,
        )
        for chunk in chunks
    )


def _record_from_metadata(
    metadata_path: Path,
    root: Path,
) -> UserPaperRecord:
    """把磁盘元数据转换为记录，并拒绝越出用户论文根目录的路径。"""

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("论文元数据必须是 JSON 对象。")
    paper_key = _validate_paper_key(str(data.get("paper_key") or ""))
    if metadata_path.parent.name != paper_key:
        raise ValueError("论文目录与内容哈希不一致。")
    file_name = str(data.get("file_name") or "").strip()
    title = str(data.get("title") or "").strip()
    if not file_name or not title:
        raise ValueError("论文元数据缺少文件名或题名。")
    page_count = int(data.get("page_count"))
    chunk_count = int(data.get("chunk_count"))
    if page_count < 1 or chunk_count < 1:
        raise ValueError("论文页数或片段数不合法。")
    stored_path = (root / paper_key / "source.pdf").resolve()
    if stored_path.parent != (root / paper_key).resolve():
        raise ValueError("论文存储路径越界。")
    return UserPaperRecord(
        paper_key=paper_key,
        file_name=file_name,
        title=title,
        page_count=page_count,
        chunk_count=chunk_count,
        stored_path=stored_path,
        created_at=str(data.get("created_at") or ""),
    )


def _validate_paper_key(paper_key: str) -> str:
    """只接受 SHA-256 十六进制标识，防止路径穿越。"""

    normalized = paper_key.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise UserPaperRAGError("论文标识格式不正确。")
    return normalized


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """以 UTF-8 写入格式化 JSON，便于初学者检查本地文件。"""

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
