"""本地多语言向量库服务。

使用 sentence-transformers 生成中文/英文文本向量，并使用 Chroma 持久化。
模型和 Chroma 都在首次真正需要时才加载，避免仅打开网页就触发下载或长时间等待。

命令行构建：
    python -m services.vector_service --build
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from config import settings
from tools.document_chunker import EvidenceChunk
from tools.knowledge_base import (
    BundledKnowledgeBase,
    KnowledgeBaseError,
    load_bundled_knowledge_base,
)


VECTOR_SCHEMA_VERSION = "2"
DEFAULT_COLLECTION_NAME = "evidence_chunks"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
VECTOR_DIR_NAME = "chroma"
HNSW_SYNC_THRESHOLD = 100_000


class VectorStoreError(Exception):
    """向量模型、Chroma 或持久化数据不可用时抛出的可展示错误。"""


class EmbeddingEncoder(Protocol):
    """向量服务需要的最小编码器接口，便于测试时替换本地模型。"""

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """把文献片段编码为向量。"""

    def encode_query(self, text: str) -> list[float]:
        """把检索问题编码为向量。"""


@dataclass(frozen=True)
class VectorStoreStatus:
    """不加载模型即可读取的向量库状态。"""

    path: Path
    collection_name: str
    model_name: str
    stored_chunks: int
    expected_chunks: int | None
    model_cached: bool
    synchronized: bool
    available: bool
    message: str


@dataclass(frozen=True)
class VectorSearchResult:
    """Chroma 返回的一条语义检索结果。"""

    chunk_id: str
    similarity: float


@dataclass(frozen=True)
class VectorSyncResult:
    """一次向量库增量同步的统计。"""

    total_chunks: int
    added_chunks: int
    updated_chunks: int
    removed_chunks: int
    unchanged_chunks: int
    encoded_chunks: int
    recovery_backup: Path | None = None

    @property
    def changed_chunks(self) -> int:
        """本次发生新增、更新或删除的片段总数。"""

        return self.added_chunks + self.updated_chunks + self.removed_chunks


class LocalEmbeddingEncoder:
    """按 E5 模型要求添加 passage/query 前缀的本地编码器。"""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        cache_dir: Path | str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else None
        self.batch_size = batch_size
        self._model: Any = None

    def _load_model(self) -> Any:
        """延迟加载 sentence-transformers 模型。"""

        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise VectorStoreError(
                "未安装 sentence-transformers，无法生成语义向量。"
            ) from exc

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=(
                    str(self.cache_dir) if self.cache_dir is not None else None
                ),
                device="cpu",
            )
        except Exception as exc:
            raise VectorStoreError(
                f"无法加载本地向量模型 {self.model_name}：{exc}。"
                "首次运行需要联网下载模型，请检查网络或模型缓存目录。"
            ) from exc
        return self._model

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """编码文献片段，使用 E5 的 passage 前缀。"""

        return self._encode([f"passage: {text}" for text in texts])

    def encode_query(self, text: str) -> list[float]:
        """编码用户问题，使用 E5 的 query 前缀。"""

        vectors = self._encode([f"query: {text.strip()}"])
        return vectors[0]

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        try:
            vectors = model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"生成本地语义向量失败：{exc}") from exc
        return [vector.tolist() for vector in vectors]


class EvidenceVectorStore:
    """Chroma 持久化向量库，支持增量同步和余弦语义检索。"""

    def __init__(
        self,
        base_dir: Path | str | None = None,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        cache_dir: Path | str | None = None,
        encoder: EmbeddingEncoder | None = None,
    ) -> None:
        root = Path(base_dir or settings.knowledge_base_dir).resolve()
        self.path = root / VECTOR_DIR_NAME
        self.model_name = model_name.strip() or DEFAULT_EMBEDDING_MODEL
        self.collection_name = (
            collection_name.strip() or DEFAULT_COLLECTION_NAME
        )
        self.cache_dir = Path(
            cache_dir or settings.embedding_cache_dir
        ).resolve()
        self.encoder = encoder or LocalEmbeddingEncoder(
            self.model_name,
            cache_dir=self.cache_dir,
        )
        self._client: Any = None
        self._collection: Any = None

    def status(self, *, expected_chunks: int | None = None) -> VectorStoreStatus:
        """读取集合数量、模型版本和同步状态，不加载向量模型。"""

        model_cached = self._is_model_cached()
        try:
            collection = self._get_collection()
            stored_chunks = int(collection.count())
            metadata = collection.metadata or {}
            metadata_matches = (
                metadata.get("embedding_model") == self.model_name
                and metadata.get("vector_schema") == VECTOR_SCHEMA_VERSION
            )
        except Exception as exc:
            return VectorStoreStatus(
                path=self.path,
                collection_name=self.collection_name,
                model_name=self.model_name,
                stored_chunks=0,
                expected_chunks=expected_chunks,
                model_cached=model_cached,
                synchronized=False,
                available=False,
                message=_describe_vector_store_error(exc),
            )

        if not metadata_matches:
            return VectorStoreStatus(
                path=self.path,
                collection_name=self.collection_name,
                model_name=self.model_name,
                stored_chunks=stored_chunks,
                expected_chunks=expected_chunks,
                model_cached=model_cached,
                synchronized=False,
                available=False,
                message="向量库使用的模型版本与当前配置不一致，需要重建。",
            )

        synchronized = (
            expected_chunks is not None
            and stored_chunks == expected_chunks
            and stored_chunks > 0
        )
        available = stored_chunks > 0
        if synchronized:
            message = (
                f"向量库已同步：{stored_chunks} 个片段，"
                f"模型 {self.model_name}。"
            )
        elif stored_chunks == 0:
            message = "向量库尚未构建，请运行向量索引构建命令。"
        elif expected_chunks is not None:
            message = (
                f"向量库有 {stored_chunks} 个片段，当前文献有 "
                f"{expected_chunks} 个片段，需要同步。"
            )
        else:
            message = f"向量库包含 {stored_chunks} 个片段。"

        return VectorStoreStatus(
            path=self.path,
            collection_name=self.collection_name,
            model_name=self.model_name,
            stored_chunks=stored_chunks,
            expected_chunks=expected_chunks,
            model_cached=model_cached,
            synchronized=synchronized,
            available=available,
            message=message,
        )

    def sync_chunks(
        self,
        chunks: Iterable[EvidenceChunk],
    ) -> VectorSyncResult:
        """增量同步片段；发现 HNSW 索引损坏时自动隔离并重建。"""

        desired = _index_chunks(chunks)
        try:
            return self._sync_chunks_once(desired)
        except Exception as exc:
            if _looks_like_hnsw_index_error(exc):
                backup_path = self._recover_corrupt_index()
                try:
                    result = self._sync_chunks_once(desired)
                except Exception as retry_exc:
                    raise VectorStoreError(
                        "检测到 HNSW 索引损坏，旧目录已隔离到 "
                        f"{backup_path}，但自动重建失败：{retry_exc}。"
                        "请确认没有其他程序占用该目录后重试。"
                    ) from retry_exc
                return replace(result, recovery_backup=backup_path)
            if isinstance(exc, VectorStoreError):
                raise
            raise VectorStoreError(f"同步 Chroma 向量库失败：{exc}") from exc

    def _sync_chunks_once(
        self,
        desired: dict[str, EvidenceChunk],
    ) -> VectorSyncResult:
        """执行一次同步；供正常同步和损坏恢复后的重试复用。"""

        collection = self._get_collection()
        existing = collection.get(include=["documents", "metadatas"])
        existing_id_list = [str(item) for item in (existing.get("ids") or [])]
        existing_ids = set(existing_id_list)
        existing_documents = existing.get("documents") or []
        existing_metadatas = existing.get("metadatas") or []
        existing_by_id = {
            chunk_id: (
                document_text or "",
                metadata or {},
            )
            for chunk_id, document_text, metadata in zip(
                existing_id_list,
                existing_documents,
                existing_metadatas,
                strict=False,
            )
        }

        obsolete_ids = sorted(existing_ids - set(desired))
        added: list[EvidenceChunk] = []
        updated: list[EvidenceChunk] = []
        unchanged_count = 0

        for chunk_id, chunk in desired.items():
            previous = existing_by_id.get(chunk_id)
            metadata = _chunk_metadata(chunk)
            if previous is None:
                added.append(chunk)
                continue
            previous_text, previous_metadata = previous
            if (
                previous_text == chunk.text
                and _metadata_matches(previous_metadata, metadata)
            ):
                unchanged_count += 1
            else:
                updated.append(chunk)

        changed_chunks = added + updated
        embeddings = self.encoder.encode_documents(
            [chunk.text for chunk in changed_chunks]
        )
        if len(embeddings) != len(changed_chunks):
            raise VectorStoreError("向量模型返回的向量数量与片段数量不一致。")

        for start in range(0, len(changed_chunks), 128):
            batch_chunks = changed_chunks[start : start + 128]
            batch_embeddings = embeddings[start : start + 128]
            collection.upsert(
                ids=[chunk.chunk_id for chunk in batch_chunks],
                embeddings=batch_embeddings,
                documents=[chunk.text for chunk in batch_chunks],
                metadatas=[
                    _chunk_metadata(chunk) for chunk in batch_chunks
                ],
            )

        for start in range(0, len(obsolete_ids), 512):
            collection.delete(ids=obsolete_ids[start : start + 512])

        stored_count = int(collection.count())
        if stored_count != len(desired):
            raise VectorStoreError(
                f"向量库同步后应有 {len(desired)} 个片段，"
                f"实际为 {stored_count} 个。"
            )

        return VectorSyncResult(
            total_chunks=len(desired),
            added_chunks=len(added),
            updated_chunks=len(updated),
            removed_chunks=len(obsolete_ids),
            unchanged_chunks=unchanged_count,
            encoded_chunks=len(changed_chunks),
        )

    def search(
        self,
        question: str,
        *,
        top_k: int = 10,
        min_similarity: float = 0.20,
    ) -> tuple[VectorSearchResult, ...]:
        """把问题编码后检索余弦相似度最高的片段。"""

        if not question.strip():
            return ()
        if not 1 <= top_k <= 50:
            raise ValueError("语义检索数量必须在 1 到 50 之间。")
        if not -1.0 <= min_similarity <= 1.0:
            raise ValueError("最低语义相似度必须在 -1 到 1 之间。")

        try:
            collection = self._get_collection()
            stored_count = int(collection.count())
            if stored_count == 0:
                raise VectorStoreError("向量库为空，请先构建语义索引。")

            query_embedding = self.encoder.encode_query(question)
            response = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, stored_count),
                include=["distances"],
            )
        except Exception as exc:
            if _looks_like_hnsw_index_error(exc):
                raise VectorStoreError(
                    "本地向量库的 HNSW 索引文件损坏，当前无法进行语义检索。"
                    "本次查询将回退 BM25；请点击“构建或更新本地向量库”，"
                    "系统会先隔离损坏目录，再从原始证据自动重建。"
                ) from exc
            if isinstance(exc, VectorStoreError):
                raise
            raise VectorStoreError(f"Chroma 语义检索失败：{exc}") from exc

        ids = (response.get("ids") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        results: list[VectorSearchResult] = []
        for chunk_id, distance in zip(ids, distances, strict=False):
            similarity = max(-1.0, min(1.0, 1.0 - float(distance)))
            if similarity >= min_similarity:
                results.append(
                    VectorSearchResult(
                        chunk_id=str(chunk_id),
                        similarity=similarity,
                    )
                )
        return tuple(results)

    def close(self) -> None:
        """关闭 Chroma 客户端，确保持久化索引和文件句柄正确释放。"""

        client = self._client
        self._collection = None
        self._client = None
        if client is None:
            return
        close = getattr(client, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception as exc:
            raise VectorStoreError(f"关闭向量数据库失败：{exc}") from exc

    def _recover_corrupt_index(self) -> Path | None:
        """隔离损坏的派生索引目录，保留现场供排查后重新创建。"""

        try:
            self.close()
        except VectorStoreError as exc:
            raise VectorStoreError(
                "检测到 HNSW 索引损坏，但无法释放 Chroma 文件句柄："
                f"{exc}。请关闭其他正在使用该项目的程序后重试。"
            ) from exc

        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
            return None

        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        base_name = f"{self.path.name}.corrupt-{timestamp}"
        backup_path = self.path.with_name(base_name)
        suffix = 1
        while backup_path.exists():
            backup_path = self.path.with_name(f"{base_name}-{suffix}")
            suffix += 1

        try:
            self.path.rename(backup_path)
        except OSError as exc:
            raise VectorStoreError(
                "检测到 HNSW 索引损坏，但无法隔离旧目录。"
                "请关闭其他正在使用该项目的程序后重试；"
                f"原目录保持不变：{self.path}。错误：{exc}"
            ) from exc

        self.path.mkdir(parents=True, exist_ok=True)
        return backup_path

    def _get_collection(self) -> Any:
        """创建或打开 Chroma 集合，模型配置变化时自动重建集合。"""

        if self._collection is not None:
            return self._collection
        try:
            import chromadb
        except ImportError as exc:
            raise VectorStoreError(
                "未安装 chromadb，无法使用本地向量数据库。"
            ) from exc

        self.path.mkdir(parents=True, exist_ok=True)
        try:
            self._client = chromadb.PersistentClient(path=str(self.path))
            collection = self._client.get_collection(
                self.collection_name,
                embedding_function=None,
            )
            metadata = collection.metadata or {}
            if not (
                metadata.get("embedding_model") == self.model_name
                and metadata.get("vector_schema") == VECTOR_SCHEMA_VERSION
            ):
                self._client.delete_collection(self.collection_name)
                collection = self._create_collection()
        except Exception as exc:
            if _looks_like_missing_collection(exc):
                try:
                    collection = self._create_collection()
                except Exception as create_exc:
                    raise VectorStoreError(
                        f"无法创建 Chroma 集合：{create_exc}"
                    ) from create_exc
            else:
                raise VectorStoreError(
                    f"无法打开 Chroma 向量库：{exc}"
                ) from exc

        self._collection = collection
        return collection

    def _create_collection(self) -> Any:
        if self._client is None:
            raise VectorStoreError("Chroma 客户端尚未初始化。")
        return self._client.get_or_create_collection(
            name=self.collection_name,
            configuration={
                "hnsw": {
                    "space": "cosine",
                    # Chroma 1.5.x 的默认阈值 1000 会在大集合压缩后生成
                    # 损坏的 HNSW 读取计划；提高阈值可稳定持久化本项目索引。
                    "sync_threshold": HNSW_SYNC_THRESHOLD,
                }
            },
            metadata={
                "embedding_model": self.model_name,
                "vector_schema": VECTOR_SCHEMA_VERSION,
                "distance": "cosine",
                "hnsw_sync_threshold": HNSW_SYNC_THRESHOLD,
            },
            embedding_function=None,
        )

    def _is_model_cached(self) -> bool:
        """根据 Hugging Face 缓存目录名判断模型是否已经下载。"""

        model_folder = f"models--{self.model_name.replace('/', '--')}"
        return (self.cache_dir / model_folder).exists()


def sync_bundled_vectors(
    bundle: BundledKnowledgeBase | None = None,
    *,
    base_dir: Path | str | None = None,
    vector_store: EvidenceVectorStore | None = None,
) -> VectorSyncResult:
    """把当前内置证据索引同步到持久化 Chroma 向量库。"""

    root = Path(base_dir or settings.knowledge_base_dir).resolve()
    current_bundle = bundle or load_bundled_knowledge_base(root)
    store = vector_store or EvidenceVectorStore(
        root,
        model_name=settings.embedding_model,
        collection_name=settings.chroma_collection_name,
    )
    return store.sync_chunks(current_bundle.chunks)


def _index_chunks(
    chunks: Iterable[EvidenceChunk],
) -> dict[str, EvidenceChunk]:
    """按片段 ID 建立有序映射，并拒绝重复或空 ID。"""

    indexed: dict[str, EvidenceChunk] = {}
    for chunk in chunks:
        if not chunk.chunk_id.strip():
            raise VectorStoreError("证据片段 ID 不能为空。")
        if chunk.chunk_id in indexed:
            raise VectorStoreError(f"证据片段 ID 重复：{chunk.chunk_id}")
        indexed[chunk.chunk_id] = chunk
    return indexed


def _chunk_metadata(chunk: EvidenceChunk) -> dict[str, str | int]:
    """生成 Chroma 支持的标量元数据。"""

    return {
        "document_name": chunk.document_name,
        "page_number": chunk.page_number,
        "chunk_id": chunk.chunk_id,
    }


def _metadata_matches(
    stored: dict[str, Any],
    desired: dict[str, str | int],
) -> bool:
    """比较 Chroma 中保存的片段定位信息。"""

    return all(stored.get(key) == value for key, value in desired.items())


def _looks_like_missing_collection(exc: Exception) -> bool:
    """兼容不同 Chroma 版本对“集合不存在”的异常措辞。"""

    message = str(exc).lower()
    return (
        "does not exist" in message
        or "not found" in message
        or "does not exists" in message
    )


def _looks_like_hnsw_index_error(exc: Exception) -> bool:
    """识别 Chroma 在不同阶段报告的 HNSW 派生索引损坏。"""

    messages: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(str(current).lower())
        current = current.__cause__ or current.__context__

    combined = " ".join(messages)
    markers = (
        "error loading hnsw index",
        "error creating hnsw segment reader",
        "error constructing hnsw segment reader",
        "error sending backfill request to compactor",
        "failed to load hnsw index",
        "could not load hnsw index",
    )
    return any(marker in combined for marker in markers)


def _describe_vector_store_error(exc: Exception) -> str:
    """把底层 Chroma 异常转换为界面可执行的中文提示。"""

    if _looks_like_hnsw_index_error(exc):
        return (
            "向量库尚不可用：本地 HNSW 索引文件损坏。"
            "原始 PDF 和证据索引未受影响；请点击“构建或更新本地向量库”，"
            "系统会隔离损坏目录并从原始证据自动重建。"
        )
    return f"向量库尚不可用：{exc}"


def main() -> int:
    """命令行入口：构建向量库或查看状态。"""

    parser = argparse.ArgumentParser(
        description="构建体研智脑本地 Chroma 语义索引"
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="根据当前证据索引增量构建或更新向量库",
    )
    args = parser.parse_args()

    store = EvidenceVectorStore(
        settings.knowledge_base_dir,
        model_name=settings.embedding_model,
        collection_name=settings.chroma_collection_name,
    )
    try:
        bundle = load_bundled_knowledge_base(settings.knowledge_base_dir)
        if args.build:
            result = sync_bundled_vectors(
                bundle,
                vector_store=store,
            )
            print(
                "向量库构建完成："
                f"{result.total_chunks} 个片段，"
                f"新增 {result.added_chunks}，"
                f"更新 {result.updated_chunks}，"
                f"删除 {result.removed_chunks}。"
            )
        else:
            status = store.status(expected_chunks=len(bundle.chunks))
            print(
                json.dumps(
                    {
                        "collection": status.collection_name,
                        "model": status.model_name,
                        "stored_chunks": status.stored_chunks,
                        "expected_chunks": status.expected_chunks,
                        "synchronized": status.synchronized,
                        "message": status.message,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
    except (KnowledgeBaseError, VectorStoreError) as exc:
        print(f"错误：{exc}")
        return 1
    finally:
        try:
            store.close()
        except VectorStoreError as exc:
            print(f"关闭向量数据库失败：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
