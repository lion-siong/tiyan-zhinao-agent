"""把新 PDF 自动加入文献清单、证据索引和研究框架。

导入过程会校验文件、去重、保存 PDF、更新 manifest.json，然后一次性重建证据
索引和 framework.json。中途失败时会恢复导入前的文件。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import settings
from services.vector_service import (
    EvidenceVectorStore,
    VectorStoreError,
    VectorSyncResult,
    sync_bundled_vectors,
)
from tools.knowledge_base import (
    INDEX_FILE_PATH,
    SCHEMA_VERSION,
    BundledKnowledgeBase,
    KnowledgeBaseError,
    build_evidence_corpus,
    load_bundled_knowledge_base,
)
from tools.literature_framework import (
    FRAMEWORK_FILE_NAME,
    LiteratureFramework,
    build_and_save_literature_framework,
    load_literature_framework,
)
from tools.pdf_metadata import PDFMetadata, extract_pdf_metadata
from tools.pdf_reader import extract_pdf_text


DOCUMENT_PDF_DIR = "pdfs"
INGEST_MAX_CHARS = 10_000_000


@dataclass(frozen=True)
class IncomingPDF:
    """尚未写入文献库的 PDF 内容。"""

    file_name: str
    data: bytes


@dataclass(frozen=True)
class ImportedDocument:
    """成功导入的一篇文献。"""

    document_id: str
    file_name: str
    title: str


@dataclass(frozen=True)
class IngestResult:
    """一次批量导入的可展示结果。"""

    imported: tuple[ImportedDocument, ...]
    duplicates: tuple[str, ...]
    skipped: tuple[str, ...]
    warnings: tuple[str, ...]
    framework: LiteratureFramework
    vector_sync: VectorSyncResult | None = None
    vector_warning: str = ""

    @property
    def imported_count(self) -> int:
        return len(self.imported)


@dataclass(frozen=True)
class _PreparedPDF:
    """通过校验、准备写入文献库的 PDF。"""

    metadata: PDFMetadata
    final_file_name: str
    data: bytes


@dataclass(frozen=True)
class RebuildResult:
    """证据索引、研究框架和向量库的联合重建结果。"""

    bundle: BundledKnowledgeBase
    framework: LiteratureFramework
    vector_sync: VectorSyncResult | None = None
    vector_warning: str = ""


def ingest_pdf_files(
    incoming_files: list[IncomingPDF] | tuple[IncomingPDF, ...],
    base_dir: Path | str | None = None,
    *,
    auto_vector_sync: bool = True,
) -> IngestResult:
    """导入一批 PDF，并自动重建证据索引与研究框架。"""

    root = _resolve_root(base_dir)
    manifest_path = root / "manifest.json"
    manifest = _read_manifest(manifest_path)
    raw_documents = manifest.get("documents")
    if not isinstance(raw_documents, list):
        raise KnowledgeBaseError("文献清单缺少 documents 列表。")

    existing_hashes = {
        str(item.get("pdf_sha256", "")).lower()
        for item in raw_documents
        if isinstance(item, dict)
    }
    existing_ids = {
        str(item.get("document_id", ""))
        for item in raw_documents
        if isinstance(item, dict)
    }
    used_file_names = {
        str(item.get("file_name", ""))
        for item in raw_documents
        if isinstance(item, dict)
    }

    prepared: list[_PreparedPDF] = []
    duplicates: list[str] = []
    skipped: list[str] = []
    warnings: list[str] = []
    max_bytes = settings.max_pdf_size_mb * 1024 * 1024

    for incoming in incoming_files:
        display_name = Path(incoming.file_name).name or "未命名.pdf"
        if not incoming.data:
            skipped.append(f"{display_name}：文件为空。")
            continue
        if len(incoming.data) > max_bytes:
            skipped.append(
                f"{display_name}：超过 {settings.max_pdf_size_mb} MB 限制。"
            )
            continue

        file_hash = hashlib.sha256(incoming.data).hexdigest()
        if file_hash in existing_hashes:
            duplicates.append(f"{display_name}：相同 PDF 已在文献库中。")
            continue

        try:
            metadata = extract_pdf_metadata(incoming.data, display_name)
        except (OSError, ValueError, RuntimeError) as exc:
            skipped.append(f"{display_name}：无法识别书目信息，{exc}")
            continue

        if metadata.pdf_sha256 in existing_hashes:
            duplicates.append(f"{display_name}：相同 PDF 已在文献库中。")
            continue
        if metadata.document_id in existing_ids:
            duplicates.append(f"{display_name}：DOI 或文献标识已存在。")
            continue

        extraction = extract_pdf_text(
            incoming.data,
            max_pages=metadata.page_count,
            max_chars=INGEST_MAX_CHARS,
        )
        if not extraction.success or not extraction.pages:
            skipped.append(
                f"{display_name}：没有可建立索引的文本，可能是扫描版 PDF。"
            )
            continue
        if extraction.truncated:
            skipped.append(
                f"{display_name}：正文超过索引字符上限，请拆分或压缩后再导入。"
            )
            continue

        final_file_name = _unique_pdf_name(
            display_name,
            metadata.pdf_sha256,
            used_file_names,
        )
        used_file_names.add(final_file_name)
        existing_hashes.add(metadata.pdf_sha256)
        existing_ids.add(metadata.document_id)
        prepared.append(
            _PreparedPDF(
                metadata=metadata,
                final_file_name=final_file_name,
                data=incoming.data,
            )
        )
        warnings.extend(
            f"{display_name}：{warning}" for warning in metadata.warnings
        )

    if not prepared:
        load_bundled_knowledge_base(root)
        framework = load_literature_framework(root)
        return IngestResult(
            imported=(),
            duplicates=tuple(duplicates),
            skipped=tuple(skipped),
            warnings=tuple(warnings),
            framework=framework,
        )

    manifest_before = manifest_path.read_bytes()
    index_path = root / INDEX_FILE_PATH
    framework_path = root / FRAMEWORK_FILE_NAME
    index_before = index_path.read_bytes() if index_path.exists() else None
    framework_before = (
        framework_path.read_bytes() if framework_path.exists() else None
    )
    written_files: list[Path] = []

    try:
        pdf_directory = root / DOCUMENT_PDF_DIR
        pdf_directory.mkdir(parents=True, exist_ok=True)
        for item in prepared:
            destination = pdf_directory / item.final_file_name
            destination.write_bytes(item.data)
            written_files.append(destination)
            raw_documents.append(
                item.metadata.to_manifest_record(item.final_file_name)
            )

        _write_json_atomic(manifest_path, manifest)
        bundle = build_evidence_corpus(root)
        framework = build_and_save_literature_framework(bundle, root)
        vector_sync, vector_warning = _try_sync_vectors(
            bundle,
            root,
            enabled=auto_vector_sync,
        )
    except Exception as exc:
        _restore_file(manifest_path, manifest_before)
        _restore_optional_file(index_path, index_before)
        _restore_optional_file(framework_path, framework_before)
        for path in written_files:
            path.unlink(missing_ok=True)
        if isinstance(exc, KnowledgeBaseError):
            raise
        raise KnowledgeBaseError(f"论文导入失败，已恢复原文献库：{exc}") from exc

    imported = tuple(
        ImportedDocument(
            document_id=item.metadata.document_id,
            file_name=item.final_file_name,
            title=item.metadata.title,
        )
        for item in prepared
    )
    return IngestResult(
        imported=imported,
        duplicates=tuple(duplicates),
        skipped=tuple(skipped),
        warnings=tuple(warnings),
        framework=framework,
        vector_sync=vector_sync,
        vector_warning=vector_warning,
    )


def sync_pdf_directory(
    base_dir: Path | str | None = None,
    *,
    auto_vector_sync: bool = True,
) -> IngestResult:
    """扫描 knowledge_base/pdfs，把清单中尚未登记的新 PDF 自动加入。"""

    root = _resolve_root(base_dir)
    manifest = _read_manifest(root / "manifest.json")
    raw_documents = manifest.get("documents", [])
    known_hashes = {
        str(item.get("pdf_sha256", "")).lower()
        for item in raw_documents
        if isinstance(item, dict)
    }

    incoming: list[IncomingPDF] = []
    for path in sorted((root / DOCUMENT_PDF_DIR).glob("*.pdf")):
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise KnowledgeBaseError(f"无法读取 PDF：{path}，{exc}") from exc
        if hashlib.sha256(data).hexdigest() not in known_hashes:
            incoming.append(IncomingPDF(path.name, data))

    return ingest_pdf_files(
        incoming,
        root,
        auto_vector_sync=auto_vector_sync,
    )


def rebuild_library(
    base_dir: Path | str | None = None,
    *,
    auto_vector_sync: bool = True,
) -> RebuildResult:
    """重新生成证据索引、研究框架和向量库。"""

    root = _resolve_root(base_dir)
    bundle = build_evidence_corpus(root)
    framework = build_and_save_literature_framework(bundle, root)
    vector_sync, vector_warning = _try_sync_vectors(
        bundle,
        root,
        enabled=auto_vector_sync,
    )
    return RebuildResult(
        bundle=bundle,
        framework=framework,
        vector_sync=vector_sync,
        vector_warning=vector_warning,
    )


def _try_sync_vectors(
    bundle: BundledKnowledgeBase,
    root: Path,
    *,
    enabled: bool,
) -> tuple[VectorSyncResult | None, str]:
    """尝试同步向量库；失败时保留已成功构建的关键词索引和框架。"""

    if not enabled:
        return None, ""
    store = EvidenceVectorStore(
        root,
        model_name=settings.embedding_model,
        collection_name=settings.chroma_collection_name,
    )
    try:
        result = sync_bundled_vectors(
            bundle,
            base_dir=root,
            vector_store=store,
        )
    except Exception as exc:
        return None, (
            f"关键词索引和研究框架已更新，但向量库同步失败：{exc}"
            "当前仍可使用 BM25 检索。"
        )
    finally:
        try:
            store.close()
        except Exception:
            pass
    return result, ""


def _read_manifest(path: Path) -> dict[str, Any]:
    """读取并检查 manifest.json。"""

    if not path.exists():
        raise KnowledgeBaseError(f"文献清单不存在：{path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise KnowledgeBaseError(f"文献清单无法读取：{exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise KnowledgeBaseError("文献清单格式或版本不受支持。")
    return data


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """先写临时文件再替换，降低写入中断造成清单损坏的风险。"""

    temporary_path = path.with_suffix(".json.tmp")
    try:
        temporary_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _restore_file(path: Path, content: bytes) -> None:
    """恢复一个必须存在的文件。"""

    temporary_path = path.with_suffix(path.suffix + ".restore")
    temporary_path.write_bytes(content)
    temporary_path.replace(path)


def _restore_optional_file(path: Path, content: bytes | None) -> None:
    """恢复可选文件；如果导入前不存在则删除。"""

    if content is None:
        path.unlink(missing_ok=True)
    else:
        _restore_file(path, content)


def _unique_pdf_name(
    original_name: str,
    file_hash: str,
    used_names: set[str],
) -> str:
    """生成安全且不冲突的 PDF 文件名。"""

    raw_name = Path(original_name).name.strip() or "文献.pdf"
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", raw_name)
    sanitized = sanitized.rstrip(" .")
    stem = Path(sanitized).stem[:120] or "文献"
    if stem.upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "LPT1",
        "LPT2",
        "LPT3",
    }:
        stem = f"_{stem}"

    candidate = f"{stem}.pdf"
    if candidate not in used_names:
        return candidate

    candidate = f"{stem}-{file_hash[:8]}.pdf"
    if candidate not in used_names:
        return candidate
    return f"{stem}-{file_hash[:16]}.pdf"


def _resolve_root(base_dir: Path | str | None) -> Path:
    """返回并检查知识库目录。"""

    root = Path(base_dir or settings.knowledge_base_dir).resolve()
    if not root.exists():
        raise KnowledgeBaseError(f"知识库目录不存在：{root}")
    if not (root / DOCUMENT_PDF_DIR).exists():
        (root / DOCUMENT_PDF_DIR).mkdir(parents=True, exist_ok=True)
    return root
