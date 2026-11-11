"""项目内置文献库的清单读取、索引构建和完整性检查工具。

构建索引：
    python -m tools.knowledge_base --build

查看索引状态：
    python -m tools.knowledge_base
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import settings
from tools.document_chunker import EvidenceChunk, chunk_pdf_pages
from tools.pdf_reader import extract_pdf_text


MANIFEST_FILE_NAME = "manifest.json"
INDEX_FILE_PATH = Path("index") / "evidence_chunks.json"
SCHEMA_VERSION = 1
DOCUMENT_PDF_DIR = "pdfs"
INDEX_MAX_CHARS = 10_000_000


class KnowledgeBaseError(Exception):
    """内置文献库无法读取、校验或构建时抛出的可展示错误。"""


@dataclass(frozen=True)
class KnowledgeDocument:
    """一篇已登记文献的可追溯元数据。"""

    document_id: str
    file_name: str
    title: str
    subtitle: str
    authors: tuple[str, ...]
    journal: str
    year: str
    volume: str
    issue: str
    pages: str
    doi: str
    source_url: str
    keywords: tuple[str, ...]
    license: str
    page_count: int
    pdf_sha256: str

    @property
    def authors_text(self) -> str:
        """返回适合界面展示的作者字符串。"""

        return "、".join(self.authors)

    @property
    def reference(self) -> str:
        """返回根据清单元数据拼成的论文引用。"""

        subtitle = f"：{self.subtitle.lstrip('—')}" if self.subtitle else ""
        volume_issue = self.volume
        if self.issue != "无":
            volume_issue = f"{volume_issue}({self.issue})"
        return (
            f"{self.authors_text}. {self.title}{subtitle}[J]. "
            f"{self.journal}, {self.year}, {volume_issue}: "
            f"{self.pages}. DOI: {self.doi}"
        )


@dataclass(frozen=True)
class BundledKnowledgeBase:
    """已加载的内置文献清单和证据片段。"""

    library_name: str
    description: str
    documents: tuple[KnowledgeDocument, ...]
    chunks: tuple[EvidenceChunk, ...]
    built_at: str


def load_bundled_knowledge_base(
    base_dir: Path | str | None = None,
    *,
    verify_pdf_hashes: bool = True,
) -> BundledKnowledgeBase:
    """读取内置文献清单和持久化证据索引。

    默认校验 PDF 的 SHA-256，防止清单和实际文件不一致。界面可捕获
    KnowledgeBaseError 并继续提供上传 PDF 的功能。
    """

    root = _resolve_base_dir(base_dir)
    manifest_path, manifest = _read_manifest(root)
    documents = _parse_documents(
        manifest,
        root,
        verify_pdf_hashes=verify_pdf_hashes,
    )
    index_path = root / INDEX_FILE_PATH
    index_data = _read_json_object(index_path, "证据索引")

    if index_data.get("schema_version") != SCHEMA_VERSION:
        raise KnowledgeBaseError("证据索引版本不受支持，请重新构建索引。")
    if index_data.get("manifest_sha256") != _sha256_file(manifest_path):
        raise KnowledgeBaseError("文献清单与证据索引不一致，请重新构建索引。")

    pages_by_file = {
        document.file_name: document.page_count for document in documents
    }
    raw_chunks = index_data.get("chunks")
    if not isinstance(raw_chunks, list):
        raise KnowledgeBaseError("证据索引缺少 chunks 列表。")

    chunks: list[EvidenceChunk] = []
    seen_chunk_ids: set[str] = set()
    for item in raw_chunks:
        if not isinstance(item, dict):
            raise KnowledgeBaseError("证据索引中存在格式错误的片段。")
        chunk = EvidenceChunk(
            chunk_id=_required_text(item, "chunk_id", "证据片段"),
            document_name=_required_text(item, "document_name", "证据片段"),
            page_number=_required_positive_int(item, "page_number", "证据片段"),
            text=_required_text(item, "text", "证据片段"),
        )
        if chunk.chunk_id in seen_chunk_ids:
            raise KnowledgeBaseError(f"证据片段 ID 重复：{chunk.chunk_id}")
        if chunk.document_name not in pages_by_file:
            raise KnowledgeBaseError(
                f"证据片段引用了未登记文献：{chunk.document_name}"
            )
        if chunk.page_number > pages_by_file[chunk.document_name]:
            raise KnowledgeBaseError(
                f"证据片段页码超出文献范围：{chunk.chunk_id}"
            )
        seen_chunk_ids.add(chunk.chunk_id)
        chunks.append(chunk)

    expected_count = index_data.get("chunk_count")
    if expected_count != len(chunks):
        raise KnowledgeBaseError("证据索引片段数量与 chunk_count 不一致。")
    if not chunks:
        raise KnowledgeBaseError("证据索引为空，请重新构建索引。")

    return BundledKnowledgeBase(
        library_name=_required_text(manifest, "library_name", "文献清单"),
        description=_required_text(manifest, "description", "文献清单"),
        documents=documents,
        chunks=tuple(chunks),
        built_at=str(index_data.get("built_at", "")),
    )


def build_evidence_corpus(base_dir: Path | str | None = None) -> BundledKnowledgeBase:
    """从清单中的 PDF 重新提取全文并写入证据索引。"""

    root = _resolve_base_dir(base_dir)
    manifest_path, manifest = _read_manifest(root)
    documents = _parse_documents(
        manifest,
        root,
        verify_pdf_hashes=True,
    )

    all_chunks: list[EvidenceChunk] = []
    for document in documents:
        pdf_path = _pdf_path(root, document.file_name)
        result = extract_pdf_text(
            pdf_path.read_bytes(),
            max_pages=document.page_count,
            max_chars=INDEX_MAX_CHARS,
        )
        if not result.success:
            raise KnowledgeBaseError(
                f"{document.file_name} 无法建立索引：{result.message}"
            )
        if result.page_count != document.page_count:
            raise KnowledgeBaseError(
                f"{document.file_name} 的实际页数为 {result.page_count}，"
                f"清单登记为 {document.page_count}。"
            )
        if result.truncated:
            raise KnowledgeBaseError(
                f"{document.file_name} 在建立索引时被截断，请提高索引字符上限。"
            )
        if not result.pages:
            raise KnowledgeBaseError(
                f"{document.file_name} 没有可提取文本，可能是扫描版 PDF。"
            )

        document_chunks = chunk_pdf_pages(
            document.file_name,
            result.pages,
            max_chars=settings.rag_chunk_chars,
            overlap_chars=settings.rag_chunk_overlap,
        )
        if not document_chunks:
            raise KnowledgeBaseError(
                f"{document.file_name} 未生成有效证据片段。"
            )
        all_chunks.extend(document_chunks)

    built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    index_data = {
        "schema_version": SCHEMA_VERSION,
        "built_at": built_at,
        "manifest_sha256": _sha256_file(manifest_path),
        "chunk_count": len(all_chunks),
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "text": chunk.text,
            }
            for chunk in all_chunks
        ],
    }
    index_path = root / INDEX_FILE_PATH
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(index_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return BundledKnowledgeBase(
        library_name=_required_text(manifest, "library_name", "文献清单"),
        description=_required_text(manifest, "description", "文献清单"),
        documents=documents,
        chunks=tuple(all_chunks),
        built_at=built_at,
    )


def _resolve_base_dir(base_dir: Path | str | None) -> Path:
    """返回内置文献库根目录，并确认目录存在。"""

    root = Path(base_dir or settings.knowledge_base_dir).resolve()
    if not root.exists():
        raise KnowledgeBaseError(f"知识库目录不存在：{root}")
    return root


def _read_manifest(root: Path) -> tuple[Path, dict[str, Any]]:
    """读取 manifest.json 并检查顶层版本字段。"""

    manifest_path = root / MANIFEST_FILE_NAME
    manifest = _read_json_object(manifest_path, "文献清单")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise KnowledgeBaseError("文献清单版本不受支持。")
    return manifest_path, manifest


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    """读取 UTF-8 JSON 对象，把底层错误转换成中文提示。"""

    if not path.exists():
        raise KnowledgeBaseError(
            f"{label}不存在：{path}。请先运行 "
            "'python -m tools.knowledge_base --build'。"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise KnowledgeBaseError(f"{label}无法读取：{exc}") from exc
    if not isinstance(data, dict):
        raise KnowledgeBaseError(f"{label}必须是 JSON 对象。")
    return data


def _parse_documents(
    manifest: dict[str, Any],
    root: Path,
    *,
    verify_pdf_hashes: bool,
) -> tuple[KnowledgeDocument, ...]:
    """解析并校验文献清单中的每一篇文档。"""

    raw_documents = manifest.get("documents")
    if not isinstance(raw_documents, list) or not raw_documents:
        raise KnowledgeBaseError("文献清单中没有 documents。")

    documents: list[KnowledgeDocument] = []
    seen_ids: set[str] = set()
    seen_files: set[str] = set()
    for item in raw_documents:
        if not isinstance(item, dict):
            raise KnowledgeBaseError("文献清单中存在格式错误的文档。")
        document = _parse_document(item)
        if document.document_id in seen_ids:
            raise KnowledgeBaseError(f"文献 ID 重复：{document.document_id}")
        if document.file_name in seen_files:
            raise KnowledgeBaseError(f"PDF 文件名重复：{document.file_name}")
        seen_ids.add(document.document_id)
        seen_files.add(document.file_name)

        pdf_path = _pdf_path(root, document.file_name)
        if not pdf_path.exists():
            raise KnowledgeBaseError(f"清单中的 PDF 不存在：{pdf_path}")
        if verify_pdf_hashes and _sha256_file(pdf_path) != document.pdf_sha256:
            raise KnowledgeBaseError(
                f"{document.file_name} 的 SHA-256 与清单不一致，"
                "请核对文件后重新构建索引。"
            )
        documents.append(document)

    return tuple(documents)


def _parse_document(item: dict[str, Any]) -> KnowledgeDocument:
    """把一条 JSON 文档记录转换为带类型的对象。"""

    file_name = _required_text(item, "file_name", "文献记录")
    if Path(file_name).name != file_name or "/" in file_name or "\\" in file_name:
        raise KnowledgeBaseError(f"PDF 文件名不合法：{file_name}")
    authors = _required_text_list(item, "authors", "文献记录")
    keywords = _required_text_list(item, "keywords", "文献记录")
    return KnowledgeDocument(
        document_id=_required_text(item, "document_id", "文献记录"),
        file_name=file_name,
        title=_required_text(item, "title", "文献记录"),
        subtitle=str(item.get("subtitle", "")).strip(),
        authors=authors,
        journal=_required_text(item, "journal", "文献记录"),
        year=_required_text(item, "year", "文献记录"),
        volume=_required_text(item, "volume", "文献记录"),
        issue=_required_text(item, "issue", "文献记录"),
        pages=_required_text(item, "pages", "文献记录"),
        doi=_required_text(item, "doi", "文献记录"),
        source_url=_required_text(item, "source_url", "文献记录"),
        keywords=keywords,
        license=_required_text(item, "license", "文献记录"),
        page_count=_required_positive_int(item, "page_count", "文献记录"),
        pdf_sha256=_required_text(item, "pdf_sha256", "文献记录").lower(),
    )


def _required_text(item: dict[str, Any], key: str, label: str) -> str:
    """读取必填非空字符串。"""

    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeBaseError(f"{label}缺少有效字段：{key}")
    return value.strip()


def _required_text_list(
    item: dict[str, Any],
    key: str,
    label: str,
) -> tuple[str, ...]:
    """读取必填的非空字符串列表。"""

    value = item.get(key)
    if not isinstance(value, list):
        raise KnowledgeBaseError(f"{label}字段 {key} 必须是列表。")
    cleaned = tuple(
        entry.strip()
        for entry in value
        if isinstance(entry, str) and entry.strip()
    )
    if len(cleaned) != len(value) or not cleaned:
        raise KnowledgeBaseError(f"{label}字段 {key} 包含空值或非字符串。")
    return cleaned


def _required_positive_int(
    item: dict[str, Any],
    key: str,
    label: str,
) -> int:
    """读取必填的正整数。"""

    value = item.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise KnowledgeBaseError(f"{label}字段 {key} 必须是正整数。")
    return value


def _pdf_path(root: Path, file_name: str) -> Path:
    """返回 PDF 的受控路径，避免清单通过相对路径访问库外文件。"""

    return root / DOCUMENT_PDF_DIR / file_name


def _sha256_file(path: Path) -> str:
    """分块计算文件 SHA-256，避免一次读入大文件。"""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise KnowledgeBaseError(f"无法读取文件 {path}：{exc}") from exc
    return digest.hexdigest()


def main() -> int:
    """命令行入口，默认检查文献库，传入 --build 时重建索引。"""

    parser = argparse.ArgumentParser(description="检查或构建体研智脑内置文献库")
    parser.add_argument(
        "--build",
        action="store_true",
        help="从 manifest.json 中的 PDF 重新生成证据索引",
    )
    args = parser.parse_args()

    try:
        if args.build:
            bundle = build_evidence_corpus()
            action = "已重新构建"
        else:
            bundle = load_bundled_knowledge_base()
            action = "已加载"
    except KnowledgeBaseError as exc:
        print(f"错误：{exc}")
        return 1

    print(
        f"{action}内置文献库：{len(bundle.documents)} 篇文档，"
        f"{len(bundle.chunks)} 个证据片段。"
    )
    for document in bundle.documents:
        print(f"- {document.title}（{document.file_name}，{document.page_count} 页）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
