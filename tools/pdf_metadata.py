"""从 PDF 中自动提取可用于文献清单的书目信息。

这里只读取 PDF 自带的元数据和前几页文本，不会联网补全字段。无法识别的字段
会使用“待核对”或“待补充”，并在返回值中给出明确警告。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover - 兼容旧版 PyMuPDF
    try:
        import fitz  # type: ignore[no-redef]
    except ImportError:
        fitz = None  # type: ignore[assignment]


_DOI_PATTERN = re.compile(r"10\.\d{4,9}/[^\s<>\"']+", re.IGNORECASE)
_CITATION_PATTERN = re.compile(
    r"(?m)^(?P<header>.{2,220}?),\s*"
    r"(?P<year>20\d{2}),\s*"
    r"(?P<volume>\d+)\s*[（(](?P<issue>\d+)[）)]\s*,\s*"
    r"(?P<start>\d+)\s*[-–—]\s*(?P<end>\d+)"
)
_CHINESE_JOURNAL_PATTERN = re.compile(r"[\u4e00-\u9fff·]{2,40}")


@dataclass(frozen=True)
class PDFMetadata:
    """自动识别出的论文元数据及需要人工核对的事项。"""

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
    warnings: tuple[str, ...] = ()

    @property
    def document_id(self) -> str:
        """生成稳定且不依赖文件名的文献标识。"""

        if self.doi != "待补充":
            safe_doi = re.sub(r"[^a-z0-9]+", "-", self.doi.lower()).strip("-")
            if safe_doi:
                return f"doi-{safe_doi}"
        return f"pdf-{self.pdf_sha256[:12]}"

    def to_manifest_record(self, final_file_name: str) -> dict[str, object]:
        """转换为 manifest.json 中的文档记录。"""

        return {
            "document_id": self.document_id,
            "file_name": final_file_name,
            "title": self.title,
            "subtitle": self.subtitle,
            "authors": list(self.authors),
            "journal": self.journal,
            "year": self.year,
            "volume": self.volume,
            "issue": self.issue,
            "pages": self.pages,
            "doi": self.doi,
            "source_url": self.source_url,
            "keywords": list(self.keywords),
            "license": self.license,
            "page_count": self.page_count,
            "pdf_sha256": self.pdf_sha256,
        }


def extract_pdf_metadata(file_bytes: bytes, file_name: str) -> PDFMetadata:
    """从 PDF 字节流提取题名、作者、期刊、DOI 等字段。

    对无法确认的内容不进行猜测，而是填入占位文本并返回警告，方便用户在导入
    后人工核对。
    """

    if not file_bytes:
        raise ValueError("PDF 文件为空，无法识别书目信息。")
    if fitz is None:
        raise ValueError("缺少 PyMuPDF 依赖，无法识别 PDF 书目信息。")

    clean_name = Path(file_name).name
    document = None
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
        if document.needs_pass:
            raise ValueError("PDF 已加密，无法自动识别书目信息。")
        if document.page_count <= 0:
            raise ValueError("PDF 中没有可读取的页面。")

        metadata = {
            str(key): str(value or "").strip()
            for key, value in (document.metadata or {}).items()
        }
        preview_pages = min(3, document.page_count)
        preview_text = "\n".join(
            document.load_page(index).get_text("text")
            for index in range(preview_pages)
        )

        title = _clean_text(metadata.get("title", ""))
        if not title:
            title = Path(clean_name).stem

        authors = _split_values(metadata.get("author", ""))
        keywords = _split_values(metadata.get("keywords", ""))
        citation = _extract_citation(preview_text)
        doi = _extract_doi(preview_text, metadata)

        warnings: list[str] = []
        if not metadata.get("title", "").strip():
            warnings.append("未读取到 PDF 题名，暂用文件名代替。")
        if not authors:
            authors = ("待核对",)
            warnings.append("未读取到作者，请导入后核对。")
        if citation is None:
            journal = "待核对"
            year = "待核对"
            volume = "待核对"
            issue = "待核对"
            pages = "待核对"
            warnings.append("未识别到期刊、年份、卷期和页码。")
        else:
            journal = citation["journal"]
            year = citation["year"]
            volume = citation["volume"]
            issue = citation["issue"]
            pages = citation["start"] + "-" + citation["end"]

        if doi == "待补充":
            source_url = "未提供"
            warnings.append("未识别到 DOI。")
        else:
            source_url = f"https://doi.org/{doi}"

        if not keywords:
            keywords = ("待补充",)
            warnings.append("未读取到关键词。")

        license_text = "CC BY 4.0" if "CC BY 4.0" in preview_text else "待核对"
        if license_text == "待核对":
            warnings.append("未识别到开放许可，请核对。")

        return PDFMetadata(
            file_name=clean_name,
            title=title,
            subtitle="",
            authors=authors,
            journal=journal,
            year=year,
            volume=volume,
            issue=issue,
            pages=pages,
            doi=doi,
            source_url=source_url,
            keywords=keywords,
            license=license_text,
            page_count=document.page_count,
            pdf_sha256=hashlib.sha256(file_bytes).hexdigest(),
            warnings=tuple(warnings),
        )
    finally:
        if document is not None:
            document.close()


def _extract_citation(text: str) -> dict[str, str] | None:
    """识别期刊首页中的“期刊, 年份, 卷(期), 起止页”格式。"""

    match = _CITATION_PATTERN.search(text)
    if match is None:
        return None

    header = match.group("header").strip()
    chinese_parts = _CHINESE_JOURNAL_PATTERN.findall(header)
    journal = chinese_parts[-1] if chinese_parts else header
    journal = _clean_text(journal)
    if not journal:
        return None
    return {
        "journal": journal,
        "year": match.group("year"),
        "volume": match.group("volume"),
        "issue": match.group("issue"),
        "start": match.group("start"),
        "end": match.group("end"),
    }


def _extract_doi(text: str, metadata: dict[str, str]) -> str:
    """优先从首页查找 DOI，找不到时返回明确占位符。"""

    for source in (text, " ".join(metadata.values())):
        match = _DOI_PATTERN.search(source)
        if match is not None:
            return match.group(0).rstrip(".,;，。；）)]}").strip()
    return "待补充"


def _split_values(value: str) -> tuple[str, ...]:
    """拆分作者或关键词，并清理中文姓名中的排版空格。"""

    parts = [
        _clean_text(part)
        for part in re.split(r"[，,、;；]+", value)
        if part.strip()
    ]
    return tuple(part for part in parts if part)


def _clean_text(value: str) -> str:
    """压缩 PDF 元数据中的换行和连续空白。"""

    return re.sub(r"\s+", " ", value).strip()
