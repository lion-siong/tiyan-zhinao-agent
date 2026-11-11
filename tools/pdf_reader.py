"""PDF 文献文本提取工具。"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover - 兼容旧版 PyMuPDF
    try:
        import fitz  # type: ignore[no-redef]
    except ImportError:
        fitz = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PDFPageText:
    """PDF 单页文本，保留页码供证据引用。"""

    page_number: int
    text: str


@dataclass(frozen=True)
class PDFExtractionResult:
    """PDF 提取结果。"""

    success: bool
    text: str = ""
    pages: tuple[PDFPageText, ...] = ()
    page_count: int = 0
    char_count: int = 0
    processed_pages: int = 0
    truncated: bool = False
    message: str = ""


def extract_pdf_text(
    file_bytes: bytes,
    max_pages: int = 80,
    max_chars: int = 40_000,
) -> PDFExtractionResult:
    """从 PDF 字节流提取文本，异常会转换为可展示的结果对象。

    该函数不会写入临时文件，也不会自动编造缺失的文本。扫描版 PDF 通常
    需要 OCR，第一阶段不包含 OCR 能力。
    """

    if not file_bytes:
        return PDFExtractionResult(
            success=False,
            message="PDF 文件为空，请重新选择文件。",
        )
    if fitz is None:
        return PDFExtractionResult(
            success=False,
            message="缺少 PyMuPDF 依赖，请先运行 pip install -r requirements.txt。",
        )
    if max_pages <= 0 or max_chars <= 0:
        return PDFExtractionResult(
            success=False,
            message="PDF 页数或字符数限制必须是正整数。",
        )

    document = None
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
        if document.needs_pass:
            return PDFExtractionResult(
                success=False,
                page_count=document.page_count,
                message="该 PDF 已加密，请先解除密码后再上传。",
            )

        page_count = document.page_count
        if page_count == 0:
            return PDFExtractionResult(
                success=False,
                message="PDF 中没有可读取的页面。",
            )

        processed_pages = min(page_count, max_pages)
        text_parts: list[str] = []
        page_texts: list[PDFPageText] = []
        used_chars = 0
        truncated = page_count > max_pages

        for page_index in range(processed_pages):
            page_text = document.load_page(page_index).get_text("text").strip()
            if not page_text:
                continue

            remaining = max_chars - used_chars
            if remaining <= 0:
                truncated = True
                break

            prefix = f"\n\n--- 第 {page_index + 1} 页 ---\n"
            if remaining <= len(prefix):
                truncated = True
                break

            available_text = remaining - len(prefix)
            page_content = page_text[:available_text].strip()
            section = f"{prefix}{page_content}"
            if len(page_text) > available_text:
                truncated = True
            text_parts.append(section)
            page_texts.append(
                PDFPageText(page_number=page_index + 1, text=page_content)
            )
            used_chars += len(section)

        extracted_text = "".join(text_parts).strip()
        if not extracted_text:
            return PDFExtractionResult(
                success=True,
                text="",
                page_count=page_count,
                processed_pages=processed_pages,
                truncated=truncated,
                message="PDF 可以打开，但没有提取到可复制文本，可能是扫描版文件。",
            )

        message = "文本提取完成。"
        if truncated:
            message += " 内容较长，已按页数或字符上限截取。"
        return PDFExtractionResult(
            success=True,
            text=extracted_text,
            pages=tuple(page_texts),
            page_count=page_count,
            char_count=len(extracted_text),
            processed_pages=processed_pages,
            truncated=truncated,
            message=message,
        )
    except Exception as exc:
        return PDFExtractionResult(
            success=False,
            message=f"PDF 无法读取，文件可能已损坏或格式不受支持。详细信息：{exc}",
        )
    finally:
        if document is not None:
            document.close()
