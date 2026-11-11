"""把 PDF 页面切分为可检索、可引用页码的证据片段。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

from tools.pdf_reader import PDFPageText


@dataclass(frozen=True)
class EvidenceChunk:
    """一段来自真实 PDF 的原文证据。"""

    chunk_id: str
    document_name: str
    page_number: int
    text: str

    @property
    def citation(self) -> str:
        """返回适合展示给用户的来源信息。"""

        return f"{self.document_name}，第 {self.page_number} 页"


def chunk_pdf_pages(
    document_name: str,
    pages: Iterable[PDFPageText],
    *,
    max_chars: int = 700,
    overlap_chars: int = 120,
) -> tuple[EvidenceChunk, ...]:
    """按页切分文本，保留文件名和页码。

    切分优先选择句末标点或换行，减少一句话被截断的情况。相邻片段保留少量
    重叠内容，避免答案恰好落在两个片段之间。
    """

    clean_name = document_name.strip()
    if not clean_name:
        raise ValueError("PDF 文件名不能为空。")
    if max_chars < 100:
        raise ValueError("证据片段长度不能小于 100 个字符。")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("重叠字符数必须大于等于 0 且小于片段长度。")

    page_list = tuple(pages)
    document_key = _document_key(clean_name, page_list)
    chunks: list[EvidenceChunk] = []

    for page in page_list:
        normalized_text = _normalize_text(page.text)
        if not normalized_text:
            continue
        for chunk_index, part in enumerate(
            _split_text(normalized_text, max_chars, overlap_chars),
            start=1,
        ):
            chunks.append(
                EvidenceChunk(
                    chunk_id=(
                        f"{document_key}-p{page.page_number}-c{chunk_index}"
                    ),
                    document_name=clean_name,
                    page_number=page.page_number,
                    text=part,
                )
            )

    return tuple(chunks)


def _document_key(document_name: str, pages: tuple[PDFPageText, ...]) -> str:
    """为文档生成稳定短标识，避免同名文件互相覆盖。"""

    digest = hashlib.sha1(
        f"{document_name}|{[page.text for page in pages]}".encode("utf-8")
    ).hexdigest()
    return digest[:10]


def _normalize_text(text: str) -> str:
    """清理多余空白，同时保留段落边界。"""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """按字符上限切分，但尽量在句号、问号或换行处结束。"""

    parts: list[str] = []
    start = 0

    while start < len(text):
        hard_end = min(start + max_chars, len(text))
        if hard_end == len(text):
            end = hard_end
        else:
            search_start = start + max_chars // 2
            end = _find_natural_break(text, search_start, hard_end)

        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
        if end >= len(text):
            break

        next_start = max(end - overlap_chars, start + 1)
        start = next_start

    return parts


def _find_natural_break(text: str, search_start: int, hard_end: int) -> int:
    """在一个窗口的后半段寻找自然断点，找不到时按字符上限切分。"""

    window = text[search_start:hard_end]
    punctuation_matches = list(re.finditer(r"[。！？!?；;]\s*|\n+", window))
    if punctuation_matches:
        return search_start + punctuation_matches[-1].end()

    space_index = window.rfind(" ")
    if space_index > 0:
        return search_start + space_index + 1
    return hard_end
