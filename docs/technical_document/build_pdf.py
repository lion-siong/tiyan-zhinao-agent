"""Build the competition PDF from the editable Markdown source."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pymupdf
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    LongTable,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    TableStyle,
    XPreformatted,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "体研智脑Agent_技术文档.md"
OUTPUT = ROOT / "体研智脑Agent_技术文档.pdf"
REPORT = ROOT / "文档生成检查.json"

PAGE_WIDTH, PAGE_HEIGHT = A4
LEFT = RIGHT = 2.0 * cm
TOP = 2.35 * cm
BOTTOM = 2.0 * cm
CONTENT_WIDTH = PAGE_WIDTH - LEFT - RIGHT

NAVY = colors.HexColor("#0F2F66")
BLUE = colors.HexColor("#2459A6")
LIGHT_BLUE = colors.HexColor("#EAF2FF")
CYAN = colors.HexColor("#16A6A1")
LIGHT_CYAN = colors.HexColor("#EAF8F7")
INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#5E6A7D")
LINE = colors.HexColor("#CBD5E1")
PAPER = colors.HexColor("#F7FAFD")


def register_fonts() -> None:
    pdfmetrics.registerFont(
        TTFont("BodyCN", r"C:\Windows\Fonts\simsun.ttc")
    )
    pdfmetrics.registerFont(
        TTFont("HeadingCN", r"C:\Windows\Fonts\simhei.ttf")
    )
    pdfmetrics.registerFont(
        TTFont("SansCN", r"C:\Windows\Fonts\msyh.ttc")
    )
    pdfmetrics.registerFont(
        TTFont("SansCNBold", r"C:\Windows\Fonts\msyhbd.ttc")
    )
    pdfmetrics.registerFont(
        TTFont("Code", r"C:\Windows\Fonts\consola.ttf")
    )
    pdfmetrics.registerFont(
        TTFont("CodeBold", r"C:\Windows\Fonts\consolab.ttf")
    )
    pdfmetrics.registerFontFamily(
        "BodyCN",
        normal="BodyCN",
        bold="HeadingCN",
        italic="BodyCN",
        boldItalic="HeadingCN",
    )
    pdfmetrics.registerFontFamily(
        "SansCN",
        normal="SansCN",
        bold="SansCNBold",
        italic="SansCN",
        boldItalic="SansCNBold",
    )


def make_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "Body",
            parent=sample["BodyText"],
            fontName="BodyCN",
            fontSize=10.2,
            leading=17.2,
            textColor=INK,
            alignment=TA_JUSTIFY,
            firstLineIndent=2 * 10.2,
            spaceAfter=6,
            wordWrap="CJK",
            splitLongWords=True,
            allowWidows=0,
            allowOrphans=0,
        ),
        "body_no_indent": ParagraphStyle(
            "BodyNoIndent",
            parent=sample["BodyText"],
            fontName="BodyCN",
            fontSize=10.2,
            leading=17.2,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
            wordWrap="CJK",
            splitLongWords=True,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=sample["BodyText"],
            fontName="BodyCN",
            fontSize=10.0,
            leading=16.8,
            textColor=INK,
            alignment=TA_JUSTIFY,
            leftIndent=15,
            firstLineIndent=-8,
            bulletIndent=4,
            spaceAfter=3,
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "Heading1",
            parent=sample["Heading1"],
            fontName="HeadingCN",
            fontSize=18,
            leading=25,
            textColor=NAVY,
            spaceBefore=0,
            spaceAfter=12,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "Heading2",
            parent=sample["Heading2"],
            fontName="SansCNBold",
            fontSize=12.6,
            leading=19,
            textColor=BLUE,
            spaceBefore=10,
            spaceAfter=5,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "abstract_title": ParagraphStyle(
            "AbstractTitle",
            parent=sample["Title"],
            fontName="HeadingCN",
            fontSize=21,
            leading=28,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=16,
            wordWrap="CJK",
        ),
        "abstract_keywords": ParagraphStyle(
            "AbstractKeywords",
            parent=sample["BodyText"],
            fontName="BodyCN",
            fontSize=9.8,
            leading=16,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceBefore=7,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=sample["BodyText"],
            fontName="SansCN",
            fontSize=8.8,
            leading=13,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=8,
            wordWrap="CJK",
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=sample["BodyText"],
            fontName="BodyCN",
            fontSize=7.5,
            leading=10.8,
            textColor=INK,
            alignment=TA_LEFT,
            wordWrap="CJK",
            splitLongWords=True,
        ),
        "table_head": ParagraphStyle(
            "TableHead",
            parent=sample["BodyText"],
            fontName="SansCNBold",
            fontSize=7.5,
            leading=10.8,
            textColor=colors.white,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "code": ParagraphStyle(
            "CodeBlock",
            parent=sample["Code"],
            fontName="Code",
            fontSize=7.8,
            leading=11,
            textColor=colors.HexColor("#E6EDF7"),
            leftIndent=5,
            rightIndent=5,
            borderPadding=8,
            backColor=colors.HexColor("#14213D"),
            borderColor=colors.HexColor("#294166"),
            borderWidth=0.6,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "toc1": ParagraphStyle(
            "TOC1",
            parent=sample["BodyText"],
            fontName="SansCNBold",
            fontSize=10.5,
            leading=17,
            textColor=NAVY,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=3,
        ),
        "toc2": ParagraphStyle(
            "TOC2",
            parent=sample["BodyText"],
            fontName="BodyCN",
            fontSize=9.2,
            leading=14.2,
            textColor=INK,
            leftIndent=15,
            firstLineIndent=0,
        ),
    }


def inline_markup(value: str) -> str:
    escaped = html.escape(value, quote=False)

    def code_font(match: re.Match[str]) -> str:
        content = match.group(1)
        font_name = (
            "SansCN"
            if re.search(r"[^\x00-\x7f]", content)
            else "Code"
        )
        return f'<font name="{font_name}" size="9">{content}</font>'

    escaped = re.sub(r"`([^`]+)`", code_font, escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    return escaped


def strip_markup(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value).replace("&amp;", "&")


class TechnicalDocument(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        cover_frame = Frame(
            6.7 * cm,
            BOTTOM,
            PAGE_WIDTH - 6.7 * cm - RIGHT,
            PAGE_HEIGHT - TOP - BOTTOM,
            id="cover-frame",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        content_frame = Frame(
            LEFT,
            BOTTOM,
            CONTENT_WIDTH,
            PAGE_HEIGHT - TOP - BOTTOM,
            id="content-frame",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(
            [
                PageTemplate(
                    id="cover",
                    frames=[cover_frame],
                    onPage=self.draw_cover,
                ),
                PageTemplate(
                    id="content",
                    frames=[content_frame],
                    onPage=self.draw_content_page,
                ),
            ]
        )
        self.first_content_page: int | None = None

    def handle_pageBegin(self):
        super().handle_pageBegin()
        if self.page > 1 and self.first_content_page is None:
            self.first_content_page = self.page

    def draw_cover(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, 5.8 * cm, PAGE_HEIGHT, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor("#133B7A"))
        canvas.circle(1.8 * cm, PAGE_HEIGHT - 1.9 * cm, 1.15 * cm, fill=1, stroke=0)
        canvas.setStrokeColor(colors.HexColor("#2C5DA8"))
        canvas.setLineWidth(0.7)
        canvas.circle(1.2 * cm, 5.2 * cm, 3.0 * cm, fill=0, stroke=1)
        canvas.setFillColor(colors.HexColor("#E4B95F"))
        canvas.rect(5.75 * cm, 0, 0.08 * cm, PAGE_HEIGHT, fill=1, stroke=0)
        canvas.setFont("SansCNBold", 16)
        canvas.setFillColor(colors.white)
        canvas.drawString(1.2 * cm, PAGE_HEIGHT - 2.7 * cm, "体研智脑")
        canvas.setFont("SansCN", 8)
        canvas.drawString(1.2 * cm, PAGE_HEIGHT - 3.2 * cm, "SPORTS RESEARCH AGENT")
        canvas.setFont("SansCN", 9)
        canvas.drawString(1.2 * cm, 2.2 * cm, "2026 · 技术创新文档")
        canvas.restoreState()

    def draw_content_page(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(PAPER)
        canvas.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)
        canvas.setFillColor(colors.white)
        canvas.rect(
            LEFT - 0.35 * cm,
            BOTTOM - 0.3 * cm,
            CONTENT_WIDTH + 0.7 * cm,
            PAGE_HEIGHT - TOP - BOTTOM + 0.55 * cm,
            fill=1,
            stroke=0,
        )
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.45)
        canvas.line(
            LEFT,
            PAGE_HEIGHT - 1.45 * cm,
            PAGE_WIDTH - RIGHT,
            PAGE_HEIGHT - 1.45 * cm,
        )
        canvas.setFont("SansCN", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(LEFT, PAGE_HEIGHT - 1.12 * cm, "体研智脑 Agent 技术文档")
        canvas.drawRightString(
            PAGE_WIDTH - RIGHT,
            PAGE_HEIGHT - 1.12 * cm,
            "传智杯 AI 大模型创新应用挑战赛",
        )
        canvas.line(
            LEFT,
            1.35 * cm,
            PAGE_WIDTH - RIGHT,
            1.35 * cm,
        )
        canvas.drawString(LEFT, 0.87 * cm, "基于真实代码与运行结果的竞赛技术文档")
        canvas.drawRightString(
            PAGE_WIDTH - RIGHT,
            0.87 * cm,
            f"第 {doc.page} 页",
        )
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        style = flowable.style.name
        if style not in {"Heading1", "Heading2"}:
            return
        if style != "Heading1":
            return
        level = 0
        text = strip_markup(flowable.getPlainText())
        key = f"heading-{self.page}-{level}-{self._heading_counter}"
        self._heading_counter += 1
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=False)
        self.notify("TOCEntry", (level, text, self.page))

    _heading_counter = 0


def cover_story(styles) -> list:
    title = ParagraphStyle(
        "CoverTitle",
        fontName="HeadingCN",
        fontSize=25,
        leading=36,
        textColor=NAVY,
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    subtitle = ParagraphStyle(
        "CoverSubtitle",
        fontName="SansCN",
        fontSize=13,
        leading=21,
        textColor=BLUE,
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    event = ParagraphStyle(
        "CoverEvent",
        fontName="SansCNBold",
        fontSize=10.2,
        leading=17,
        textColor=INK,
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    blank = ParagraphStyle(
        "CoverBlank",
        fontName="BodyCN",
        fontSize=10,
        leading=18,
        textColor=MUTED,
    )
    return [
        Spacer(1, 3.0 * cm),
        Spacer(1, 0.15 * cm),
        Paragraph(
            "体研智脑 Agent",
            title,
        ),
        Spacer(1, 0.25 * cm),
        Paragraph(
            "基于大语言模型与智能体协作的<br/>体育科研全流程智能助手",
            subtitle,
        ),
        Spacer(1, 1.35 * cm),
        Paragraph("2026 年第九届传智杯全国大学生数智创新与 AI 应用大赛", event),
        Spacer(1, 0.16 * cm),
        Paragraph("AI 大模型创新应用挑战赛（个人赛、大模型应用赛道）", event),
        Spacer(1, 0.55 * cm),
        Paragraph("技 术 文 档", ParagraphStyle(
            "CoverDocType",
            fontName="HeadingCN",
            fontSize=15,
            leading=22,
            textColor=CYAN,
        )),
        Spacer(1, 2.85 * cm),
        Paragraph("作者：刘轩", blank),
        Paragraph("单位：四川工商学院", blank),
        Paragraph("日期：2026 年 11 月", blank),
    ]


def image_flowable(path: Path, caption: str, styles) -> KeepTogether:
    image = Image(str(path))
    if path.name == "06-knowledge-graph-component.png":
        max_width, max_height = 13.8 * cm, 13.4 * cm
    elif "evidence" in path.parts:
        max_width, max_height = 15.0 * cm, 9.8 * cm
    else:
        max_width, max_height = 17.0 * cm, 12.2 * cm
    image._restrictSize(max_width, max_height)
    return KeepTogether(
        [
            image,
            Paragraph(inline_markup(caption), styles["caption"]),
        ]
    )


def table_flowable(rows: list[list[str]], styles) -> LongTable:
    header = [
        Paragraph(inline_markup(cell), styles["table_head"])
        for cell in rows[0]
    ]
    body = [
        [Paragraph(inline_markup(cell), styles["table_cell"]) for cell in row]
        for row in rows[1:]
    ]
    data = [header, *body]
    count = max(len(row) for row in rows)
    average_lengths = []
    for index in range(count):
        values = [
            len(row[index]) if index < len(row) else 0
            for row in rows
        ]
        average_lengths.append(max(7, sum(values) / max(1, len(values))))
    total = sum(average_lengths)
    widths = [
        max(1.15 * cm, CONTENT_WIDTH * value / total)
        for value in average_lengths
    ]
    scale = CONTENT_WIDTH / sum(widths)
    widths = [width * scale for width in widths]
    table = LongTable(
        data,
        colWidths=widths,
        repeatRows=1,
        hAlign="CENTER",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PAPER]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def parse_body(lines: list[str], styles) -> list:
    story: list = []
    index = 0
    paragraph_buffer: list[str] = []
    continue_next_heading = False

    def flush_paragraph() -> None:
        if not paragraph_buffer:
            return
        text = " ".join(item.strip() for item in paragraph_buffer)
        story.append(Paragraph(inline_markup(text), styles["body"]))
        paragraph_buffer.clear()

    while index < len(lines):
        raw = lines[index]
        line = raw.strip()
        if not line:
            flush_paragraph()
            index += 1
            continue
        if line == "<!-- PAGEBREAK -->":
            flush_paragraph()
            story.append(PageBreak())
            index += 1
            continue
        if line == "<!-- CONTINUE -->":
            flush_paragraph()
            continue_next_heading = True
            index += 1
            continue
        if line == "<!-- TOC -->":
            flush_paragraph()
            toc = TableOfContents()
            toc.levelStyles = [styles["toc1"], styles["toc2"]]
            toc.dotsMinLevel = 0
            story.extend(
                [
                    Paragraph("目录", styles["abstract_title"]),
                    Spacer(1, 0.2 * cm),
                    toc,
                ]
            )
            index += 1
            continue
        image_match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if image_match:
            flush_paragraph()
            caption, rel = image_match.groups()
            path = (ROOT / rel).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            story.append(image_flowable(path, caption, styles))
            index += 1
            continue
        if line.startswith("```"):
            flush_paragraph()
            code_lines = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            story.append(
                XPreformatted(
                    html.escape("\n".join(code_lines)),
                    styles["code"],
                )
            )
            index += 1
            continue
        if line.startswith("|") and line.endswith("|"):
            flush_paragraph()
            table_lines = []
            while (
                index < len(lines)
                and lines[index].strip().startswith("|")
                and lines[index].strip().endswith("|")
            ):
                table_lines.append(lines[index].strip())
                index += 1
            raw_rows = [
                [cell.strip() for cell in row.strip("|").split("|")]
                for row in table_lines
            ]
            if (
                len(raw_rows) > 1
                and all(re.fullmatch(r":?-{2,}:?", cell) for cell in raw_rows[1])
            ):
                raw_rows.pop(1)
            story.append(table_flowable(raw_rows, styles))
            story.append(Spacer(1, 0.18 * cm))
            continue
        if line.startswith("#"):
            flush_paragraph()
            level = len(line) - len(line.lstrip("#"))
            text = line[level:].strip()
            if level == 1:
                if (
                    story
                    and not isinstance(story[-1], PageBreak)
                    and not continue_next_heading
                ):
                    story.append(PageBreak())
                continue_next_heading = False
                story.append(Paragraph(inline_markup(text), styles["h1"]))
                story.append(
                    Paragraph(
                        "",
                        ParagraphStyle(
                            "Rule",
                            fontSize=1,
                            leading=1,
                            spaceAfter=5,
                        ),
                    )
                )
            elif level == 2:
                if text == "摘要":
                    story.append(Paragraph(text, styles["abstract_title"]))
                elif text == "目录":
                    pass
                else:
                    story.append(Paragraph(inline_markup(text), styles["h2"]))
            else:
                story.append(Paragraph(inline_markup(text), styles["h2"]))
            index += 1
            continue
        bullet_match = re.match(r"^[-*]\s+(.*)$", line)
        if bullet_match:
            flush_paragraph()
            story.append(
                Paragraph(
                    inline_markup(bullet_match.group(1)),
                    styles["bullet"],
                    bulletText="•",
                )
            )
            index += 1
            continue
        number_match = re.match(r"^(\d+)\.\s+(.*)$", line)
        if number_match:
            flush_paragraph()
            story.append(
                Paragraph(
                    inline_markup(number_match.group(2)),
                    styles["bullet"],
                    bulletText=f"{number_match.group(1)}.",
                )
            )
            index += 1
            continue
        if line.startswith("**关键词：**"):
            flush_paragraph()
            story.append(
                Paragraph(inline_markup(line), styles["abstract_keywords"])
            )
            index += 1
            continue
        paragraph_buffer.append(line)
        index += 1

    flush_paragraph()
    return story


def build_pdf() -> dict:
    register_fonts()
    styles = make_styles()
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    body_start = next(
        index
        for index, line in enumerate(lines)
        if line.strip() == "<!-- PAGEBREAK -->"
    )
    body_lines = lines[body_start + 1 :]
    story: list = cover_story(styles)
    story.extend([NextPageTemplate("content"), PageBreak()])
    story.extend(parse_body(body_lines, styles))

    doc = TechnicalDocument(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=LEFT,
        rightMargin=RIGHT,
        topMargin=TOP,
        bottomMargin=BOTTOM,
        title="体研智脑 Agent 技术文档",
        author="刘轩",
        subject="2026 年第九届传智杯 AI 大模型创新应用挑战赛技术文档",
        creator="ReportLab",
    )
    doc.multiBuild(story)
    return validate_pdf()


def validate_pdf() -> dict:
    document = pymupdf.open(OUTPUT)
    page_count = document.page_count
    empty_pages = []
    mojibake_markers = ("�", "????")
    mojibake_pages = []
    for page_index in range(page_count):
        text = document.load_page(page_index).get_text("text").strip()
        if not text:
            empty_pages.append(page_index + 1)
        if any(marker in text for marker in mojibake_markers):
            mojibake_pages.append(page_index + 1)
    text = "\n".join(
        document.load_page(index).get_text("text")
        for index in range(page_count)
    )
    report = {
        "pdf": str(OUTPUT),
        "page_count": page_count,
        "within_30_pages": page_count <= 30,
        "empty_pages": empty_pages,
        "mojibake_pages": mojibake_pages,
        "has_abstract": "摘要" in text,
        "has_chapter_1": "第一章 项目概述" in text,
        "has_chapter_8": "第八章 总结与展望" in text,
        "has_toc": "目录" in text,
        "has_references": "参考资料" in text,
        "image_count": len(document.get_page_images(0)),
    }
    total_images = 0
    for index in range(page_count):
        total_images += len(document.get_page_images(index))
    report["image_count"] = total_images
    report["passed"] = all(
        (
            page_count <= 30,
            not empty_pages,
            not mojibake_pages,
            report["has_abstract"],
            report["has_chapter_1"],
            report["has_chapter_8"],
            report["has_toc"],
            report["has_references"],
            total_images >= 12,
        )
    )
    document.close()
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    result = build_pdf()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
