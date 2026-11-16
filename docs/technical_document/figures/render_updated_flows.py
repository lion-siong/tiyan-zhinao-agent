"""Render the two corrected flow diagrams to PNG without external services."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
FONT = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"
NAVY = "#0F2F66"
BLUE = "#2459A6"
CYAN = "#16A6A1"
INK = "#172033"
LIGHT = "#F3F7FC"
LINE = "#8FA5C4"
WHITE = "#FFFFFF"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    size: int,
    *,
    color: str = INK,
    bold: bool = False,
) -> None:
    selected = font(size, bold)
    lines = text.split("\n")
    heights = [
        draw.textbbox((0, 0), line, font=selected)[3]
        - draw.textbbox((0, 0), line, font=selected)[1]
        for line in lines
    ]
    total_height = sum(heights) + max(0, len(lines) - 1) * 8
    y = (box[1] + box[3] - total_height) / 2
    for line, line_height in zip(lines, heights):
        width = draw.textbbox((0, 0), line, font=selected)[2]
        draw.text(
            ((box[0] + box[2] - width) / 2, y),
            line,
            font=selected,
            fill=color,
        )
        y += line_height + 8


def box(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    text: str,
    *,
    fill: str = WHITE,
    outline: str = BLUE,
    color: str = INK,
    size: int = 28,
    bold: bool = False,
    radius: int = 18,
) -> None:
    draw.rounded_rectangle(
        rect,
        radius=radius,
        fill=fill,
        outline=outline,
        width=3,
    )
    centered_text(
        draw,
        rect,
        text,
        size,
        color=color,
        bold=bold,
    )


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str = LINE,
    width: int = 4,
) -> None:
    draw.line((start, end), fill=color, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(dx) >= abs(dy):
        direction = 1 if dx >= 0 else -1
        points = [
            end,
            (end[0] - 16 * direction, end[1] - 9),
            (end[0] - 16 * direction, end[1] + 9),
        ]
    else:
        direction = 1 if dy >= 0 else -1
        points = [
            end,
            (end[0] - 9, end[1] - 16 * direction),
            (end[0] + 9, end[1] - 16 * direction),
        ]
    draw.polygon(points, fill=color)


def render_literature_flow() -> None:
    image = Image.new("RGB", (1800, 1180), LIGHT)
    draw = ImageDraw.Draw(image)
    draw.text(
        (70, 42),
        "文献检索来源与证据数据流",
        font=font(42, True),
        fill=NAVY,
    )
    draw.text(
        (72, 98),
        "两个页面的入口不同；底层来源能力不能互相冒充",
        font=font(24),
        fill="#5E6A7D",
    )

    box(draw, (70, 165, 390, 245), "文献自动采集页", fill="#EAF2FF", bold=True)
    box(draw, (70, 285, 390, 365), "LiteratureCollector")
    box(draw, (70, 405, 390, 485), "OpenAlex 优先")
    box(draw, (70, 525, 390, 605), "Crossref 失败回退")
    arrow(draw, (230, 245), (230, 285))
    arrow(draw, (230, 365), (230, 405))
    arrow(draw, (230, 485), (230, 525))

    box(draw, (520, 165, 880, 245), "智能文献研究页", fill="#EAF8F7", bold=True)
    box(
        draw,
        (520, 285, 880, 365),
        "Function Calling\nsemantic_scholar_search",
        fill="#EAF8F7",
        outline=CYAN,
    )
    box(
        draw,
        (520, 405, 880, 485),
        "MultiSourceRetriever",
        fill="#EAF8F7",
        outline=CYAN,
        bold=True,
    )
    arrow(draw, (700, 245), (700, 285), color=CYAN)
    arrow(draw, (700, 365), (700, 405), color=CYAN)

    source_names = (
        "Semantic Scholar",
        "arXiv",
        "OpenAlex",
        "Crossref",
        "PubMed",
    )
    for index, name in enumerate(source_names):
        x0 = 480 + index * 250
        rect = (x0, 535, x0 + 210, 610)
        box(
            draw,
            rect,
            name,
            fill="#F8FCFF",
            outline=CYAN,
            size=23,
        )
        if index:
            arrow(
                draw,
                (x0 - 40, 572),
                (x0 - 6, 572),
                color=LINE,
                width=3,
            )
    arrow(draw, (700, 485), (584, 535), color=CYAN)
    centered_text(
        draw,
        (480, 625, 1650, 675),
        "固定顺序聚合；单源失败隔离，汇总状态与来源",
        22,
        color="#5E6A7D",
    )

    box(draw, (220, 720, 600, 790), "候选题录\n专题相关度预检", fill="#EAF2FF")
    box(
        draw,
        (1050, 720, 1430, 790),
        "DOI / 规范化标题去重",
        fill="#EAF2FF",
    )
    arrow(draw, (700, 675), (410, 720), color=LINE)
    arrow(draw, (600, 755), (1050, 755), color=LINE)
    box(
        draw,
        (1050, 835, 1430, 905),
        "SQLite 题录\n仅开放许可才下载 PDF",
        fill="#EAF8F7",
        outline=CYAN,
    )
    arrow(draw, (1240, 790), (1240, 835), color=CYAN)
    box(
        draw,
        (520, 835, 900, 905),
        "PDF 校验 → 页级文本\n→ EvidenceRef",
        fill="#EAF8F7",
        outline=CYAN,
    )
    arrow(draw, (1050, 870), (900, 870), color=CYAN)
    box(
        draw,
        (70, 835, 430, 905),
        "SQLite / BM25 / Chroma\n均保留论文与页码",
        fill="#EAF2FF",
    )
    arrow(draw, (520, 870), (430, 870), color=LINE)

    centered_text(
        draw,
        (70, 1030, 1730, 1100),
        "说明：自动采集页为 OpenAlex→Crossref 两源链路；智能文献研究页注入五源 "
        "MultiSourceRetriever。二者仅共享后续题录、PDF 与证据处理。",
        21,
        color="#5E6A7D",
    )
    image.save(ROOT / "03-literature-data-flow.png", optimize=True)


def render_debate_flow() -> None:
    image = Image.new("RGB", (1800, 1100), LIGHT)
    draw = ImageDraw.Draw(image)
    draw.text(
        (70, 42),
        "同题结果比较与争议候选",
        font=font(42, True),
        fill=NAVY,
    )
    draw.text(
        (72, 98),
        "未达统计显著不进入反方；候选必须回到原文与人工复核",
        font=font(24),
        fill="#5E6A7D",
    )

    box(draw, (70, 170, 430, 250), "活动论文与页级证据")
    box(draw, (540, 170, 900, 250), "排除版权、作者、参考文献")
    box(draw, (1010, 170, 1370, 250), "按研究问题匹配论文")
    arrow(draw, (430, 210), (540, 210))
    arrow(draw, (900, 210), (1010, 210))

    labels = (
        ("正向结果", 70, "#EAF2FF", BLUE),
        ("明确负向结果", 500, "#FFF1F0", "#C2413A"),
        ("未达统计显著", 930, "#F3F4F6", "#64748B"),
        ("混合 / 待评估", 1360, "#FFF8E8", "#B7791F"),
    )
    for text, x, fill, outline in labels:
        box(
            draw,
            (x, 340, x + 330, 420),
            text,
            fill=fill,
            outline=outline,
            size=25,
            bold=True,
        )
        arrow(draw, (1190, 250), (x + 165, 340), color=LINE, width=3)

    box(
        draw,
        (415, 515, 1045, 605),
        "同题是否同时存在正向与明确负向？",
        fill="#F8FCFF",
        outline=NAVY,
        size=29,
        bold=True,
    )
    arrow(draw, (235, 420), (520, 515), color=BLUE)
    arrow(draw, (665, 420), (790, 515), color="#C2413A")
    arrow(draw, (1095, 420), (1095, 480), color="#64748B")
    arrow(draw, (1525, 420), (1260, 515), color="#B7791F")
    centered_text(
        draw,
        (1000, 470, 1370, 510),
        "不计反方，单独保留",
        20,
        color="#64748B",
    )

    box(
        draw,
        (1120, 680, 1680, 770),
        "待审核争议候选\n核对结局、剂量、对照、时点",
        fill="#FFF1F0",
        outline="#C2413A",
        size=24,
    )
    box(
        draw,
        (100, 680, 730, 770),
        "一致发现 / 无显著一致 / 证据不足",
        fill="#EAF8F7",
        outline=CYAN,
        size=24,
    )
    arrow(draw, (920, 605), (1400, 680), color="#C2413A")
    arrow(draw, (580, 605), (415, 680), color=CYAN)
    centered_text(draw, (820, 620, 980, 660), "是", 24, color="#C2413A", bold=True)
    centered_text(draw, (260, 620, 380, 660), "否", 24, color=CYAN, bold=True)

    box(
        draw,
        (330, 865, 1480, 945),
        "保存 DebateAnalysis → 知识图谱与学习测评只使用带来源证据",
        fill="#EAF2FF",
        outline=BLUE,
        size=25,
        bold=True,
    )
    arrow(draw, (1400, 770), (1480, 865))
    arrow(draw, (415, 770), (330, 865))
    centered_text(
        draw,
        (70, 995, 1730, 1050),
        "自动候选不是正式 GRADE 结论；没有真实相反结果时展示一致或证据不足。",
        22,
        color="#5E6A7D",
    )
    image.save(ROOT / "04-debate-analysis.png", optimize=True)


def main() -> int:
    render_literature_flow()
    render_debate_flow()
    print("Rendered 03-literature-data-flow.png and 04-debate-analysis.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
