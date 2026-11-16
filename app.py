"""体研智脑 Agent 的 Streamlit 中文界面。

启动方式：
    streamlit run app.py
"""

from __future__ import annotations

import importlib.metadata
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from agents.planner import ResearchPlannerAgent, SIMULATION_NOTICE
from config import settings
from researchmind.ui import (
    RESEARCHMIND_PAGE_ALIASES,
    RESEARCHMIND_PAGES,
    get_researchmind_workflow,
    render_researchmind_page,
)
from researchmind.literature_research_ui import render_literature_research
from researchmind.workbench import render_workbench
from services.llm_service import LLMResponse, LLMService
from services.rag_service import (
    EvidenceRAGService,
    HybridEvidenceIndex,
    LocalEvidenceIndex,
    MAX_RETRIEVAL_COUNT,
    normalize_retrieval_count,
)
from services.vector_service import (
    EvidenceVectorStore,
    VectorStoreError,
    VectorSyncResult,
    sync_bundled_vectors,
)
from tools.data_analysis import (
    DataAnalysisError,
    analyze_dataframe,
    create_numeric_histogram,
    load_dataframe,
)
from tools.document_chunker import EvidenceChunk, chunk_pdf_pages
from tools.knowledge_base import (
    BundledKnowledgeBase,
    KnowledgeBaseError,
    load_bundled_knowledge_base,
)
from tools.library_ingest import (
    DOCUMENT_PDF_DIR,
    IncomingPDF,
    IngestResult,
    ingest_pdf_files,
    rebuild_library,
    sync_pdf_directory,
)
from tools.literature_framework import (
    LiteratureFramework,
    load_literature_framework,
)
from tools.literature_search import build_search_strategy
from tools.pdf_reader import extract_pdf_text


PAGES = (
    "首页",
    "科研工作台",
    "智能科研问答",
    "科研选题",
    "文献阅读",
    "智能文献研究",
    "文献证据问答",
    "文献框架",
    "数据分析",
    "系统状态",
    *RESEARCHMIND_PAGES,
)

MODULE_CARDS = (
    (
        "科研工作台",
        "把研究问题、检索策略、真实证据、研究设计和数据分析连成可检查的工作流。",
        "工作流",
        "科研工作台",
        ":material/space_dashboard:",
    ),
    (
        "科研选题",
        "从研究方向生成结构化选题模板，明确背景、问题、对象与价值。",
        "规划",
        "科研选题",
        ":material/lightbulb:",
    ),
    (
        "文献阅读",
        "读取 PDF 文本并展示提取结果，为后续证据整理与 RAG 做准备。",
        "证据",
        "文献阅读",
        ":material/document_scanner:",
    ),
    (
        "智能文献研究",
        "用 Function Calling 检索公开论文，并在用户上传的 PDF 中做本地 RAG。",
        "研究",
        "智能文献研究",
        ":material/travel_explore:",
    ),
    (
        "文献证据问答",
        "结合关键词与本地语义检索真实 PDF 原文，保留文件名和页码。",
        "溯源",
        "文献证据问答",
        ":material/manage_search:",
    ),
    (
        "文献框架",
        "把新论文自动归入九个研究维度，保留原文页码并持续更新框架。",
        "框架",
        "文献框架",
        ":material/account_tree:",
    ),
    (
        "数据分析",
        "读取 CSV/XLSX，检查数据规模和缺失情况，计算基础描述统计。",
        "数据",
        "数据分析",
        ":material/table_chart:",
    ),
    (
        "智能科研问答",
        "提供科研方法问答；未配置密钥时使用明确标注的模拟回答。",
        "问答",
        "智能科研问答",
        ":material/forum:",
    ),
    (
        "系统状态",
        "检查模型配置、依赖与项目目录，区分真实接口和模拟能力。",
        "运行",
        "系统状态",
        ":material/monitor_heart:",
    ),
    (
        "ResearchMind 学术引擎",
        "进入领域管理，继续使用文献采集、论文阅读、知识图谱和学习测评。",
        "研究",
        "学术领域管理",
        ":material/hub:",
    ),
    (
        "个性化学习",
        "查看答题记录、五维掌握度、薄弱知识点和下一轮复习建议。",
        "学习",
        "学习成长报告",
        ":material/school:",
    ),
)

HOME_WORKFLOW_STEPS = (
    ("01", "研究规划", "拆解问题与检索策略"),
    ("02", "证据阅读", "追踪原文、文件与页码"),
    ("03", "数据分析", "读取真实数据与统计结果"),
    ("04", "成果整理", "导出可检查的工作底稿"),
)

# 两套主题共用同一组语义颜色，避免页面背景切换后卡片和文字仍然停留在旧主题。
THEME_PALETTES: dict[str, dict[str, str]] = {
    "light": {
        "background": "#F6F8FB",
        "surface": "#FFFFFF",
        "surface_soft": "#EEF3F8",
        "surface_strong": "#E6EDF5",
        "sidebar": "#FFFFFF",
        "text": "#182235",
        "muted": "#5B6B82",
        "faint": "#8494A9",
        "primary": "#2457D6",
        "primary_hover": "#1B47BB",
        "primary_soft": "#E8EEFF",
        "primary_text": "#FFFFFF",
        "border": "#D7E0EA",
        "border_strong": "#BAC8D8",
        "success": "#147D64",
        "warning": "#A55A0A",
        "danger": "#B83A45",
        "shadow": "rgba(38, 55, 82, 0.12)",
        "grid_line": "rgba(36, 87, 214, 0.06)",
    },
    "dark": {
        "background": "#0B111B",
        "surface": "#121B2A",
        "surface_soft": "#101827",
        "surface_strong": "#1A2639",
        "sidebar": "#0E1625",
        "text": "#E7EEF9",
        "muted": "#9AABC2",
        "faint": "#73839D",
        "primary": "#3B68EE",
        "primary_hover": "#315DD6",
        "primary_soft": "#18264A",
        "primary_text": "#FFFFFF",
        "border": "#26354D",
        "border_strong": "#3B4F6E",
        "success": "#55D6A7",
        "warning": "#FFB454",
        "danger": "#FF7A85",
        "shadow": "rgba(0, 0, 0, 0.28)",
        "grid_line": "rgba(255, 255, 255, 0.035)",
    },
}

# 页面助手的回答范围会随当前功能变化，避免所有页面都给出同一套泛化答案。
PAGE_ASSISTANT_CONTEXTS: dict[str, str] = {
    "首页": "介绍系统功能、科研工作流程，以及用户下一步应该进入哪个功能。",
    "科研工作台": "拆解研究问题、制定检索策略、选择研究设计、规划数据分析并检查证据链。",
    "智能科研问答": "解答研究设计、统计方法、论文写作和科研规范问题。",
    "科研选题": "从研究方向形成研究背景、研究问题、研究对象、变量、研究价值和可行性分析。",
    "文献阅读": "说明 PDF 上传、文本提取、结构化阅读和异常文件的处理方式。",
    "智能文献研究": "使用 Semantic Scholar 与 arXiv 检索真实论文元数据，整理综述和 GB/T 7714 引用。",
    "文献证据问答": "基于用户已导入的本地论文进行证据检索和原文问答，并提示页码与来源。",
    "文献框架": "解释文献主题、证据卡片、研究变量和知识框架之间的关系。",
    "数据分析": "指导 CSV、XLSX 数据检查、描述统计、缺失值处理和基础图表解读。",
    "系统状态": "帮助检查模型配置、文献库、向量索引和本地服务状态。",
    "学术领域管理": "管理研究领域、核心关键词和论文库的领域归属。",
    "文献自动采集": "说明公开学术来源检索、去重、元数据整理和导入流程。",
    "我的论文库": "查询已导入论文、原文证据、页码以及论文库维护方式。",
    "论文智能阅读": "解释论文目的、对象、方法、结果、结论和局限性的结构化提取。",
    "学术观点对比": "比较不同论文的观点、样本、研究设计、测量指标和统计方法。",
    "学术分歧分析": "识别学术共识、争议、证据强度和研究空白，并保留原文来源。",
    "知识图谱": "解释领域概念、理论关系、研究主题、研究方向、变量和证据之间的结构化联系。",
    "知识图谱总结": "解释研究主题、学术共识、争议、研究局限、空白、创新方向及原文证据追溯。",
    "AI 学习测评": "根据知识库生成题目、说明评分标准并组织当前测试。",
    "错题本": "分析错题原因、关联薄弱知识点并推荐复习材料。",
    "学习成长报告": "解读五维能力掌握度、学习趋势、薄弱点和下一轮复习计划。",
}

THEME_SWITCHER_HTML = """
<div class="sidebar-theme-switcher">
    <div class="sidebar-theme-head">
        <span>界面主题</span>
        <strong data-theme-current>跟随系统</strong>
    </div>
    <div class="sidebar-theme-options" role="group" aria-label="切换界面主题">
        <button type="button" data-theme-choice="Light">浅色</button>
        <button type="button" data-theme-choice="Dark">深色</button>
        <button type="button" data-theme-choice="System">自动</button>
    </div>
</div>
<script>
(() => {
    const roots = document.querySelectorAll(".sidebar-theme-switcher");
    const root = roots[roots.length - 1];
    if (!root) return;

    const labels = { Light: "浅色", Dark: "深色", System: "自动" };
    const buttons = [...root.querySelectorAll("[data-theme-choice]")];
    const currentLabel = root.querySelector("[data-theme-current]");
    let preference = "System";
    try {
        preference = JSON.parse(
            localStorage.getItem("stActiveTheme-/-v2") || '"System"'
        );
    } catch {
        preference = "System";
    }
    if (!labels[preference]) preference = "System";

    currentLabel.textContent = labels[preference];
    buttons.forEach((button) => {
        const active = button.dataset.themeChoice === preference;
        button.classList.toggle("is-active", active);
        button.setAttribute("aria-pressed", String(active));
        button.addEventListener("click", () => {
            const choice = button.dataset.themeChoice;
            if (!choice || choice === preference) return;
            localStorage.setItem(
                "stActiveTheme-/-v2",
                JSON.stringify(choice)
            );
            window.location.reload();
        });
    });
    document.documentElement.dataset.themeSwitchReady = "true";
})();
</script>
"""


@dataclass(frozen=True)
class HomeResearchSnapshot:
    """首页展示的真实文献与学习记录概览。"""

    domain_name: str
    papers: int
    evidence: int
    questions: int | None
    attempts: int | None
    source_titles: tuple[str, ...]


def inject_styles() -> None:
    """注入统一的学术科技风格样式。"""

    theme_type = getattr(st.context.theme, "type", "light")
    palette = THEME_PALETTES.get(theme_type, THEME_PALETTES["light"])
    styles = """
        <style>
        :root {
            --ink: #172832;
            --muted: #60717a;
            --navy: #082b33;
            --teal: #0f766e;
            --teal-soft: #e4f3f0;
            --paper: #f5f7f7;
            --line: #d9e2e2;
            --amber: #b66a13;
            --mint: #79cfc3;
            --console: #061f26;
        }
        .stApp {
            background: var(--paper);
            color: var(--ink);
            font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
        }
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input,
        [data-testid="stChatInput"] textarea {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink);
            background-color: #ffffff !important;
            caret-color: var(--teal);
        }
        [data-testid="stTextInput"] [data-baseweb="base-input"],
        [data-testid="stTextInput"] [data-baseweb="input"],
        [data-testid="stTextArea"] [data-baseweb="textarea"],
        [data-testid="stNumberInput"] [data-baseweb="input"],
        [data-testid="stChatInput"] {
            background-color: #ffffff !important;
            border-color: var(--line) !important;
        }
        [data-testid="stTextInput"] [data-baseweb="input"]:focus-within,
        [data-testid="stTextArea"] [data-baseweb="textarea"]:focus-within,
        [data-testid="stNumberInput"] [data-baseweb="input"]:focus-within,
        [data-testid="stChatInput"]:focus-within {
            border-color: var(--teal) !important;
            box-shadow: 0 0 0 1px var(--teal);
        }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink);
            background-color: #ffffff !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] span {
            color: var(--ink) !important;
        }
        input::placeholder,
        textarea::placeholder {
            color: var(--muted) !important;
            -webkit-text-fill-color: var(--muted);
            opacity: 1 !important;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2.75rem;
            padding-bottom: 1.75rem;
        }
        [data-testid="stSidebar"] {
            background: var(--navy);
            border-right: 1px solid #24464b;
        }
        [data-testid="stSidebar"] * {
            color: #edf6f4;
        }
        [data-testid="stSidebar"] .stRadio label {
            padding: 0.3rem 0.1rem;
        }
        [data-testid="stSidebar"] div[data-testid="stExpander"] {
            background: transparent !important;
            border-color: #315158 !important;
        }
        [data-testid="stSidebar"] div[data-testid="stExpander"] details {
            background: transparent !important;
            border-color: #315158 !important;
        }
        [data-testid="stSidebar"] div[data-testid="stExpander"] summary,
        [data-testid="stSidebar"] div[data-testid="stExpander"] summary * {
            background: transparent !important;
            color: #edf6f4 !important;
            fill: #edf6f4 !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: #315158;
        }
        .brand-mark {
            color: #f3f7f6;
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: 0;
            margin: 0.4rem 0 0.25rem;
        }
        .brand-sub {
            color: #a9c3c3;
            font-size: 0.78rem;
            line-height: 1.55;
            margin-bottom: 1.3rem;
        }
        .hero {
            position: relative;
            overflow: hidden;
            background: var(--navy);
            background-image:
                linear-gradient(
                    rgba(255, 255, 255, 0.035) 1px,
                    transparent 1px
                ),
                linear-gradient(
                    90deg,
                    rgba(255, 255, 255, 0.035) 1px,
                    transparent 1px
                );
            background-size: 48px 48px;
            border: 1px solid #1b454c;
            border-radius: 8px;
            color: #f7fbfa;
            padding: 1.45rem 1.55rem 1.2rem 1.8rem;
            margin-bottom: 0.9rem;
        }
        .hero::before {
            content: "";
            position: absolute;
            top: 0;
            bottom: 0;
            left: 0;
            width: 6px;
            background: linear-gradient(
                180deg,
                var(--amber) 0 28%,
                #315b62 28% 100%
            );
        }
        .hero::after {
            content: "";
            position: absolute;
            right: 0;
            bottom: 0;
            left: 0;
            height: 3px;
            background: linear-gradient(
                90deg,
                var(--amber) 0 17%,
                #2b5960 17% 100%
            );
        }
        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.55fr) minmax(255px, 0.72fr);
            gap: 1.25rem;
            align-items: stretch;
        }
        .hero-copy {
            min-width: 0;
        }
        .hero-heading-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
        }
        .hero-kicker {
            color: #72c8bc;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .hero-status {
            display: inline-flex;
            align-items: center;
            gap: 0.42rem;
            color: #d8ebe8;
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid rgba(131, 215, 204, 0.28);
            border-radius: 999px;
            padding: 0.3rem 0.7rem;
            font-size: 0.75rem;
        }
        .hero-status-dot {
            width: 7px;
            height: 7px;
            flex: 0 0 7px;
            border-radius: 50%;
            background: #72c8bc;
            box-shadow: 0 0 0 3px rgba(114, 200, 188, 0.13);
        }
        .hero h1 {
            color: #ffffff;
            font-size: clamp(2rem, 4.2vw, 3rem);
            line-height: 1.06;
            letter-spacing: 0;
            margin: 0.45rem 0 0.55rem;
        }
        .hero p {
            color: #c9dad8;
            font-size: 0.95rem;
            line-height: 1.7;
            max-width: 690px;
            margin: 0;
        }
        .hero-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.7rem;
        }
        .hero-tags span {
            color: #bfd8d4;
            border: 1px solid #35616a;
            border-radius: 999px;
            padding: 0.24rem 0.62rem;
            font-size: 0.74rem;
        }
        .hero-console {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-width: 0;
            padding: 0.85rem 0.95rem;
            background: rgba(3, 24, 30, 0.76);
            border: 1px solid #31545b;
            border-radius: 8px;
            box-shadow: inset 0 3px 0 var(--amber);
        }
        .console-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.7rem;
            color: #b8d7d3;
            font-size: 0.66rem;
            font-weight: 700;
            letter-spacing: 0.08em;
        }
        .console-live {
            color: var(--mint);
            white-space: nowrap;
        }
        .console-domain {
            min-width: 0;
            margin: 0.7rem 0 0.55rem;
            color: #f0f7f6;
            font-size: 0.86rem;
            font-weight: 700;
            line-height: 1.45;
            overflow-wrap: anywhere;
        }
        .console-domain small {
            display: block;
            color: #8fb1af;
            font-size: 0.66rem;
            font-weight: 400;
            margin-bottom: 0.15rem;
        }
        .console-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            border-top: 1px solid #284c53;
            border-left: 1px solid #284c53;
        }
        .console-cell {
            min-width: 0;
            padding: 0.5rem 0.55rem;
            border-right: 1px solid #284c53;
            border-bottom: 1px solid #284c53;
        }
        .console-cell span {
            display: block;
            color: #87aaa9;
            font-size: 0.6rem;
            font-weight: 700;
            letter-spacing: 0.08em;
        }
        .console-cell strong {
            display: block;
            color: #ffffff;
            font-size: 1.15rem;
            line-height: 1.35;
            margin-top: 0.08rem;
        }
        .console-cell small {
            color: #a8c4c2;
            font-size: 0.64rem;
        }
        .console-foot {
            color: #789b9c;
            font-size: 0.64rem;
            line-height: 1.45;
            margin-top: 0.55rem;
        }
        .hero-workflow {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0;
            border-top: 1px solid #31545b;
            margin-top: 1.05rem;
            padding-top: 0.75rem;
        }
        .hero-workflow-item {
            min-width: 0;
            padding: 0 1rem;
            border-left: 1px solid #31545b;
        }
        .hero-workflow-item:first-child {
            padding-left: 0;
            border-left: 0;
        }
        .hero-workflow-index {
            color: #d89b48;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.08em;
        }
        .hero-workflow-title {
            display: block;
            color: #f3f8f7;
            font-size: 0.88rem;
            font-weight: 700;
            margin: 0.2rem 0 0.16rem;
        }
        .hero-workflow-text {
            display: block;
            color: #9fc0be;
            font-size: 0.73rem;
            line-height: 1.5;
        }
        .task-intro {
            margin-bottom: 0.55rem;
        }
        .task-title-row {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
        }
        .task-title-row h2 {
            margin: 0;
            color: var(--ink);
            font-size: 1.72rem;
            line-height: 1.3;
        }
        .task-title-note {
            color: var(--muted);
            font-size: 0.78rem;
        }
        .task-intro p {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.65;
            margin: 0.18rem 0 0.45rem;
        }
        .task-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
        }
        .task-tags span {
            color: #315159;
            background: #eaf1f0;
            border: 1px solid #d5e1e0;
            border-radius: 4px;
            padding: 0.2rem 0.48rem;
            font-size: 0.72rem;
        }
        [class*="st-key-home_task_station"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.75rem 0.9rem 0.35rem;
            box-shadow: 0 5px 18px rgba(15, 72, 73, 0.05);
        }
        [class*="st-key-home_task_station"] div[data-testid="stForm"] {
            background: transparent;
            border: 0;
            padding: 0;
        }
        .task-station-label {
            color: var(--teal);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.07em;
            margin-bottom: 0.28rem;
        }
        .st-key-home_task_input input {
            min-height: 3rem;
            font-size: 0.94rem;
        }
        [class*="st-key-home_task_submit"] button {
            min-height: 3rem;
            width: 100%;
            white-space: nowrap;
        }
        .module-heading {
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
            margin-bottom: 0.5rem;
        }
        .module-heading h2 {
            margin: 0.08rem 0 0;
            color: var(--ink);
            font-size: 1.72rem;
            line-height: 1.3;
        }
        .module-heading p {
            color: var(--muted);
            font-size: 0.8rem;
            margin: 0;
        }
        .section-label {
            color: var(--teal);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            margin-bottom: 0.2rem;
        }
        .module-card {
            min-height: 138px;
            border-top: 3px solid var(--teal);
            background: #ffffff;
            padding: 1rem 1.1rem 1.05rem;
        }
        .module-title {
            color: var(--ink);
            font-size: 1.02rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }
        .module-text {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.65;
        }
        .module-tag {
            display: inline-flex;
            color: #0a625b;
            background: #cfe8e2;
            border-radius: 4px;
            padding: 0.16rem 0.46rem;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            margin: 0 0 0.45rem;
        }
        .module-overline {
            display: flex;
            align-items: center;
            gap: 0.28rem;
            margin-bottom: 0.32rem;
        }
        .module-overline .module-tag {
            margin: 0;
        }
        .module-index {
            color: #8aa4a4;
            font-family: "Cascadia Code", "Consolas", monospace;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            margin-right: 0.3rem;
        }
        [class*="st-key-module_panel_"] {
            background:
                linear-gradient(135deg, rgba(15, 118, 110, 0.035), transparent 58%),
                #eaf3f1;
            border-color: #c8ddd8 !important;
            border-radius: 8px;
            box-shadow: inset 0 3px 0 var(--teal);
            transition:
                border-color 120ms ease,
                box-shadow 120ms ease,
                transform 120ms ease;
        }
        [class*="st-key-module_panel_"]:hover {
            border-color: #8fbfb7 !important;
            box-shadow:
                inset 0 3px 0 var(--teal),
                0 6px 18px rgba(13, 72, 74, 0.08);
            transform: translateY(-1px);
        }
        [class*="st-key-module_card_"] button {
            justify-content: center;
            min-height: 2.15rem;
            padding: 0.25rem 0.45rem;
            color: #ffffff;
            background: var(--teal);
            border-color: var(--teal);
            box-shadow: 0 1px 2px rgba(13, 48, 56, 0.12);
        }
        [class*="st-key-module_card_"] button:hover {
            color: #ffffff;
            background: #0b5f58;
            border-color: #0b5f58;
        }
        [class*="st-key-module_card_"] button p {
            color: #ffffff;
            font-size: 1rem;
            font-weight: 700;
        }
        .status-line {
            display: inline-block;
            color: #315159;
            background: #e9efef;
            border: 1px solid #d2dede;
            border-radius: 999px;
            padding: 0.22rem 0.65rem;
            font-size: 0.75rem;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
        }
        .simulation-note {
            border-left: 4px solid var(--amber);
            background: #fff8ed;
            color: #5f421b;
            padding: 0.75rem 0.9rem;
            font-size: 0.86rem;
            margin: 0.7rem 0 1rem;
        }
        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }
        div[data-testid="stForm"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.8rem 0.95rem 0.25rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
        }
        div[data-testid="stMetricLabel"] {
            color: var(--muted);
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(
            .module-tag
        ) {
            background: #ffffff;
            border-color: var(--line) !important;
            border-radius: 8px;
        }
        .stButton > button {
            border-radius: 6px;
            font-weight: 700;
        }
        div[data-testid="stExpander"] {
            border-radius: 7px;
            background: #ffffff;
        }
        .footer-note {
            color: #7c8c91;
            border-top: 1px solid var(--line);
            margin-top: 1.25rem;
            padding-top: 0.8rem;
            font-size: 0.76rem;
        }
        @media (max-width: 1050px) {
            .hero-grid {
                grid-template-columns: minmax(0, 1fr);
            }
        }
        @media (max-width: 700px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
                padding-top: 2.15rem;
            }
            .hero {
                padding: 1.2rem 1rem 1.1rem 1.2rem;
            }
            .hero-grid {
                grid-template-columns: minmax(0, 1fr);
                gap: 0.9rem;
            }
            .hero h1 {
                font-size: 2rem;
            }
            .hero p {
                font-size: 0.9rem;
            }
            .hero-heading-row {
                align-items: flex-start;
            }
            .hero-status {
                margin-top: -0.15rem;
            }
            .hero-workflow {
                grid-template-columns: repeat(2, minmax(0, 1fr));
                row-gap: 0.75rem;
            }
            .hero-workflow-item {
                padding: 0 0.85rem;
            }
            .hero-workflow-item:nth-child(odd) {
                padding-left: 0;
                border-left: 0;
            }
            .hero-console {
                margin-top: 0.1rem;
            }
            .task-title-row h2,
            .module-heading h2 {
                font-size: 1.48rem;
            }
            [class*="st-key-home_task_station"] {
                padding: 0.7rem 0.75rem 0.3rem;
            }
            .module-card {
                min-height: auto;
            }
        }

        /* Research workspace theme */
        :root {
            --ink: #edf3ff;
            --muted: #8795ae;
            --navy: #071024;
            --teal: #5b82ff;
            --teal-soft: #142244;
            --paper: #050a17;
            --line: #1b2a47;
            --amber: #ffb454;
            --mint: #80a2ff;
            --console: #081226;
        }
        html,
        body,
        .stApp,
        [data-testid="stAppViewContainer"] {
            background: #050a17 !important;
            color: var(--ink);
        }
        [data-testid="stHeader"] {
            background: rgba(5, 10, 23, 0.94);
            border-bottom: 1px solid #13213a;
        }
        [data-testid="stAppDeployButton"] {
            display: none !important;
        }
        [data-testid="stDecoration"],
        #MainMenu {
            display: none;
        }
        .block-container {
            max-width: 1360px;
            padding-top: 2.2rem;
            padding-bottom: 2rem;
        }
        h1,
        h2,
        h3,
        h4,
        label,
        p {
            color: var(--ink);
        }
        small,
        .stCaptionContainer,
        [data-testid="stCaptionContainer"] {
            color: var(--muted);
        }
        [data-testid="stSidebar"] {
            min-width: 264px;
            max-width: 264px;
            background: #071024;
            border-right: 1px solid #16243e;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding-top: 0.95rem;
        }
        [data-testid="stSidebar"] hr {
            border-color: #1a2945;
            margin: 0.7rem 0;
        }
        .brand-lockup {
            display: flex;
            align-items: center;
            gap: 0.72rem;
            padding: 0.15rem 0 1.15rem;
            border-bottom: 1px solid #15243f;
            margin-bottom: 1rem;
        }
        .brand-symbol {
            display: grid;
            place-items: center;
            width: 38px;
            height: 38px;
            flex: 0 0 38px;
            border-radius: 10px;
            background: #3f6fff;
            color: #ffffff;
            font-size: 1rem;
            font-weight: 800;
            box-shadow: 0 0 0 1px rgba(132, 158, 255, 0.3);
        }
        .brand-mark {
            color: #f4f7ff;
            font-size: 1.1rem;
            font-weight: 750;
            line-height: 1.2;
            margin: 0;
        }
        .brand-sub {
            color: #8190ad;
            font-size: 0.68rem;
            line-height: 1.4;
            margin: 0.16rem 0 0;
        }
        [data-testid="stSidebar"] [data-testid="stSelectbox"] label {
            color: #8494b0 !important;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            min-height: 2.85rem;
            color: #e7edfb !important;
            -webkit-text-fill-color: #e7edfb;
            background: #0b1730 !important;
            border-color: #1d3156 !important;
            box-shadow: inset 3px 0 0 #3f6fff;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] span,
        [data-testid="stSidebar"] [data-baseweb="select"] svg {
            color: #e7edfb !important;
            fill: #8ea1c4 !important;
        }
        [data-testid="stSidebar"] .st-key-sidebar_navigation [data-baseweb="select"] > div,
        [data-testid="stSidebar"] .st-key-sidebar_navigation [data-baseweb="select"] div[role="button"] {
            color: #e7edfb !important;
            -webkit-text-fill-color: #e7edfb !important;
            background: #0b1730 !important;
            border-color: #1d3156 !important;
        }
        [data-testid="stSidebar"] .st-key-sidebar_navigation input {
            color: #e7edfb !important;
            -webkit-text-fill-color: #e7edfb !important;
            background: transparent !important;
        }
        [data-testid="stSidebar"] .st-key-sidebar_navigation [data-baseweb="select"] span,
        [data-testid="stSidebar"] .st-key-sidebar_navigation [data-baseweb="select"] svg {
            color: #e7edfb !important;
            fill: #8ea1c4 !important;
        }
        [data-baseweb="popover"] [role="listbox"],
        [data-baseweb="popover"] [role="option"] {
            color: #dfe7f7 !important;
            background: #0b1730 !important;
        }
        [data-baseweb="popover"] [role="option"]:hover,
        [data-baseweb="popover"] [aria-selected="true"] {
            color: #ffffff !important;
            background: #172b50 !important;
        }
        .sidebar-engine {
            padding: 0.85rem 0.9rem;
            background: #09152b;
            border: 1px solid #172947;
            border-radius: 10px;
        }
        .sidebar-engine-head {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            color: #dce6fa;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.72rem;
        }
        .sidebar-engine-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #ffb454;
            box-shadow: 0 0 0 3px rgba(255, 180, 84, 0.1);
        }
        .sidebar-engine-dot.online {
            background: #55d6a7;
            box-shadow: 0 0 0 3px rgba(85, 214, 167, 0.1);
        }
        .sidebar-engine-row {
            display: flex;
            justify-content: space-between;
            gap: 0.6rem;
            padding: 0.35rem 0;
            border-top: 1px solid #14233d;
            color: #7f8eaa;
            font-size: 0.7rem;
        }
        .sidebar-engine-row strong {
            color: #b9c8e3;
            font-family: "Cascadia Code", "Consolas", monospace;
            font-size: 0.64rem;
            font-weight: 600;
        }
        .rm-page-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1.5rem;
            min-height: 96px;
            padding: 0.1rem 0 1.2rem;
            border-bottom: 1px solid #16233d;
            margin-bottom: 0.9rem;
        }
        .rm-page-copy {
            max-width: 820px;
        }
        .rm-kicker {
            color: #6f8fff;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.12em;
        }
        .rm-page-head h1 {
            color: #f4f7ff;
            font-size: 1.9rem;
            line-height: 1.2;
            margin: 0.36rem 0 0.4rem;
        }
        .rm-page-head p {
            color: #8e9bb4;
            font-size: 0.86rem;
            line-height: 1.7;
            margin: 0;
        }
        .rm-head-status {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            color: #cbd6ea;
            background: #0a1428;
            border: 1px solid #1c2e4e;
            border-radius: 999px;
            padding: 0.52rem 0.8rem;
            font-size: 0.7rem;
            white-space: nowrap;
        }
        .rm-status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #55d6a7;
            box-shadow: 0 0 0 3px rgba(85, 214, 167, 0.1);
        }
        [class*="st-key-home_quick_action_"] button {
            min-height: 2.65rem;
            justify-content: flex-start;
            color: #b9c6df !important;
            background: #0a1428 !important;
            border-color: #1a2a47 !important;
            border-radius: 999px !important;
            box-shadow: none !important;
        }
        [class*="st-key-home_quick_action_"] button:hover {
            color: #ffffff !important;
            background: #101f3b !important;
            border-color: #335184 !important;
        }
        [class*="st-key-home_quick_action_"] button p {
            color: inherit !important;
            font-size: 0.78rem;
            font-weight: 650;
        }
        .rm-command-heading {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 1rem;
            margin: 0.15rem 0 0.55rem;
        }
        .rm-command-heading span {
            color: #7894ff;
            font-size: 0.68rem;
            font-weight: 750;
            letter-spacing: 0.1em;
        }
        .rm-command-heading small {
            color: #7787a3;
            font-size: 0.68rem;
        }
        [class*="st-key-home_task_station"] {
            background: #081226 !important;
            border: 1px solid #1d2d4c !important;
            border-radius: 14px !important;
            padding: 0.65rem 0.8rem 0.55rem !important;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.025),
                0 12px 30px rgba(0, 0, 0, 0.18) !important;
        }
        [class*="st-key-home_task_station"] div[data-testid="stForm"] {
            background: transparent !important;
            border: 0 !important;
            padding: 0 !important;
        }
        [data-testid="stTextArea"] textarea,
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stChatInput"] textarea {
            color: #edf3ff !important;
            -webkit-text-fill-color: #edf3ff;
            background: #050c1c !important;
            border-color: #1a2947 !important;
            caret-color: #6e91ff;
        }
        [data-testid="stTextArea"] [data-baseweb="textarea"],
        [data-testid="stTextInput"] [data-baseweb="input"],
        [data-testid="stNumberInput"] [data-baseweb="input"],
        [data-testid="stChatInput"] {
            background: #050c1c !important;
            border-color: #1a2947 !important;
        }
        [data-testid="stTextArea"] textarea::placeholder,
        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stChatInput"] textarea::placeholder {
            color: #64738e !important;
            -webkit-text-fill-color: #64738e;
        }
        .st-key-home_task_input textarea {
            min-height: 106px !important;
            border: 0 !important;
            box-shadow: none !important;
            font-size: 0.92rem;
            line-height: 1.65;
            resize: none;
        }
        .stButton > button,
        .stDownloadButton > button,
        [data-testid="stFormSubmitButton"] button {
            color: #dbe5f8;
            background: #0c1830;
            border: 1px solid #223658;
            border-radius: 8px;
            box-shadow: none;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            color: #ffffff;
            background: #142442;
            border-color: #3a5687;
        }
        .stButton > button[kind="primary"],
        [data-testid="stFormSubmitButton"] button[kind="primary"] {
            color: #ffffff;
            background: #3f6fff;
            border-color: #5b82ff;
            box-shadow: 0 8px 20px rgba(63, 111, 255, 0.24);
        }
        .stButton > button[kind="primary"]:hover,
        [data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
            color: #ffffff;
            background: #527cff;
            border-color: #7897ff;
        }
        .stButton > button p,
        .stDownloadButton > button p,
        [data-testid="stFormSubmitButton"] button p {
            color: inherit;
            font-weight: 650;
        }
        .rm-scope-card {
            min-height: 219px;
            height: 100%;
            padding: 0.85rem 0.9rem 0.75rem;
            background: #081226;
            border: 1px solid #1d2d4c;
            border-radius: 12px;
            box-shadow: inset 3px 0 0 #3f6fff;
        }
        .rm-panel-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.7rem;
            color: #8190aa;
            font-size: 0.64rem;
            font-weight: 750;
            letter-spacing: 0.08em;
        }
        .rm-panel-head strong {
            display: grid;
            place-items: center;
            min-width: 1.45rem;
            height: 1.45rem;
            color: #9cb2ff;
            background: #111f3c;
            border: 1px solid #253b63;
            border-radius: 6px;
            font-size: 0.7rem;
        }
        .rm-scope-domain {
            margin: 0.72rem 0 0.65rem;
            color: #d6e1f5;
            font-size: 0.82rem;
            font-weight: 650;
            line-height: 1.45;
            overflow-wrap: anywhere;
        }
        .rm-scope-list {
            display: grid;
            gap: 0.42rem;
        }
        .rm-scope-item {
            display: grid;
            grid-template-columns: 7px minmax(0, 1fr);
            align-items: start;
            gap: 0.48rem;
            padding: 0.44rem 0.5rem;
            color: #aebbd3;
            background: #070f21;
            border: 1px solid #172744;
            border-radius: 7px;
            font-size: 0.69rem;
            line-height: 1.4;
        }
        .rm-scope-item > span:last-child {
            display: -webkit-box;
            overflow: hidden;
            -webkit-box-orient: vertical;
            -webkit-line-clamp: 2;
        }
        .rm-source-mark {
            width: 6px;
            height: 6px;
            margin-top: 0.27rem;
            border-radius: 2px;
            background: #5f83ff;
        }
        .rm-scope-empty {
            padding: 0.65rem;
            color: #71809a;
            background: #070f21;
            border: 1px dashed #1b2c49;
            border-radius: 7px;
            font-size: 0.7rem;
            line-height: 1.5;
        }
        .rm-scope-foot {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            margin-top: 0.62rem;
            color: #647590;
            font-size: 0.62rem;
        }
        .rm-research-grid {
            display: grid;
            grid-template-columns: minmax(0, 2.25fr) minmax(275px, 0.88fr);
            gap: 1rem;
            align-items: stretch;
        }
        .rm-synthesis-panel,
        .rm-citations-panel {
            min-width: 0;
            background: #081226;
            border: 1px solid #1a2b48;
            border-radius: 12px;
            overflow: hidden;
        }
        .rm-synthesis-head {
            display: flex;
            align-items: center;
            gap: 0.72rem;
            min-height: 72px;
            padding: 0.85rem 1rem;
            border-bottom: 1px solid #182945;
        }
        .rm-synthesis-icon {
            display: grid;
            place-items: center;
            width: 34px;
            height: 34px;
            flex: 0 0 34px;
            color: #9bb2ff;
            background: #101f3c;
            border: 1px solid #253a61;
            border-radius: 8px;
            font-family: "Cascadia Code", "Consolas", monospace;
            font-size: 0.7rem;
            font-weight: 800;
        }
        .rm-panel-title {
            color: #e5ebf8;
            font-size: 0.72rem;
            font-weight: 750;
            letter-spacing: 0.08em;
        }
        .rm-meta-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.42rem;
            margin-top: 0.38rem;
        }
        .rm-meta-row span {
            color: #8190aa;
            background: #0b1730;
            border: 1px solid #1a2c4b;
            border-radius: 4px;
            padding: 0.12rem 0.38rem;
            font-size: 0.58rem;
        }
        .rm-meta-row .rm-live-chip {
            color: #7d9cff;
            background: rgba(63, 111, 255, 0.09);
            border-color: rgba(91, 130, 255, 0.3);
        }
        .rm-synthesis-body {
            padding: 1.1rem 1rem 1rem;
        }
        .rm-synthesis-label {
            color: #6f8fff;
            font-size: 0.62rem;
            font-weight: 750;
            letter-spacing: 0.1em;
        }
        .rm-synthesis-body h2 {
            color: #f2f5fc;
            font-size: 1.25rem;
            line-height: 1.35;
            margin: 0.36rem 0 0.38rem;
        }
        .rm-synthesis-body > p {
            max-width: 720px;
            color: #8b99b2;
            font-size: 0.78rem;
            line-height: 1.7;
            margin: 0;
        }
        .rm-step-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.55rem;
            margin-top: 0.9rem;
        }
        .rm-step {
            display: grid;
            grid-template-columns: auto minmax(0, 1fr);
            gap: 0.55rem;
            padding: 0.66rem 0.7rem;
            background: #070f21;
            border: 1px solid #172744;
            border-radius: 8px;
        }
        .rm-step-index {
            color: #6f8fff;
            font-family: "Cascadia Code", "Consolas", monospace;
            font-size: 0.66rem;
            font-weight: 700;
        }
        .rm-step strong {
            display: block;
            color: #d9e2f4;
            font-size: 0.76rem;
            margin-bottom: 0.18rem;
        }
        .rm-step p {
            color: #73829c;
            font-size: 0.68rem;
            line-height: 1.5;
            margin: 0;
        }
        .rm-citations-panel {
            padding: 0.9rem;
        }
        .rm-citation-list {
            display: grid;
            gap: 0.65rem;
            margin-top: 0.75rem;
        }
        .rm-citation-card {
            padding: 0.7rem 0.75rem;
            background: #070f21;
            border: 1px solid #182945;
            border-radius: 8px;
        }
        .rm-source-top {
            display: flex;
            justify-content: space-between;
            gap: 0.65rem;
            color: #6f8fff;
            font-family: "Cascadia Code", "Consolas", monospace;
            font-size: 0.58rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }
        .rm-source-top span:last-child {
            color: #60708c;
        }
        .rm-citation-card strong {
            display: -webkit-box;
            overflow: hidden;
            color: #cfd9ec;
            font-size: 0.7rem;
            line-height: 1.48;
            -webkit-box-orient: vertical;
            -webkit-line-clamp: 2;
        }
        .rm-citation-card p {
            color: #6e7d98;
            font-size: 0.63rem;
            line-height: 1.55;
            margin: 0.3rem 0 0;
        }
        .rm-citation-note {
            color: #61708a;
            font-size: 0.6rem;
            line-height: 1.5;
            margin-top: 0.72rem;
        }
        .module-heading {
            margin-top: 0.1rem;
        }
        .module-heading h2,
        .task-title-row h2 {
            color: #f0f4fd;
        }
        .module-heading p,
        .task-title-note,
        .task-intro p {
            color: #7e8da7;
        }
        .section-label {
            color: #7894ff;
        }
        .module-tag {
            color: #a8bbff;
            background: rgba(63, 111, 255, 0.11);
            border: 1px solid rgba(104, 139, 255, 0.2);
        }
        .module-index {
            color: #53647f;
        }
        .module-text {
            color: #8796b0;
        }
        [class*="st-key-module_panel_"] {
            background: #081226 !important;
            border-color: #192a46 !important;
            border-radius: 10px !important;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.025) !important;
        }
        [class*="st-key-module_panel_"]:hover {
            border-color: #315181 !important;
            box-shadow: 0 10px 26px rgba(0, 0, 0, 0.18) !important;
            transform: translateY(-1px);
        }
        [class*="st-key-module_card_"] button {
            justify-content: flex-start;
            color: #e1e9f9 !important;
            background: #0c1932 !important;
            border-color: #203659 !important;
        }
        [class*="st-key-module_card_"] button:hover {
            color: #ffffff !important;
            background: #122442 !important;
            border-color: #3b5c90 !important;
        }
        [class*="st-key-module_card_"] button p {
            color: inherit !important;
            font-size: 0.9rem;
        }
        div[data-testid="stForm"],
        div[data-testid="stMetric"],
        div[data-testid="stExpander"] {
            background: #081226;
            border-color: #1a2b48;
            color: #edf3ff;
        }
        div[data-testid="stExpander"] details,
        div[data-testid="stExpander"] summary {
            background: transparent;
            color: #dfe7f6;
        }
        [data-testid="stAlert"] {
            border-radius: 9px;
        }
        .status-line {
            color: #aab9d2;
            background: #0a152b;
            border-color: #1b2d4c;
        }
        .simulation-note {
            color: #d8c19c;
            background: #171322;
            border-left-color: #ffb454;
        }
        .footer-note {
            color: #5f708b;
            border-top-color: #15243e;
        }
        @media (max-width: 1050px) {
            .rm-research-grid {
                grid-template-columns: minmax(0, 1fr);
            }
        }
        @media (max-width: 700px) {
            html,
            body,
            .stApp,
            [data-testid="stAppViewContainer"] {
                width: 100% !important;
                max-width: 100% !important;
                overflow-x: hidden !important;
            }
            .block-container {
                width: 100% !important;
                max-width: 100% !important;
                padding-top: 3.1rem !important;
                padding-right: 1rem !important;
                padding-left: 1rem !important;
                box-sizing: border-box !important;
            }
            [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
                gap: 0.65rem !important;
            }
            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                width: 100% !important;
                min-width: 0 !important;
                flex: 1 1 100% !important;
            }
            [class*="st-key-home_task_station"],
            .rm-scope-card,
            .rm-research-grid,
            .rm-synthesis-panel,
            .rm-citations-panel,
            div[data-testid="stForm"],
            textarea,
            input {
                width: 100% !important;
                max-width: 100% !important;
                box-sizing: border-box !important;
            }
            .rm-page-copy,
            .rm-page-head p,
            .rm-scope-item,
            .rm-citation-card,
            .rm-step {
                min-width: 0 !important;
                overflow-wrap: anywhere;
            }
            .rm-page-head {
                display: block;
            }
            .rm-head-status {
                margin-top: 0.75rem;
            }
            .rm-command-heading {
                display: block;
            }
            .rm-command-heading small {
                display: block;
                margin-top: 0.2rem;
            }
            .rm-step-grid {
                grid-template-columns: minmax(0, 1fr);
            }
            .rm-page-head h1 {
                font-size: 1.65rem;
            }
        }

        /* Dual light/dark palette bridge */
        :root {
            --app-bg: __APP_BG__;
            --app-surface: __APP_SURFACE__;
            --app-surface-soft: __APP_SURFACE_SOFT__;
            --app-surface-strong: __APP_SURFACE_STRONG__;
            --app-sidebar: __APP_SIDEBAR__;
            --app-text: __APP_TEXT__;
            --app-muted: __APP_MUTED__;
            --app-faint: __APP_FAINT__;
            --app-primary: __APP_PRIMARY__;
            --app-primary-hover: __APP_PRIMARY_HOVER__;
            --app-primary-soft: __APP_PRIMARY_SOFT__;
            --app-primary-text: __APP_PRIMARY_TEXT__;
            --app-border: __APP_BORDER__;
            --app-border-strong: __APP_BORDER_STRONG__;
            --app-success: __APP_SUCCESS__;
            --app-warning: __APP_WARNING__;
            --app-danger: __APP_DANGER__;
            --app-shadow: __APP_SHADOW__;
            --app-grid-line: __APP_GRID_LINE__;
            --ink: var(--app-text);
            --muted: var(--app-muted);
            --navy: var(--app-sidebar);
            --teal: var(--app-primary);
            --teal-soft: var(--app-primary-soft);
            --paper: var(--app-bg);
            --line: var(--app-border);
            --amber: var(--app-warning);
            --mint: var(--app-success);
            --console: var(--app-surface-soft);
        }
        html,
        body,
        .stApp,
        [data-testid="stAppViewContainer"] {
            background: var(--app-bg) !important;
            color: var(--app-text) !important;
        }
        [data-testid="stHeader"] {
            background: color-mix(
                in srgb,
                var(--app-bg) 92%,
                transparent
            ) !important;
            border-bottom: 1px solid var(--app-border) !important;
            backdrop-filter: blur(14px);
        }
        h1,
        h2,
        h3,
        h4,
        label,
        p,
        .stMarkdown {
            color: var(--app-text);
        }
        small,
        .stCaptionContainer,
        [data-testid="stCaptionContainer"] {
            color: var(--app-muted) !important;
        }
        a {
            color: var(--app-primary) !important;
        }
        [data-testid="stSidebar"] {
            background: var(--app-sidebar) !important;
            border-right: 1px solid var(--app-border) !important;
        }
        [data-testid="stSidebar"] * {
            color: var(--app-text) !important;
        }
        [data-testid="stSidebar"] svg {
            fill: currentColor !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: var(--app-border) !important;
        }
        .sidebar-theme-switcher {
            margin: 0.85rem 0 0;
            padding: 0.6rem 0 0.15rem;
            background: transparent;
            border-top: 1px solid var(--app-border);
        }
        .sidebar-theme-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            margin-bottom: 0.38rem;
            color: var(--app-muted) !important;
            font-size: 0.62rem;
            font-weight: 650;
        }
        .sidebar-theme-head strong {
            color: var(--app-text) !important;
            font-size: 0.6rem;
            font-weight: 600;
        }
        .sidebar-theme-options {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.22rem;
        }
        .sidebar-theme-options button {
            width: 100%;
            min-height: 1.55rem;
            padding: 0.18rem 0.12rem;
            color: var(--app-muted) !important;
            background: var(--app-surface) !important;
            border: 1px solid var(--app-border) !important;
            border-radius: 6px !important;
            font-family: inherit;
            font-size: 0.62rem;
            font-weight: 600;
            cursor: pointer;
            transition:
                color 160ms ease,
                background-color 160ms ease,
                border-color 160ms ease;
        }
        .sidebar-theme-options button:hover {
            color: var(--app-text) !important;
            background: var(--app-surface-strong) !important;
            border-color: var(--app-border-strong) !important;
        }
        .sidebar-theme-options button.is-active {
            color: var(--app-primary-text) !important;
            background: var(--app-primary) !important;
            border-color: var(--app-primary) !important;
        }
        .sidebar-theme-options button:focus-visible {
            outline: 2px solid var(--app-primary) !important;
            outline-offset: 2px;
        }
        .brand-lockup,
        .rm-page-head {
            border-color: var(--app-border) !important;
        }
        .brand-symbol {
            background: var(--app-primary) !important;
            color: var(--app-primary-text) !important;
            box-shadow: 0 0 0 1px var(--app-border-strong) !important;
        }
        .brand-mark,
        .rm-page-head h1,
        .rm-synthesis-body h2,
        .module-heading h2,
        .task-title-row h2 {
            color: var(--app-text) !important;
        }
        .brand-sub,
        .rm-page-head p,
        .rm-synthesis-body > p,
        .module-heading p,
        .task-title-note,
        .task-intro p,
        .module-text,
        .rm-citation-card p,
        .rm-step p,
        .rm-citation-note {
            color: var(--app-muted) !important;
        }
        .rm-kicker,
        .section-label,
        .rm-synthesis-label,
        .rm-source-top,
        .rm-step-index {
            color: var(--app-primary) !important;
        }
        .rm-head-status {
            color: var(--app-text) !important;
            background: var(--app-surface-soft) !important;
            border-color: var(--app-border) !important;
        }
        .rm-status-dot,
        .sidebar-engine-dot.online,
        .rm-source-mark {
            background: var(--app-success) !important;
            box-shadow: 0 0 0 3px color-mix(
                in srgb,
                var(--app-success) 14%,
                transparent
            ) !important;
        }
        .sidebar-engine-dot {
            background: var(--app-warning) !important;
        }
        .sidebar-engine,
        .rm-scope-card,
        .rm-synthesis-panel,
        .rm-citations-panel,
        div[data-testid="stForm"],
        div[data-testid="stMetric"],
        div[data-testid="stExpander"] {
            color: var(--app-text) !important;
            background: var(--app-surface) !important;
            border-color: var(--app-border) !important;
        }
        .sidebar-engine-row,
        .rm-synthesis-head,
        .rm-meta-row span,
        .rm-step,
        .rm-citation-card,
        .status-line {
            border-color: var(--app-border) !important;
        }
        .sidebar-engine-row,
        .rm-panel-head,
        .rm-scope-foot,
        .rm-citation-note,
        .footer-note {
            color: var(--app-muted) !important;
        }
        .sidebar-engine-row strong,
        .rm-scope-domain,
        .rm-panel-title,
        .rm-scope-item,
        .rm-step strong,
        .rm-citation-card strong {
            color: var(--app-text) !important;
        }
        .rm-scope-item,
        .rm-meta-row span,
        .rm-step,
        .rm-citation-card,
        .status-line {
            background: var(--app-surface-soft) !important;
        }
        .rm-citation-card:hover,
        .rm-step:hover {
            background: var(--app-surface-strong) !important;
            transition: background-color 160ms ease;
        }
        .rm-scope-empty {
            color: var(--app-muted) !important;
            background: var(--app-surface-soft) !important;
            border-color: var(--app-border-strong) !important;
        }
        .rm-synthesis-icon,
        .rm-panel-head strong {
            color: var(--app-primary) !important;
            background: var(--app-primary-soft) !important;
            border-color: var(--app-border) !important;
        }
        .rm-meta-row .rm-live-chip,
        .module-tag {
            color: var(--app-primary) !important;
            background: var(--app-primary-soft) !important;
            border-color: var(--app-border-strong) !important;
        }
        .module-index {
            color: var(--app-faint) !important;
        }
        .hero {
            color: var(--app-text) !important;
            background-color: var(--app-surface) !important;
            background-image:
                linear-gradient(var(--app-grid-line) 1px, transparent 1px),
                linear-gradient(90deg, var(--app-grid-line) 1px, transparent 1px) !important;
            border-color: var(--app-border) !important;
        }
        .hero h1,
        .hero h2,
        .hero-copy,
        .hero-console {
            color: var(--app-text) !important;
        }
        .hero-kicker,
        .hero-status-dot {
            color: var(--app-primary) !important;
        }
        .hero-status-dot {
            background: var(--app-primary) !important;
        }
        .hero-status {
            color: var(--app-muted) !important;
            background: var(--app-surface-soft) !important;
            border-color: var(--app-border) !important;
        }
        [class*="st-key-home_task_station"],
        [class*="st-key-module_panel_"] {
            background: var(--app-surface) !important;
            border-color: var(--app-border) !important;
            box-shadow:
                inset 0 1px 0 color-mix(
                    in srgb,
                    var(--app-text) 5%,
                    transparent
                ),
                0 12px 30px var(--app-shadow) !important;
        }
        [class*="st-key-module_panel_"]:hover {
            border-color: var(--app-primary) !important;
            box-shadow: 0 10px 26px var(--app-shadow) !important;
        }
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input,
        [data-testid="stChatInput"] textarea,
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
            color: var(--app-text) !important;
            -webkit-text-fill-color: var(--app-text) !important;
            background: var(--app-surface) !important;
            border-color: var(--app-border) !important;
            caret-color: var(--app-primary) !important;
        }
        [data-testid="stSidebar"] .st-key-sidebar_navigation
        [data-testid="stSelectbox"] input {
            color: var(--app-text) !important;
            -webkit-text-fill-color: var(--app-text) !important;
            background: transparent !important;
            caret-color: var(--app-primary) !important;
        }
        [data-testid="stTextInput"] [data-baseweb="base-input"],
        [data-testid="stTextInput"] [data-baseweb="input"],
        [data-testid="stTextArea"] [data-baseweb="textarea"],
        [data-testid="stNumberInput"] [data-baseweb="input"],
        [data-testid="stChatInput"],
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: var(--app-surface) !important;
            border-color: var(--app-border) !important;
            box-shadow: none !important;
        }
        [data-testid="stTextInput"] [data-baseweb="input"]:focus-within,
        [data-testid="stTextArea"] [data-baseweb="textarea"]:focus-within,
        [data-testid="stNumberInput"] [data-baseweb="input"]:focus-within,
        [data-testid="stChatInput"]:focus-within {
            border-color: var(--app-primary) !important;
            box-shadow: 0 0 0 1px var(--app-primary) !important;
        }
        input::placeholder,
        textarea::placeholder,
        [data-testid="stTextArea"] textarea::placeholder,
        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stChatInput"] textarea::placeholder {
            color: var(--app-faint) !important;
            -webkit-text-fill-color: var(--app-faint) !important;
            opacity: 1 !important;
        }
        [data-baseweb="popover"] [role="listbox"],
        [data-baseweb="popover"] [role="option"] {
            color: var(--app-text) !important;
            background: var(--app-surface) !important;
        }
        [data-baseweb="popover"] [role="option"]:hover,
        [data-baseweb="popover"] [aria-selected="true"] {
            color: var(--app-text) !important;
            background: var(--app-primary-soft) !important;
        }
        .stButton > button,
        .stDownloadButton > button,
        [data-testid="stFormSubmitButton"] button,
        [class*="st-key-home_quick_action_"] button,
        [class*="st-key-module_card_"] button {
            color: var(--app-text) !important;
            background: var(--app-surface) !important;
            border-color: var(--app-border) !important;
            box-shadow: none !important;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover,
        [data-testid="stFormSubmitButton"] button:hover,
        [class*="st-key-home_quick_action_"] button:hover,
        [class*="st-key-module_card_"] button:hover {
            color: var(--app-text) !important;
            background: var(--app-surface-strong) !important;
            border-color: var(--app-border-strong) !important;
        }
        .stButton > button[kind="primary"],
        [data-testid="stFormSubmitButton"] button[kind="primary"] {
            color: var(--app-primary-text) !important;
            background: var(--app-primary) !important;
            border-color: var(--app-primary) !important;
            box-shadow: 0 8px 20px color-mix(
                in srgb,
                var(--app-primary) 24%,
                transparent
            ) !important;
        }
        .stButton > button[kind="primary"]:hover,
        [data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
            color: var(--app-primary-text) !important;
            background: var(--app-primary-hover) !important;
            border-color: var(--app-primary-hover) !important;
        }
        .stButton > button p,
        .stDownloadButton > button p,
        [data-testid="stFormSubmitButton"] button p,
        [class*="st-key-home_quick_action_"] button p,
        [class*="st-key-module_card_"] button p {
            color: inherit !important;
        }
        div[data-testid="stExpander"] details,
        div[data-testid="stExpander"] summary {
            color: var(--app-text) !important;
            background: transparent !important;
            border-color: var(--app-border) !important;
        }
        pre,
        code,
        [data-testid="stCode"] {
            color: var(--st-code-text-color, var(--app-text)) !important;
            background-color: var(--st-code-background-color, var(--app-surface-soft)) !important;
            border-color: var(--app-border) !important;
        }
        .simulation-note {
            color: var(--app-warning) !important;
            background: color-mix(
                in srgb,
                var(--app-warning) 10%,
                var(--app-surface)
            ) !important;
            border-left-color: var(--app-warning) !important;
        }
        .footer-note {
            border-top-color: var(--app-border) !important;
        }
        </style>
        """
    token_values = {
        "__APP_BG__": palette["background"],
        "__APP_SURFACE__": palette["surface"],
        "__APP_SURFACE_SOFT__": palette["surface_soft"],
        "__APP_SURFACE_STRONG__": palette["surface_strong"],
        "__APP_SIDEBAR__": palette["sidebar"],
        "__APP_TEXT__": palette["text"],
        "__APP_MUTED__": palette["muted"],
        "__APP_FAINT__": palette["faint"],
        "__APP_PRIMARY__": palette["primary"],
        "__APP_PRIMARY_HOVER__": palette["primary_hover"],
        "__APP_PRIMARY_SOFT__": palette["primary_soft"],
        "__APP_PRIMARY_TEXT__": palette["primary_text"],
        "__APP_BORDER__": palette["border"],
        "__APP_BORDER_STRONG__": palette["border_strong"],
        "__APP_SUCCESS__": palette["success"],
        "__APP_WARNING__": palette["warning"],
        "__APP_DANGER__": palette["danger"],
        "__APP_SHADOW__": palette["shadow"],
        "__APP_GRID_LINE__": palette["grid_line"],
    }
    for token, value in token_values.items():
        styles = styles.replace(token, value)
    st.markdown(styles, unsafe_allow_html=True)


@st.cache_resource
def get_agent() -> ResearchPlannerAgent:
    return ResearchPlannerAgent()


@st.cache_resource
def get_llm_service() -> LLMService:
    return LLMService(settings)


@st.cache_resource
def get_bundled_knowledge_base() -> BundledKnowledgeBase:
    """加载并缓存项目内置文献库，避免每次页面刷新重复校验文件。"""

    return load_bundled_knowledge_base(settings.knowledge_base_dir)


@st.cache_resource
def get_literature_framework() -> LiteratureFramework:
    """加载并缓存根据当前全部论文自动生成的研究框架。"""

    return load_literature_framework(settings.knowledge_base_dir)


@st.cache_resource
def get_vector_store() -> EvidenceVectorStore:
    """返回共享的 Chroma 客户端；打开页面时不会加载向量模型。"""

    return EvidenceVectorStore(
        settings.knowledge_base_dir,
        model_name=settings.embedding_model,
        collection_name=settings.chroma_collection_name,
        cache_dir=settings.embedding_cache_dir,
    )


def _format_vector_sync_result(result: VectorSyncResult) -> str:
    """生成向量同步提示；恢复场景明确告知旧索引备份位置。"""

    message = (
        f"向量库已同步：{result.total_chunks} 个片段，"
        f"本次新增 {result.added_chunks} 个、"
        f"更新 {result.updated_chunks} 个。"
    )
    if result.recovery_backup is not None:
        message += (
            " 检测到原 HNSW 索引损坏，已隔离到 "
            f"{result.recovery_backup}，并完成自动重建。"
        )
    return message


@st.cache_data(ttl=30, show_spinner=False)
def get_home_research_snapshot() -> HomeResearchSnapshot:
    """读取首页真实论文、证据和学习记录数量，不触发论文解析或模型加载。"""

    domain_name = "尚未建立研究领域"
    papers = 0
    evidence = 0
    questions: int | None = None
    attempts: int | None = None
    source_titles: tuple[str, ...] = ()

    try:
        workflow = get_researchmind_workflow()
        bundle = load_bundled_knowledge_base(
            workflow.knowledge_base_dir
        )
        papers = len(bundle.documents)
        evidence = len(bundle.chunks)
        source_titles = tuple(
            document.title for document in bundle.documents[:3]
        )
    except (KnowledgeBaseError, OSError, ValueError):
        pass

    try:
        workflow = get_researchmind_workflow()
        domain = workflow.store.get_default_domain()
        if domain is not None:
            counts = workflow.store.counts(domain.domain_id)
            domain_name = domain.name
            questions = counts["questions"]
            attempts = counts["attempts"]
    except Exception:
        # 首页概览不能因为学习数据库暂时不可用而阻止其他页面运行。
        pass

    return HomeResearchSnapshot(
        domain_name=domain_name,
        papers=papers,
        evidence=evidence,
        questions=questions,
        attempts=attempts,
        source_titles=source_titles,
    )


def request_navigation(page: str) -> None:
    """请求切换到指定页面，并在下一次界面刷新前更新导航状态。"""

    page = RESEARCHMIND_PAGE_ALIASES.get(page, page)
    if page not in PAGES:
        raise ValueError(f"未知页面：{page}")
    st.session_state["pending_navigation"] = page
    st.rerun()


def update_navigation_from_sidebar() -> None:
    """将侧边栏下拉框的选择同步为当前页面。"""

    selected_page = RESEARCHMIND_PAGE_ALIASES.get(
        st.session_state.get("sidebar_navigation"),
        st.session_state.get("sidebar_navigation"),
    )
    if selected_page in PAGES:
        st.session_state["navigation"] = selected_page


def build_page_assistant_system_prompt(page: str) -> str:
    """构造当前功能页专属的问答系统提示。"""

    context = PAGE_ASSISTANT_CONTEXTS.get(
        page,
        "解释当前功能页的操作方法，并帮助用户判断下一步。",
    )
    return (
        f"你是体研智脑的“{page}”功能助手。\n"
        f"当前页面用途：{context}\n"
        "请使用简洁、清晰的中文回答，优先给出可执行的操作步骤。\n"
        "只回答与当前页面和体育科研相关的合理问题；超出范围时，"
        "说明应使用哪个功能。\n"
        "不得编造论文、作者、DOI、页码、实验数据或统计结果。"
        "证据不足时必须明确说明，并建议用户检索或上传真实原文核验。"
    )


def render_page_assistant(page: str) -> None:
    """渲染所有功能页共用的上下文智能问答。"""

    context = PAGE_ASSISTANT_CONTEXTS.get(
        page,
        "解释当前功能页的操作方法，并帮助用户判断下一步。",
    )
    service = get_llm_service()
    history_key = f"page_assistant_history::{page}"
    history: list[dict[str, str]] = st.session_state.setdefault(
        history_key,
        [],
    )

    with st.popover(
        "本页智能问答",
        icon=":material/forum:",
        width="stretch",
    ):
        st.markdown(f"**{page} · 页面助手**")
        st.caption(context)
        if not service.is_configured:
            st.caption("当前为模拟回答模式，不会生成真实文献引用。")

        for message in history[-4:]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("mode"):
                    st.caption(f"来源：{message['mode']}")

        with st.form(f"page_assistant_form::{page}", clear_on_submit=True):
            question = st.text_area(
                "向本页助手提问",
                placeholder="例如：这个功能第一步应该怎么做？",
                height=88,
            )
            asked = st.form_submit_button(
                "获取解答",
                type="primary",
                width="stretch",
            )

        if asked:
            clean_question = question.strip()
            if not clean_question:
                st.warning("请先输入问题。")
            else:
                with st.spinner("正在整理本页解答..."):
                    response = _safe_answer(
                        service,
                        clean_question,
                        not service.is_configured,
                        system_prompt=build_page_assistant_system_prompt(page),
                    )
                history.append({"role": "user", "content": clean_question})
                if response.success:
                    answer = response.answer
                    if response.is_simulated:
                        answer = (
                            f"**当前页面：{page}**\n\n{context}\n\n{answer}"
                        )
                    history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "mode": response.mode,
                        }
                    )
                else:
                    history.append(
                        {
                            "role": "assistant",
                            "content": f"**未能回答：** {response.message}",
                            "mode": response.mode,
                        }
                    )
                st.rerun()

        if history and st.button(
            "清空本页问答",
            key=f"page_assistant_clear::{page}",
            width="stretch",
        ):
            st.session_state[history_key] = []
            st.rerun()


def render_sidebar() -> str:
    """渲染侧边栏并返回当前页面。"""

    st.sidebar.markdown(
        """
        <div class="brand-lockup">
            <div class="brand-symbol">智</div>
            <div>
                <div class="brand-mark">体研智脑</div>
                <div class="brand-sub">Sports Research Agent</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    pending_page = st.session_state.pop("pending_navigation", None)
    pending_page = RESEARCHMIND_PAGE_ALIASES.get(pending_page, pending_page)
    if pending_page in PAGES:
        st.session_state["navigation"] = pending_page

    page = RESEARCHMIND_PAGE_ALIASES.get(
        st.session_state.get("navigation", "首页"),
        st.session_state.get("navigation", "首页"),
    )
    if page not in PAGES:
        page = "首页"
    st.session_state["navigation"] = page

    with st.sidebar:
        # 页面入口保持单层结构，避免侧边栏被多组二级菜单占满。
        if st.session_state.get("sidebar_navigation") != page:
            st.session_state["sidebar_navigation"] = page
        st.selectbox(
            "工作空间",
            options=PAGES,
            key="sidebar_navigation",
            format_func=lambda value: value,
            on_change=update_navigation_from_sidebar,
        )

        render_page_assistant(page)

        st.divider()
        engine_status = "模型已配置" if settings.llm_configured else "本地模拟模式"
        engine_class = "online" if settings.llm_configured else "offline"
        st.markdown(
            f"""
            <div class="sidebar-engine">
                <div class="sidebar-engine-head">
                    <span class="sidebar-engine-dot {engine_class}"></span>
                    <span>{engine_status}</span>
                </div>
                <div class="sidebar-engine-row">
                    <span>论文库</span><strong>LOCAL</strong>
                </div>
                <div class="sidebar-engine-row">
                    <span>学习记录</span><strong>SQLITE</strong>
                </div>
                <div class="sidebar-engine-row">
                    <span>研究流程</span><strong>6 AGENTS</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("本地数据 · 来源可追溯")
        st.html(
            THEME_SWITCHER_HTML,
            unsafe_allow_javascript=True,
        )
    return page


def render_hero(snapshot: HomeResearchSnapshot) -> None:
    """渲染首页顶部的科研工作台标题与运行状态。"""

    hero_status = "模型已配置" if settings.llm_configured else "本地模拟模式"
    st.html(
        f"""
        <header class="rm-page-head">
            <div class="rm-page-copy">
                <div class="rm-kicker">SPORTS RESEARCH INTELLIGENCE</div>
                <h1>{escape(settings.app_name)}</h1>
                <p>从研究问题、真实论文证据到数据分析和学习反馈，
                组织成一条可以验证和追溯的科研路径。</p>
            </div>
            <div class="rm-head-status">
                <span class="rm-status-dot"></span>
                <span>{hero_status}</span>
            </div>
        </header>
        """
    )


def render_home_quick_nav(
    request_navigation: Callable[[str], None],
) -> None:
    """渲染首页顶部的紧凑功能入口。"""

    actions = (
        ("科研工作台", "科研工作台", ":material/space_dashboard:"),
        ("文献采集", "文献自动采集", ":material/library_add:"),
        ("证据问答", "文献证据问答", ":material/manage_search:"),
        ("智能问答", "智能科研问答", ":material/forum:"),
    )
    columns = st.columns(len(actions), gap="small")
    for index, (column, action) in enumerate(zip(columns, actions, strict=True)):
        label, target_page, icon = action
        with column:
            if st.button(
                label,
                key=f"home_quick_action_{index}",
                icon=icon,
                width="stretch",
            ):
                request_navigation(target_page)


def render_home_task_form() -> tuple[bool, str]:
    """渲染参考图风格的科研任务输入区。"""

    st.markdown(
        """
        <div class="rm-command-heading">
            <span>AI RESEARCH COMMAND</span>
            <small>输入问题后进入可追溯的科研工作流</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key="home_task_station"):
        with st.form("home_task_form", border=False):
            task = st.text_area(
                "科研任务",
                placeholder=(
                    "分析研究方法与核心结论，例如：研究青少年体育锻炼干预"
                    "与体质健康水平的关系，并制定数据分析计划"
                ),
                label_visibility="collapsed",
                height=106,
                key="home_task_input",
            )
            button_spacer, button_column = st.columns(
                [3.2, 1.0],
                gap="small",
                vertical_alignment="bottom",
            )
            with button_spacer:
                st.caption("结果会保留论文文件、原文页码和真实分析数据")
            with button_column:
                submitted = st.form_submit_button(
                    "启动 AI 研究",
                    type="primary",
                    icon=":material/auto_awesome:",
                    width="stretch",
                )
    return submitted, task


def render_home_paper_scope(snapshot: HomeResearchSnapshot) -> None:
    """渲染首页右侧的真实论文范围卡片。"""

    source_items = "".join(
        f"""
        <div class="rm-scope-item">
            <span class="rm-source-mark"></span>
            <span>{escape(title)}</span>
        </div>
        """
        for title in snapshot.source_titles
    )
    if not source_items:
        source_items = (
            '<div class="rm-scope-empty">当前尚未导入论文，'
            "可从文献采集开始。</div>"
        )
    st.html(
        f"""
        <aside class="rm-scope-card">
            <div class="rm-panel-head">
                <span>PAPER SCOPE</span>
                <strong>{snapshot.papers}</strong>
            </div>
            <div class="rm-scope-domain">{escape(snapshot.domain_name)}</div>
            <div class="rm-scope-list">{source_items}</div>
            <div class="rm-scope-foot">
                <span>{snapshot.evidence} 条证据片段</span>
                <span>LOCAL INDEX</span>
            </div>
        </aside>
        """
    )


def render_home_research_panel(snapshot: HomeResearchSnapshot) -> None:
    """渲染参考图风格的研究工作流与来源引用面板。"""

    workflow_html = "".join(
        f"""
        <div class="rm-step">
            <span class="rm-step-index">{index}</span>
            <div>
                <strong>{escape(title)}</strong>
                <p>{escape(description)}</p>
            </div>
        </div>
        """
        for index, title, description in HOME_WORKFLOW_STEPS
    )
    source_titles = snapshot.source_titles or ("尚未导入论文",)
    source_cards = "".join(
        f"""
        <article class="rm-citation-card">
            <div class="rm-source-top">
                <span>SR-{index:02d}</span>
                <span>{"PDF" if snapshot.source_titles else "EMPTY"}</span>
            </div>
            <strong>{escape(title)}</strong>
            <p>来自当前本地论文索引，可在文献证据问答中追踪原文和页码。</p>
        </article>
        """
        for index, title in enumerate(source_titles, start=1)
    )
    database_label = (
        "学习数据库已连接"
        if snapshot.questions is not None
        else "学习数据库待连接"
    )
    st.html(
        f"""
        <section class="rm-research-grid">
            <article class="rm-synthesis-panel">
                <div class="rm-synthesis-head">
                    <div class="rm-synthesis-icon">AI</div>
                    <div>
                        <div class="rm-panel-title">AI SYNTHESIS</div>
                        <div class="rm-meta-row">
                            <span>{database_label}</span>
                            <span>{snapshot.papers} papers</span>
                            <span>{snapshot.evidence} chunks</span>
                            <span class="rm-live-chip">LOCAL</span>
                        </div>
                    </div>
                </div>
                <div class="rm-synthesis-body">
                    <div class="rm-synthesis-label">RESEARCH PATH</div>
                    <h2>从研究问题到可验证证据</h2>
                    <p>每个阶段都保留输入、工具结果和证据来源，避免把模拟内容
                    当成真实科研结论。</p>
                    <div class="rm-step-grid">{workflow_html}</div>
                </div>
            </article>
            <aside class="rm-citations-panel">
                <div class="rm-panel-head">
                    <span>SOURCE CITATIONS</span>
                    <strong>{snapshot.papers}</strong>
                </div>
                <div class="rm-citation-list">{source_cards}</div>
                <div class="rm-citation-note">
                    引用卡片只展示本地索引中的真实论文标题。
                </div>
            </aside>
        </section>
        """
    )


def render_module_overview(request_navigation: Callable[[str], None]) -> None:
    """展示可进入对应页面的功能模块卡片。"""

    st.markdown(
        """
        <div class="module-heading">
            <div>
                <div class="section-label">RESEARCH CAPABILITY MATRIX</div>
                <h2>科研能力矩阵</h2>
            </div>
            <p>从研究规划到学习反馈，十个真实功能入口。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    module_pairs = [
        MODULE_CARDS[index : index + 2]
        for index in range(0, len(MODULE_CARDS), 2)
    ]
    card_index = 0
    for module_pair in module_pairs:
        columns = st.columns(2, gap="small")
        for column, module in zip(columns, module_pair, strict=False):
            title, description, tag, target_page, icon = module
            with column:
                with st.container(
                    border=True,
                    height="stretch",
                    key=f"module_panel_{card_index}",
                    gap=6,
                ):
                    st.markdown(
                        f"""
                        <div class="module-overline">
                            <span class="module-index">C{card_index + 1:02d}</span>
                            <span class="module-tag">{tag}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        title,
                        key=f"module_card_{card_index}",
                        icon=icon,
                        width="stretch",
                        help=f"进入“{target_page}”页面",
                    ):
                        request_navigation(target_page)
                    st.markdown(
                        f'<div class="module-text">{description}</div>',
                        unsafe_allow_html=True,
                    )
            card_index += 1


def render_home(request_navigation: Callable[[str], None]) -> None:
    snapshot = get_home_research_snapshot()
    render_hero(snapshot)
    render_home_quick_nav(request_navigation)
    st.space("small")

    command_column, scope_column = st.columns(
        [2.55, 0.9],
        gap="medium",
        vertical_alignment="top",
    )
    with command_column:
        submitted, task = render_home_task_form()
    with scope_column:
        render_home_paper_scope(snapshot)

    if submitted:
        clean_task = task.strip()
        if not clean_task:
            st.error("请输入科研任务。")
        else:
            st.session_state["workbench_prefill"] = clean_task
            request_navigation("科研工作台")

    st.space(8)
    render_home_research_panel(snapshot)
    st.space(12)
    render_module_overview(request_navigation)


def render_question_answer() -> None:
    """渲染科研问答页面。"""

    st.markdown('<div class="section-label">RESEARCH Q&A</div>', unsafe_allow_html=True)
    st.title("智能科研问答")
    st.write("围绕研究设计、统计方法、论文写作和科研规范提出问题。")

    service = get_llm_service()
    if not service.is_configured:
        st.info(
            "当前未配置 OPENAI_API_KEY，系统会使用本地模拟模板。"
            "模拟回答不会提供真实文献引用或统计结论。"
        )
        use_real_model = False
    else:
        selected_mode = st.radio(
            "回答模式",
            ("模拟回答", "调用已配置大模型"),
            horizontal=True,
        )
        use_real_model = selected_mode == "调用已配置大模型"
        st.caption(service.status_text())

    history: list[dict[str, str]] = st.session_state.setdefault("qa_history", [])
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("mode"):
                st.caption(f"来源：{message['mode']}")

    with st.form("qa_form", clear_on_submit=True):
        question = st.text_area(
            "科研问题",
            placeholder="例如：准实验研究的前测和后测分别适合使用什么分析方法？",
            height=100,
        )
        ask_submitted = st.form_submit_button("提交问题", type="primary")

    if ask_submitted:
        clean_question = question.strip()
        if not clean_question:
            st.error("请输入科研问题。")
        else:
            with st.spinner("正在整理回答..."):
                response = _safe_answer(service, clean_question, not use_real_model)
            history.append({"role": "user", "content": clean_question})
            if response.success:
                history.append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "mode": response.mode,
                    }
                )
            else:
                history.append(
                    {
                        "role": "assistant",
                        "content": f"**未能回答：** {response.message}",
                        "mode": response.mode,
                    }
                )
            st.rerun()

    if history and st.button("清空问答记录"):
        st.session_state["qa_history"] = []
        st.rerun()


def _safe_answer(
    service: LLMService,
    question: str,
    allow_simulation: bool,
    system_prompt: str = "",
) -> LLMResponse:
    """捕获意外异常，保证问答页面不会因服务错误崩溃。"""

    try:
        return service.answer(
            question,
            allow_simulation=allow_simulation,
            system_prompt=system_prompt,
        )
    except Exception as exc:
        return LLMResponse(
            answer="",
            mode="调用失败",
            success=False,
            message=f"服务出现未预期错误：{exc}",
        )


def render_topic_selection() -> None:
    """渲染科研选题页面。"""

    st.markdown('<div class="section-label">TOPIC DISCOVERY</div>', unsafe_allow_html=True)
    st.title("科研选题")
    st.write("输入一个研究方向，使用固定模板生成候选选题。生成结果不是文献事实。")

    with st.form("topic_form"):
        direction = st.text_input(
            "研究方向",
            placeholder="例如：体育课堂中的即时反馈",
        )
        count = st.slider("候选数量", min_value=1, max_value=5, value=3)
        submitted = st.form_submit_button("生成选题建议", type="primary")

    if submitted:
        try:
            suggestions = get_agent().generate_topic_suggestions(direction, count)
            st.session_state["topic_direction"] = direction.strip()
            st.session_state["topic_suggestions"] = suggestions
        except ValueError as exc:
            st.error(str(exc))

    suggestions = st.session_state.get("topic_suggestions", ())
    if suggestions:
        st.markdown(
            f'<div class="simulation-note">{SIMULATION_NOTICE}</div>',
            unsafe_allow_html=True,
        )
        for index, suggestion in enumerate(suggestions, start=1):
            with st.expander(f"选题 {index}：{suggestion.title}", expanded=index == 1):
                st.markdown("**研究背景**")
                st.write(suggestion.background)
                st.markdown("**研究问题**")
                st.write(suggestion.research_question)
                st.markdown("**研究对象**")
                st.write(suggestion.research_subject)
                st.markdown("**研究价值**")
                st.write(suggestion.research_value)
                st.markdown("**设计提示**")
                st.write(suggestion.suggested_design)

        st.divider()
        st.subheader("文献检索策略")
        st.caption("只生成检索表达式，不联网返回论文，不产生虚构引用。")
        try:
            strategy = build_search_strategy(
                st.session_state.get("topic_direction", "")
            )
        except ValueError:
            strategy = None
        if strategy:
            st.markdown("**中文数据库检索式草稿**")
            st.code(strategy.chinese_query, language="text")
            st.markdown("**英文检索结构提示**")
            st.code(strategy.english_query_hint, language="text")
            st.markdown("**建议数据库**")
            st.write("、".join(strategy.databases))
            with st.expander("检索与引用注意事项"):
                for note in strategy.notes:
                    st.write(f"- {note}")


def render_pdf_reader() -> None:
    """渲染 PDF 文献阅读页面。"""

    st.markdown('<div class="section-label">LITERATURE READER</div>', unsafe_allow_html=True)
    st.title("PDF 文献阅读")
    st.write("上传 PDF 后提取可复制文本；扫描版 PDF 暂不包含 OCR。")

    uploaded = st.file_uploader("选择 PDF 文件", type=["pdf"])
    if uploaded is None:
        st.caption(f"单个文件上限：{settings.max_pdf_size_mb} MB。")
        return

    file_bytes = uploaded.getvalue()
    max_bytes = settings.max_pdf_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        st.error(f"文件超过 {settings.max_pdf_size_mb} MB 限制，请压缩后重试。")
        return

    st.caption(f"文件：{uploaded.name}　|　大小：{len(file_bytes) / 1024:.1f} KB")
    if st.button("开始提取文本", type="primary"):
        with st.spinner("正在解析 PDF..."):
            result = extract_pdf_text(
                file_bytes,
                max_pages=settings.max_pdf_pages,
                max_chars=settings.max_pdf_chars,
            )
        st.session_state["pdf_result"] = result
        st.session_state["pdf_result_name"] = uploaded.name

    if st.session_state.get("pdf_result_name") != uploaded.name:
        return

    result = st.session_state.get("pdf_result")
    if result is None:
        return
    if not result.success:
        st.error(result.message)
        return

    columns = st.columns(3)
    columns[0].metric("总页数", result.page_count)
    columns[1].metric("已处理页数", result.processed_pages)
    columns[2].metric("提取字符数", result.char_count)
    if result.truncated:
        st.warning(result.message)
    elif not result.text:
        st.warning(result.message)
    else:
        st.success(result.message)

    if result.text:
        st.text_area("提取文本", value=result.text, height=480)
        st.info(
            "提取文本仅供阅读整理。引用前必须核对原文、作者、年份和页码，"
            "系统不会自动生成文献引用。"
        )


def render_literature_framework() -> None:
    """展示自动生成的研究框架，并支持持久化导入新论文。"""

    st.markdown(
        '<div class="section-label">LITERATURE FRAMEWORK</div>',
        unsafe_allow_html=True,
    )
    st.title("文献研究框架")
    st.write(
        "系统从已导入论文中检索真实原文，把证据归入固定的九个研究维度。"
        "新增论文通过界面导入或放入文献目录后，会同步进入证据索引、"
        "向量库和研究框架。"
    )
    st.warning(
        "当前归类使用关键词规则匹配，不是大模型综述。"
        "证据摘要均为 PDF 原文片段，引用前必须回到论文核对题名、页码和上下文。",
        icon=":material/fact_check:",
    )

    _sync_pdf_directory_once()
    rebuild_notice = st.session_state.pop("framework_rebuild_notice", None)
    if rebuild_notice:
        st.success(rebuild_notice)
    ingest_result = st.session_state.pop("framework_ingest_result", None)
    if isinstance(ingest_result, IngestResult):
        _render_ingest_result(ingest_result)

    try:
        bundle = get_bundled_knowledge_base()
        framework = get_literature_framework()
    except KnowledgeBaseError as exc:
        st.error(f"文献框架暂时无法加载：{exc}")
        st.info(
            "如果现有索引损坏，可尝试重建。重建只读取已登记的 PDF，"
            "不会生成论文中没有的内容。"
        )
        _render_library_rebuild_button()
        return
    except Exception as exc:
        st.error(f"加载文献框架时出现未预期错误：{exc}")
        return

    metric_columns = st.columns(4)
    metric_columns[0].metric("已纳入论文", f"{framework.documents_count} 篇")
    metric_columns[1].metric("研究维度", f"{len(framework.dimensions)} 个")
    metric_columns[2].metric("证据映射", f"{framework.evidence_count} 条")
    metric_columns[3].metric("框架归类", "规则 + BM25")

    with st.container(border=True):
        st.subheader(framework.name)
        st.write(framework.description)
        st.markdown("**研究逻辑链**")
        st.markdown(" → ".join(f"**{item}**" for item in framework.logic_chain))
        st.caption(
            f"框架生成时间：{framework.built_at}　|　"
            "同一原文片段可以服务多个研究维度"
        )

    st.subheader("框架论文")
    document_rows = []
    for document in bundle.documents:
        mapped_count = sum(
            1
            for dimension in framework.dimensions
            for evidence in dimension.evidence
            if evidence.document_id == document.document_id
        )
        document_rows.append(
            {
                "题名": document.title,
                "作者": document.authors_text,
                "期刊": document.journal,
                "年份": document.year,
                "本地文件": document.file_name,
                "页数": document.page_count,
                "证据映射": mapped_count,
            }
        )
    st.dataframe(
        pd.DataFrame(document_rows),
        width="stretch",
        hide_index=True,
        alt="研究框架已纳入论文清单",
    )

    st.subheader("九个研究维度")
    for index, dimension in enumerate(framework.dimensions, start=1):
        with st.expander(
            f"{index:02d}　{dimension.title}　"
            f"（{dimension.document_count} 篇 / {len(dimension.evidence)} 条证据）",
            expanded=index == 1,
        ):
            st.write(dimension.description)
            st.caption("规则关键词：" + "、".join(dimension.keywords))
            if not dimension.evidence:
                st.info("当前文献中没有命中该维度的原文片段。")
                continue

            for evidence in dimension.evidence:
                st.markdown(f"**{evidence.document_title}**")
                st.caption(
                    f"{evidence.file_name}　|　第 {evidence.page_number} 页　|　"
                    f"片段 {evidence.chunk_id}　|　匹配分 {evidence.score:.3f}"
                )
                st.write(evidence.excerpt)
                st.divider()

    download_data = json.dumps(
        framework.to_dict(),
        ensure_ascii=False,
        indent=2,
    )
    st.download_button(
        "下载研究框架 JSON",
        data=download_data.encode("utf-8"),
        file_name="framework.json",
        mime="application/json",
        icon=":material/download:",
    )

    st.divider()
    st.subheader("添加论文并自动更新框架")
    st.caption(
        "上传成功后会永久保存到 "
        f"`{settings.knowledge_base_dir / DOCUMENT_PDF_DIR}`。"
        "系统会自动识别书目信息、去重、重建证据索引、同步向量库并更新框架。"
    )
    with st.form("framework_pdf_upload_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "选择一篇或多篇 PDF",
            type=["pdf"],
            accept_multiple_files=True,
            key="framework_pdf_uploader",
        )
        submitted = st.form_submit_button(
            "导入论文并更新框架",
            type="primary",
            icon=":material/library_add:",
        )

    if submitted:
        if not uploaded_files:
            st.error("请先选择 PDF 文件。")
        elif len(uploaded_files) > settings.rag_max_documents:
            st.error(f"一次最多导入 {settings.rag_max_documents} 篇 PDF。")
        else:
            incoming = [
                IncomingPDF(uploaded_file.name, uploaded_file.getvalue())
                for uploaded_file in uploaded_files
            ]
            _run_pdf_ingest(incoming)

    with st.container(horizontal=True, vertical_alignment="center"):
        if st.button(
            "扫描文献目录",
            icon=":material/folder_open:",
            help="扫描 knowledge_base/pdfs 中尚未登记的 PDF。",
        ):
            result = _run_pdf_directory_sync(show_empty_message=False)
            if result is not None and not result.imported_count:
                st.info("文献目录中没有发现尚未登记的新 PDF。")
        _render_library_rebuild_button()

    st.caption(
        "也可以把 PDF 直接放入 knowledge_base/pdfs，然后点击“扫描文献目录”。"
        "扫描版 PDF 暂不支持 OCR。"
    )


def _sync_pdf_directory_once() -> None:
    """每个浏览器会话首次打开框架页时，自动同步文献目录一次。"""

    session_key = "framework_directory_initial_sync"
    if st.session_state.get(session_key):
        return
    st.session_state[session_key] = True
    _run_pdf_directory_sync(show_empty_message=False)


def _run_pdf_directory_sync(*, show_empty_message: bool) -> IngestResult | None:
    """扫描文献目录，导入新文件并刷新相关缓存。"""

    try:
        with st.spinner("正在扫描文献并同步证据索引、向量库与框架..."):
            result = sync_pdf_directory(settings.knowledge_base_dir)
    except Exception as exc:
        st.error(f"扫描文献目录失败：{exc}")
        return None

    if result.imported_count:
        _refresh_library_after_import(result)
        st.rerun()
    elif result.skipped:
        _render_ingest_result(result)
    elif show_empty_message:
        st.info("文献目录中没有发现尚未登记的新 PDF。")
    return result


def _run_pdf_ingest(incoming: list[IncomingPDF]) -> None:
    """导入上传的 PDF，并在成功后刷新页面和证据会话。"""

    try:
        with st.spinner("正在导入论文并同步证据索引、向量库与框架..."):
            result = ingest_pdf_files(incoming, settings.knowledge_base_dir)
    except Exception as exc:
        st.error(f"论文导入失败：{exc}")
        return

    if result.imported_count:
        _refresh_library_after_import(result)
        st.rerun()
    else:
        _render_ingest_result(result)


def _refresh_library_after_import(result: IngestResult) -> None:
    """清理旧缓存，并把最新文献证据同步到当前问答会话。"""

    get_bundled_knowledge_base.clear()
    get_literature_framework.clear()
    bundle = get_bundled_knowledge_base()
    _set_evidence_index(
        bundle.chunks,
        document_count=len(bundle.documents),
        source=f"持久化文献库（{len(bundle.documents)} 篇）",
    )
    st.session_state["framework_ingest_result"] = result


def _render_ingest_result(result: IngestResult) -> None:
    """展示一次导入的论文、重复文件、跳过文件和元数据提示。"""

    if result.imported:
        st.success(
            "已导入 "
            + "、".join(
                f"《{document.title}》（{document.file_name}）"
                for document in result.imported
            )
        )
        st.caption("证据索引和研究框架已根据当前全部论文自动重建。")
    if result.vector_sync is not None:
        st.caption(
            "向量库已同步："
            f"{result.vector_sync.total_chunks} 个片段，"
            f"本次新增 {result.vector_sync.added_chunks} 个、"
            f"更新 {result.vector_sync.updated_chunks} 个、"
            f"删除 {result.vector_sync.removed_chunks} 个。"
        )
    elif result.vector_warning:
        st.warning(result.vector_warning)
    if result.duplicates:
        st.info("重复文件未再次导入：" + "；".join(result.duplicates))
    if result.skipped:
        st.warning("未导入文件：" + "；".join(result.skipped))
    if result.warnings:
        with st.expander(f"书目信息核对提示（{len(result.warnings)} 条）"):
            for warning in result.warnings:
                st.warning(warning)


def _render_library_rebuild_button() -> None:
    """提供一个显式重建入口，便于修复索引和框架不一致。"""

    if st.button(
        "重建索引与框架",
        icon=":material/refresh:",
        help="重新读取已登记 PDF，重建证据索引、向量库和 framework.json。",
    ):
        try:
            with st.spinner("正在重建证据索引、向量库和研究框架..."):
                result = rebuild_library(settings.knowledge_base_dir)
        except Exception as exc:
            st.error(f"重建失败：{exc}")
            return

        bundle = result.bundle
        framework = result.framework
        get_bundled_knowledge_base.clear()
        get_literature_framework.clear()
        _set_evidence_index(
            bundle.chunks,
            document_count=len(bundle.documents),
            source=f"持久化文献库（{len(bundle.documents)} 篇）",
        )
        st.session_state["framework_rebuild_notice"] = (
            f"已重建：{framework.documents_count} 篇论文，"
            f"{len(framework.dimensions)} 个维度，"
            f"{framework.evidence_count} 条证据映射；"
            + (
                f"向量库 {result.vector_sync.total_chunks} 个片段已同步。"
                if result.vector_sync is not None
                else result.vector_warning
            )
        )
        st.rerun()


def render_evidence_qa() -> None:
    """渲染基于真实 PDF 原文的证据问答页面。"""

    st.markdown('<div class="section-label">EVIDENCE RAG</div>', unsafe_allow_html=True)
    st.title("PDF 文献证据问答")
    st.write(
        "项目内置文献和上传的真实 PDF 都会按页切分原文并检索相关内容。"
        "回答区始终展示文件名和页码，便于回到原文核验。"
    )
    st.info(
        "默认使用 BM25 + 本地多语言语义混合检索，底层由 Chroma 持久化。"
        "向量模型或向量库暂不可用时会自动回退 BM25，并明确提示。"
        "未配置 API Key 时只展示原文摘录；不会把摘录包装成模型综合结论。"
    )

    bundle: BundledKnowledgeBase | None = None
    try:
        bundle = get_bundled_knowledge_base()
    except KnowledgeBaseError as exc:
        st.error(f"项目内置文献库暂不可用：{exc}")

    st.subheader("项目内置文献库")
    if bundle is not None:
        st.caption(bundle.description)
        document_rows = [
            {
                "题名": document.title,
                "作者": document.authors_text,
                "来源": (
                    f"{document.journal}, {document.year}, "
                    f"{document.volume}({document.issue})"
                ),
                "DOI": document.doi,
                "页数": document.page_count,
            }
            for document in bundle.documents
        ]
        st.dataframe(
            pd.DataFrame(document_rows),
            width="stretch",
            hide_index=True,
        )
        with st.expander("查看论文元数据与规范引用"):
            for document in bundle.documents:
                st.markdown(f"**{document.title}**")
                st.caption(document.reference)
                st.caption(
                    f"本地文件：{document.file_name}　|　"
                    f"许可：{document.license}"
                )

        if "evidence_index" not in st.session_state:
            _set_evidence_index(
                bundle.chunks,
                document_count=len(bundle.documents),
                source=f"内置文献库（{len(bundle.documents)} 篇）",
            )
        if st.button("重新加载内置文献库", key="load_bundled_library"):
            _set_evidence_index(
                bundle.chunks,
                document_count=len(bundle.documents),
                source=f"内置文献库（{len(bundle.documents)} 篇）",
            )
            st.success("内置文献库已重新加载。")
    else:
        st.code(
            r".\.venv\Scripts\python.exe -m tools.knowledge_base --build",
            language="powershell",
        )

    with st.expander("上传自己的 PDF，建立临时证据库"):
        st.caption(
            "临时证据库只在当前浏览器会话中保留，使用 BM25 关键词检索。"
            "需要语义检索并长期保存时，请到“文献框架”页面导入。"
        )
        uploaded_files = st.file_uploader(
            "选择 1-5 篇 PDF 文献",
            type=["pdf"],
            accept_multiple_files=True,
            key="evidence_pdf_uploader",
        )

        if uploaded_files:
            if len(uploaded_files) > settings.rag_max_documents:
                st.error(
                    f"一次最多建立 {settings.rag_max_documents} 篇 PDF 的证据库，"
                    "请减少文件数量。"
                )
            else:
                st.caption(
                    "　".join(uploaded_file.name for uploaded_file in uploaded_files)
                )
                if st.button("建立临时证据库", type="primary"):
                    _build_evidence_index(uploaded_files)

    index = st.session_state.get("evidence_index")
    if not isinstance(index, (LocalEvidenceIndex, HybridEvidenceIndex)):
        st.caption("请重新加载内置文献库，或上传 PDF 后建立临时证据库。")
        return
    if not index.chunks:
        st.warning("当前证据库中没有论文，请先导入论文或加载内置文献库。")
        return

    vector_status = index.status() if isinstance(index, HybridEvidenceIndex) else None
    if vector_status is not None:
        if vector_status.synchronized:
            st.success(
                vector_status.message,
                icon=":material/database:",
            )
        else:
            st.warning(
                vector_status.message,
                icon=":material/database_off:",
            )
            if bundle is not None and st.button(
                "构建或更新本地向量库",
                icon=":material/sync:",
                help=(
                    "首次构建需要下载多语言向量模型，并会为新增或变化的"
                    "证据片段生成本地向量。"
                ),
            ):
                try:
                    with st.spinner(
                        "正在生成向量并同步 Chroma，首次运行可能需要数分钟..."
                    ):
                        sync_result = sync_bundled_vectors(
                            bundle,
                            base_dir=settings.knowledge_base_dir,
                            vector_store=get_vector_store(),
                        )
                except (VectorStoreError, ValueError, OSError) as exc:
                    st.error(f"向量库构建失败：{exc}")
                else:
                    st.session_state["vector_sync_notice"] = sync_result
                    _set_evidence_index(
                        bundle.chunks,
                        document_count=len(bundle.documents),
                        source=f"持久化文献库（{len(bundle.documents)} 篇）",
                    )
                    st.rerun()

    vector_sync_notice = st.session_state.pop("vector_sync_notice", None)
    if isinstance(vector_sync_notice, VectorSyncResult):
        st.success(_format_vector_sync_result(vector_sync_notice))

    metrics = st.columns(3)
    metrics[0].metric(
        "已建立文档",
        st.session_state.get("evidence_document_count", 0),
    )
    metrics[1].metric("证据片段", len(index.chunks))
    metrics[2].metric(
        "检索方式",
        (
            "BM25 + 语义"
            if isinstance(index, HybridEvidenceIndex)
            else "BM25（临时库）"
        ),
    )
    st.caption(
        f"当前证据库：{st.session_state.get('evidence_source', '未知来源')}"
    )

    build_warnings = st.session_state.get("evidence_build_warnings", [])
    if build_warnings:
        with st.expander(f"文件处理提示（{len(build_warnings)} 条）"):
            for warning in build_warnings:
                st.warning(warning)

    service = get_llm_service()
    if service.is_configured:
        selected_mode = st.radio(
            "回答模式",
            ("本地证据摘录", "调用大模型基于证据回答"),
            horizontal=True,
            key="evidence_answer_mode",
        )
        use_llm = selected_mode == "调用大模型基于证据回答"
    else:
        use_llm = False
        st.caption("未配置 API Key，当前仅可使用本地证据摘录。")

    with st.form("evidence_question_form"):
        question = st.text_area(
            "围绕当前文献库提问",
            placeholder="例如：这篇研究采用了什么研究设计？主要结局指标有哪些？",
            height=100,
        )
        max_retrieval_count = min(MAX_RETRIEVAL_COUNT, len(index.chunks))
        default_retrieval_count = min(
            normalize_retrieval_count(settings.rag_top_k),
            max_retrieval_count,
        )
        top_k = st.slider(
            "检索证据数量",
            min_value=1,
            max_value=max_retrieval_count,
            value=default_retrieval_count,
        )
        if isinstance(index, HybridEvidenceIndex):
            retrieval_labels = {
                "混合检索": "hybrid",
                "语义检索": "semantic",
                "BM25 检索": "bm25",
            }
            default_label = {
                "hybrid": "混合检索",
                "semantic": "语义检索",
                "bm25": "BM25 检索",
            }.get(settings.rag_retrieval_mode, "混合检索")
            selected_retrieval = st.segmented_control(
                "检索方式",
                tuple(retrieval_labels),
                default=default_label,
                key="evidence_retrieval_mode",
            )
            retrieval_mode = retrieval_labels[
                selected_retrieval or default_label
            ]
        else:
            retrieval_mode = "bm25"
        asked = st.form_submit_button("检索并回答", type="primary")

    if asked:
        try:
            with st.spinner("正在检索原文证据..."):
                answer = EvidenceRAGService(index, service).answer_question(
                    question,
                    top_k=top_k,
                    use_llm=use_llm,
                    retrieval_mode=retrieval_mode,
                )
        except Exception as exc:
            st.error(f"证据检索失败：{exc}")
        else:
            st.session_state["evidence_answer"] = answer

    answer = st.session_state.get("evidence_answer")
    if answer is not None:
        st.divider()
        st.subheader("回答")
        if answer.success:
            st.markdown(answer.answer)
        else:
            st.error(answer.message)
        st.caption(f"回答模式：{answer.mode}")
        if answer.message and answer.success:
            st.info(answer.message)

        if answer.evidence:
            st.subheader("检索到的证据来源")
            documents_by_file = (
                {
                    document.file_name: document
                    for document in bundle.documents
                }
                if bundle is not None
                else {}
            )
            for evidence_index, item in enumerate(answer.evidence, start=1):
                with st.expander(
                    f"证据 {evidence_index} · {item.chunk.citation}",
                    expanded=evidence_index == 1,
                ):
                    document = documents_by_file.get(item.chunk.document_name)
                    if document is not None:
                        st.markdown(f"**{document.title}**")
                        st.caption(document.reference)
                    source_text = "、".join(item.retrieval_sources) or "BM25"
                    st.caption(
                        f"融合分数：{item.score:.3f}　|　来源：{source_text}　|　"
                        f"BM25：{item.bm25_score:.3f}　|　"
                        f"语义：{item.semantic_score:.3f}"
                    )
                    st.write(item.chunk.text)

    if st.button("清空当前证据库"):
        for key in (
            "evidence_index",
            "evidence_document_count",
            "evidence_source",
            "evidence_build_warnings",
            "evidence_answer",
        ):
            st.session_state.pop(key, None)
        st.session_state["evidence_index"] = None
        st.rerun()


def _set_evidence_index(
    chunks: Iterable[EvidenceChunk],
    *,
    document_count: int,
    source: str,
    warnings: list[str] | None = None,
    use_vector_store: bool = True,
) -> None:
    """把证据片段写入当前浏览器会话，并统一清理旧回答。"""

    chunk_tuple = tuple(chunks)
    if use_vector_store:
        st.session_state["evidence_index"] = HybridEvidenceIndex(
            chunk_tuple,
            vector_store=get_vector_store(),
        )
    else:
        st.session_state["evidence_index"] = LocalEvidenceIndex(chunk_tuple)
    st.session_state["evidence_document_count"] = document_count
    st.session_state["evidence_source"] = source
    st.session_state["evidence_build_warnings"] = warnings or []
    st.session_state.pop("evidence_answer", None)


def _build_evidence_index(uploaded_files) -> None:
    """解析上传文件、切分证据并保存到当前浏览器会话。"""

    chunks = []
    warnings: list[str] = []
    successful_documents = 0
    max_bytes = settings.max_pdf_size_mb * 1024 * 1024

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        if not file_bytes:
            warnings.append(f"{uploaded_file.name}：文件为空，已跳过。")
            continue
        if len(file_bytes) > max_bytes:
            warnings.append(
                f"{uploaded_file.name}：超过 {settings.max_pdf_size_mb} MB，已跳过。"
            )
            continue

        result = extract_pdf_text(
            file_bytes,
            max_pages=settings.max_pdf_pages,
            max_chars=settings.max_pdf_chars,
        )
        if not result.success:
            warnings.append(f"{uploaded_file.name}：{result.message}")
            continue
        if not result.pages:
            warnings.append(
                f"{uploaded_file.name}：没有可检索文本，可能是扫描版 PDF。"
            )
            continue

        document_chunks = chunk_pdf_pages(
            uploaded_file.name,
            result.pages,
            max_chars=settings.rag_chunk_chars,
            overlap_chars=settings.rag_chunk_overlap,
        )
        if not document_chunks:
            warnings.append(f"{uploaded_file.name}：未生成有效证据片段。")
            continue

        chunks.extend(document_chunks)
        successful_documents += 1
        if result.truncated:
            warnings.append(
                f"{uploaded_file.name}：内容较长，已按系统限制截取部分页面或文字。"
            )

    if not chunks:
        st.error("没有可建立的证据片段。请检查 PDF 是否为可复制文本文件。")
        for warning in warnings:
            st.warning(warning)
        return

    _set_evidence_index(
        chunks,
        document_count=successful_documents,
        source="本地上传（仅当前会话）",
        warnings=warnings,
        use_vector_store=False,
    )
    st.success(
        f"证据库已建立：{successful_documents} 篇文档，{len(chunks)} 个证据片段。"
    )


def render_data_analysis(request_navigation: Callable[[str], None]) -> None:
    """渲染 Excel/CSV 数据分析页面。"""

    st.markdown('<div class="section-label">DATA LAB</div>', unsafe_allow_html=True)
    st.title("Excel 数据分析")
    st.write("上传真实数据文件，查看数据规模、缺失值、数值描述统计和基础图表。")

    uploaded = st.file_uploader("选择 CSV 或 XLSX 文件", type=["csv", "xlsx"])
    if uploaded is None:
        st.caption(
            f"支持 UTF-8/GB18030 CSV 和 XLSX，单个文件上限 "
            f"{settings.max_upload_size_mb} MB。"
        )
        return

    file_bytes = uploaded.getvalue()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        st.error(f"文件超过 {settings.max_upload_size_mb} MB 限制。")
        return

    try:
        dataframe = load_dataframe(file_bytes, uploaded.name)
        result = analyze_dataframe(dataframe)
    except DataAnalysisError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"数据处理出现未预期错误：{exc}")
        return

    st.session_state["workbench_data_summary"] = {
        "file_name": uploaded.name,
        "rows": result.rows,
        "columns": result.columns,
        "numeric_columns": len(result.numeric_summary),
        "duplicate_rows": result.duplicate_rows,
        "missing_cells": int(result.dataframe.isna().sum().sum()),
    }
    st.caption(f"文件：{uploaded.name}")
    metrics = st.columns(4)
    metrics[0].metric("数据行数", result.rows)
    metrics[1].metric("数据列数", result.columns)
    metrics[2].metric("数值列数", len(result.numeric_summary))
    metrics[3].metric("完全重复行", result.duplicate_rows)

    preview_tab, stats_tab, missing_tab, chart_tab = st.tabs(
        ("数据预览", "描述统计", "缺失值", "基础图表")
    )
    with preview_tab:
        st.dataframe(result.dataframe.head(100), width="stretch")
        st.caption("最多预览前 100 行。")
        type_table = pd.DataFrame(
            {
                "字段": [str(column) for column in result.dataframe.columns],
                "数据类型": [str(dtype) for dtype in result.dataframe.dtypes],
            }
        )
        st.markdown("**字段类型**")
        st.dataframe(type_table, width="stretch", hide_index=True)

    with stats_tab:
        if result.numeric_summary.empty:
            st.info("未检测到数值列，因此没有均值和标准差结果。")
        else:
            st.dataframe(
                result.numeric_summary,
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "均值按非空数值计算；标准差采用样本标准差（ddof=1）。"
                "仅有一个有效值时，标准差无法计算。"
            )

    with missing_tab:
        if result.missing_summary.empty:
            st.success("当前数据未检测到缺失值。")
        else:
            st.dataframe(
                result.missing_summary,
                width="stretch",
                hide_index=True,
            )
            st.caption("缺失值只做统计展示，不会自动填补或删除。")

    with chart_tab:
        numeric_columns = result.numeric_summary["字段"].tolist()
        if not numeric_columns:
            st.info("没有可用于绘图的数值列。")
        else:
            selected_column = st.selectbox("选择数值字段", numeric_columns)
            try:
                figure = create_numeric_histogram(
                    result.dataframe, str(selected_column)
                )
                st.pyplot(figure, width="stretch")
            except DataAnalysisError as exc:
                st.error(str(exc))

    st.info(
        "本次真实数据摘要已同步到科研工作台，可在工作阶段和导出底稿中查看。",
        icon=":material/sync:",
    )
    if st.button(
        "返回科研工作台",
        key="data_analysis_back_to_workbench",
        icon=":material/space_dashboard:",
    ):
        request_navigation("科研工作台")


def render_system_status() -> None:
    """渲染系统与依赖状态页面。"""

    st.markdown('<div class="section-label">SYSTEM CHECK</div>', unsafe_allow_html=True)
    st.title("系统运行状态")
    st.write("这里只显示真实检查结果，不会把待开发能力标记为已完成。")

    st.subheader("模型接口")
    if settings.llm_configured:
        st.success(f"已配置模型：{settings.openai_model}")
        st.caption(f"接口地址：{settings.openai_base_url}")
    else:
        st.warning("未配置 OPENAI_API_KEY，科研问答使用模拟回答。")
        st.code(
            "Copy-Item .env.example .env\n"
            "# 编辑 .env 后填写 OPENAI_API_KEY，再重启应用",
            language="powershell",
        )

    st.subheader("依赖检查")
    dependencies = (
        ("streamlit", "streamlit"),
        ("pandas", "pandas"),
        ("PyMuPDF", "fitz"),
        ("matplotlib", "matplotlib"),
        ("python-dotenv", "dotenv"),
        ("openpyxl", "openpyxl"),
        ("pytest", "pytest"),
        ("chromadb", "chromadb"),
        ("sentence-transformers", "sentence_transformers"),
    )
    dependency_rows = []
    for package_name, import_name in dependencies:
        try:
            version = importlib.metadata.version(package_name)
            status = "已安装"
        except importlib.metadata.PackageNotFoundError:
            version = "-"
            status = "未安装"
        dependency_rows.append(
            {
                "依赖": package_name,
                "导入名": import_name,
                "版本": version,
                "状态": status,
            }
        )
    st.dataframe(
        pd.DataFrame(dependency_rows),
        width="stretch",
        hide_index=True,
    )

    st.subheader("项目目录")
    directory_rows = []
    for label, path in (
        ("数据目录", settings.data_dir),
        ("向量模型缓存", settings.embedding_cache_dir),
        ("知识库目录", settings.knowledge_base_dir),
        ("向量数据库", settings.knowledge_base_dir / "chroma"),
        ("输出目录", settings.outputs_dir),
    ):
        directory_rows.append(
            {
                "目录": label,
                "路径": str(path),
                "状态": "存在" if path.exists() else "缺失",
            }
        )
    st.dataframe(
        pd.DataFrame(directory_rows),
        width="stretch",
        hide_index=True,
    )

    st.subheader("本地语义向量库")
    vector_status = None
    bundle = None
    try:
        bundle = get_bundled_knowledge_base()
        vector_status = get_vector_store().status(
            expected_chunks=len(bundle.chunks)
        )
    except (KnowledgeBaseError, VectorStoreError) as exc:
        st.error(f"向量库状态读取失败：{exc}")

    if vector_status is not None:
        vector_metrics = st.columns(4)
        vector_metrics[0].metric(
            "向量片段",
            vector_status.stored_chunks,
        )
        vector_metrics[1].metric(
            "证据片段",
            vector_status.expected_chunks or 0,
        )
        vector_metrics[2].metric(
            "同步状态",
            "已同步" if vector_status.synchronized else "待同步",
        )
        vector_metrics[3].metric(
            "模型缓存",
            "已下载" if vector_status.model_cached else "待下载",
        )
        st.caption(
            f"模型：{vector_status.model_name}　|　"
            f"集合：{vector_status.collection_name}　|　"
            f"目录：{vector_status.path}"
        )
        if vector_status.synchronized:
            st.success(vector_status.message)
        else:
            st.warning(vector_status.message)
            if bundle is not None and st.button(
                "构建或更新本地向量库",
                key="system_build_vector_store",
                icon=":material/sync:",
            ):
                try:
                    with st.spinner(
                        "正在生成向量并同步 Chroma，首次运行可能需要数分钟..."
                    ):
                        sync_result = sync_bundled_vectors(
                            bundle,
                            base_dir=settings.knowledge_base_dir,
                            vector_store=get_vector_store(),
                        )
                except (VectorStoreError, ValueError, OSError) as exc:
                    st.error(f"向量库构建失败：{exc}")
                else:
                    st.session_state["vector_sync_notice"] = sync_result
                    st.rerun()

    vector_sync_notice = st.session_state.pop("vector_sync_notice", None)
    if isinstance(vector_sync_notice, VectorSyncResult):
        st.success(_format_vector_sync_result(vector_sync_notice))

    st.subheader("能力边界")
    stages = pd.DataFrame(
        [
            {"能力": "规则任务规划", "当前状态": "已实现"},
            {"能力": "PDF 文本提取", "当前状态": "已实现"},
            {
                "能力": "PDF 原文证据检索",
                "当前状态": "已实现（BM25 + 语义混合，可回退 BM25）",
            },
            {"能力": "项目内置文献库", "当前状态": "已实现（清单 + JSON 索引）"},
            {"能力": "自动文献导入与去重", "当前状态": "已实现"},
            {"能力": "向量索引增量同步", "当前状态": "已实现"},
            {"能力": "研究框架自动更新", "当前状态": "已实现（规则归类）"},
            {
                "能力": "OpenAlex/Crossref 元数据检索",
                "当前状态": "已实现（联网可用性取决于网络）",
            },
            {
                "能力": "Semantic Scholar 公开元数据检索",
                "当前状态": "已实现（匿名可用，可选密钥提升额度）",
            },
            {
                "能力": "Function Calling 检索与综述",
                "当前状态": "已实现（需要有效的大模型配置）",
            },
            {
                "能力": "arXiv 公开预印本回退",
                "当前状态": "已实现（Semantic Scholar 无结果或失败时）",
            },
            {
                "能力": "GB/T 7714 引用整理",
                "当前状态": "已实现（缺失字段不猜测，提交前需人工核验）",
            },
            {
                "能力": "用户 PDF 独立 Chroma RAG",
                "当前状态": "已实现（按内容哈希隔离，可回退 BM25）",
            },
            {
                "能力": "公开元数据与用户 PDF 隔离",
                "当前状态": "已实现（外部元数据不写入用户论文向量库）",
            },
            {
                "能力": "ResearchMind SQLite 论文库",
                "当前状态": "已实现（领域、论文、证据、题库和学习记录）",
            },
            {
                "能力": "ResearchMind 六个 Agent",
                "当前状态": "已实现（顺序工作流）",
            },
            {
                "能力": "学术分歧与知识体系",
                "当前状态": "已实现（本地证据规则分析）",
            },
            {
                "能力": "自动组卷与学习闭环",
                "当前状态": "已实现（客观题自动、主观题按细则）",
            },
            {
                "能力": "Zotero Web API",
                "当前状态": "代码已实现，真实账号同步待验证",
            },
            {"能力": "CSV/XLSX 基础分析", "当前状态": "已实现"},
            {
                "能力": "大模型兼容接口",
                "当前状态": "已实现；成功调用取决于密钥、模型和网络",
            },
            {
                "能力": "Chroma 持久化向量库",
                "当前状态": "已实现（项目文献库与用户论文库独立集合）",
            },
            {"能力": "LangGraph 编排", "当前状态": "未集成"},
        ]
    )
    st.dataframe(stages, width="stretch", hide_index=True)


def render_footer() -> None:
    st.markdown(
        '<div class="footer-note">体研智脑 Agent · 公开文献检索与本地证据 RAG · '
        "模拟内容均会明确标注 · 科研结论需以真实文献与数据核验为准</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    """应用入口。"""

    st.set_page_config(
        page_title=settings.app_name,
        layout="wide",
        initial_sidebar_state="auto",
    )
    inject_styles()
    page = render_sidebar()

    if page == "首页":
        render_home(request_navigation)
    elif page == "科研工作台":
        render_workbench(get_researchmind_workflow(), request_navigation)
    elif page == "智能科研问答":
        render_question_answer()
    elif page == "科研选题":
        render_topic_selection()
    elif page == "文献阅读":
        render_pdf_reader()
    elif page == "智能文献研究":
        render_literature_research()
    elif page == "文献证据问答":
        render_evidence_qa()
    elif page == "文献框架":
        render_literature_framework()
    elif page == "数据分析":
        render_data_analysis(request_navigation)
    elif page == "系统状态":
        render_system_status()
    elif page in RESEARCHMIND_PAGES:
        render_researchmind_page(page)

    render_footer()


if __name__ == "__main__":
    main()
