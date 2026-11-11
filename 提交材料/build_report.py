"""生成体研智脑 Agent 参赛技术报告的 Word 与 PDF 版本。

脚本只使用项目当前可以核验的实现事实。运行环境由 Codex 工作区依赖提供，
生成两份最终文件：
1. 体研智脑Agent_技术报告.docx
2. 体研智脑Agent_技术报告.pdf

同时生成两张架构图，供复现或后续修改报告时复用。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
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
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent
ASSET_DIR = OUTPUT_DIR / "assets"
DOCX_PATH = OUTPUT_DIR / "体研智脑Agent_技术报告.docx"
PDF_PATH = OUTPUT_DIR / "体研智脑Agent_技术报告.pdf"

FONT_SANS = Path(r"C:\Windows\Fonts\Deng.ttf")
FONT_SANS_BOLD = Path(r"C:\Windows\Fonts\Dengb.ttf")
FONT_SERIF = Path(r"C:\Windows\Fonts\NotoSerifSC-VF.ttf")
FONT_SANS_SC = Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf")

BLUE = "2457D6"
NAVY = "18324F"
PALE_BLUE = "EEF3FB"
PALE_GRAY = "F5F7FA"
MID_GRAY = "D9E0E8"
TEXT = "1C2635"
MUTED = "5C6979"


TITLE = "体研智脑 Agent 技术报告"
SUBTITLE = "面向体育科研全流程的可追溯智能助手设计与实现"
FULL_PROJECT_TITLE = "基于大语言模型与多智能体协作的体育科研全流程智能助手"

ABSTRACT = (
    "体育科研初学者常常需要在选题、文献检索、全文阅读、证据核验、数据分析和学习反馈之间反复切换。"
    "普通聊天机器人却容易把规划、检索和生成混在同一段文本中，导致来源难追踪、步骤难复核。"
    "本文设计并实现体研智脑 Agent，以 Streamlit 构建中文交互界面，以规则规划、Function Calling、"
    "PDF 页级切分、BM25 与 Chroma 混合检索、SQLite 学习记录和六阶段顺序工作流组织科研任务。"
    "系统将外部公开文献元数据与用户主动上传的本地 PDF 隔离，要求关键结论返回论文题名、文件名、"
    "页码和原文，并在证据不足时停止推断。学习模块支持六种题型、客观题自动评分、主观题评分细则、"
    "五维掌握度、错题本和下一轮针对性复习。当前版本包含 20 个功能页面、12 篇已登记论文和 1095 条"
    "页级证据；自动化测试 74 项全部通过，运行服务健康检查返回 ok。报告同时说明系统在 OCR、"
    "多用户权限、自动综述和 LangGraph 编排方面的边界，并给出可复现的部署、测试和验收路径。"
)

KEYWORDS = "体育科研；大语言模型；多智能体；检索增强生成；证据溯源；学习测评"


REFERENCES = [
    "[1] Python Software Foundation. Python 3.12 Documentation[EB/OL]. https://docs.python.org/3.12/, 2026-11-11.",
    "[2] Streamlit. Streamlit Documentation[EB/OL]. https://docs.streamlit.io/, 2026-11-11.",
    "[3] Chroma. Chroma Documentation[EB/OL]. https://docs.trychroma.com/, 2026-11-11.",
    "[4] Sentence Transformers. Pretrained Models[EB/OL]. https://www.sbert.net/, 2026-11-11.",
    "[5] Semantic Scholar. API Documentation[EB/OL]. https://api.semanticscholar.org/, 2026-11-11.",
    "[6] arXiv. API Documentation[EB/OL]. https://info.arxiv.org/help/api/index.html, 2026-11-11.",
    "[7] OpenAlex. API Documentation[EB/OL]. https://docs.openalex.org/, 2026-11-11.",
    "[8] Crossref. REST API[EB/OL]. https://api.crossref.org/, 2026-11-11.",
    "[9] Zotero. Web API v3[EB/OL]. https://www.zotero.org/support/dev/web_api/v3/start, 2026-11-11.",
    "[10] OpenAI. Chat Completions API[EB/OL]. https://platform.openai.com/docs/api-reference/chat, 2026-11-11.",
]


CONTENT: list[tuple[str, Any]] = [
    ("h1", "1 引言"),
    ("p", "体育教育专业学生和科研初学者在开始一项研究时，通常并不缺少一个可以聊天的界面，而是缺少一条能够被检查的工作链：研究问题是否被正确拆解，检索词是否覆盖研究对象与结局指标，文献是否真实存在，结论能否回到原文页码，数据分析是否只报告了实际上传的数据，学习反馈是否依据明确评分标准。大语言模型可以生成连贯文本，但如果缺少任务边界、工具调用和证据约束，也容易产生来源不明的结论。"),
    ("h2", "1.1 研究背景"),
    ("p", "大学生创新竞赛中的 AI 应用常从聊天问答起步，但要形成可演示、可复核的科研助手，需要同时解决三类问题。第一，科研任务具有阶段性和依赖关系，选题、检索、阅读、分析和报告不能被视为同一种问句。第二，学术材料具有版权与真实性边界，公开元数据、开放获取全文和用户自行上传的 PDF 需要分开管理。第三，模型输出具有不确定性，必须通过结构化证据、失败回退和明确提示降低幻觉风险。"),
    ("p", "本项目选择体育科研作为垂直场景。该场景同时包含问卷与体测数据、横断面与干预研究、教育场景观察和文献证据整合，适合验证一个科研助手能否在真实任务中保持可追溯性。项目不把自动写出一篇论文作为目标，而是把选题、检索、阅读、证据组织、数据分析和学习反馈连接为可检查的流程。"),
    ("h2", "1.2 问题定义"),
    ("p", "本文将待解决的问题定义为：给定一个体育科研任务，系统应识别任务意图，生成有限且可执行的步骤；在需要时调用公开学术接口或本地文献工具；对返回结果执行去重、字段校验和证据定位；在输出中区分事实、模拟内容和证据不足；将用户答题记录持久化，并据此更新掌握度和下一步学习建议。"),
    ("p", "围绕该定义，系统设置三条底线。真实资料必须可以核验；模拟内容必须显式标注；没有执行的步骤不能显示为已完成。上述底线同时约束界面状态、数据结构和测试断言，避免“页面看起来完整”替代“功能实际可用”。"),
    ("h2", "1.3 主要工作"),
    ("bullets", [
        "构建 20 个中文功能页面，覆盖科研工作台、智能问答、选题、PDF 阅读、公开文献检索、证据问答、数据分析、学术领域管理、论文库、观点对比、知识图谱和学习测评。",
        "实现 Semantic Scholar Function Calling 检索流程：模型先整理研究对象、核心变量、结局指标和同义词。随后调用公开检索工具，最后依据真实元数据生成研究现状；未配置模型时明确降级为规则整理。",
        "建立 PDF 页级证据链：文件哈希校验、文本提取、按页切分、BM25 与 Chroma 混合检索、EvidenceRef 来源对象和页面展示。",
        "实现 ResearchMind 六阶段顺序工作流和六种题型学习闭环，保存题目、答案、评分细则、答题记录、掌握度和错题本。",
        "以 74 项自动化测试、语法检查和服务健康检查形成可复现证据，并在报告中明确当前能力边界。",
    ]),
    ("h1", "2 需求分析与设计目标"),
    ("h2", "2.1 用户与任务场景"),
    ("p", "目标用户包括体育教育专业学生、体育教师和科研初学者。典型任务不是单轮知识问答，而是从一个较宽泛的方向逐步收敛到可执行研究问题，再查找相关研究、核对方法、读取结果并组织自己的研究设计。用户需要看到系统做了什么、依据来自哪里、哪些步骤尚未执行，而不是只得到最终段落。"),
    ("p", "因此，系统以“工作底稿”而不是“自动论文”为产品中心。科研工作台输出研究简报、检索策略、设计模板、真实证据命中和阶段状态；科研报告始终显示为待撰写。文献阅读、观点比较和知识综合提供结构化线索，最终判断仍由研究者完成。"),
    ("h2", "2.2 功能需求"),
    ("table", {
        "caption": "表 1  第一阶段功能需求与当前实现",
        "headers": ["功能域", "需求", "当前实现"],
        "widths_cm": [3.1, 6.8, 6.2],
        "rows": [
            ["科研任务规划", "把宽泛问题拆成步骤，生成检索式和研究设计模板", "规则规划器、中文检索式草稿、工作台 JSON 底稿"],
            ["公开文献检索", "获取真实题名、作者、年份、摘要、DOI 和开放 PDF 链接", "Semantic Scholar Function Calling；失败时回退 arXiv"],
            ["本地文献阅读", "读取用户 PDF，保留页码和原文", "PyMuPDF 提取、SHA-256 去重、页级片段和 EvidenceRef"],
            ["证据问答", "基于检索证据回答，不能编造来源", "BM25/语义/混合检索、证据约束提示词、无证据拒绝回答"],
            ["学术分析", "比较观点、识别分歧、综合知识与研究空白", "六阶段顺序 Agent，输出关联论文与页码证据"],
            ["学习测评", "六类题型、自动评分、错题本、掌握度和下一轮推荐", "客观题规则评分、主观题细则评分、SQLite 学习闭环"],
            ["数据分析", "CSV/XLSX 预览、缺失值、均值和标准差、基础图表", "Pandas、openpyxl 与 Matplotlib 描述性分析"],
            ["系统状态", "说明文献库、向量库、数据库和外部服务状态", "本地状态页、明确的降级提示和健康检查"],
        ],
    }),
    ("h2", "2.3 非功能需求"),
    ("p", "系统面向编程基础较弱的学生和现场演示环境，因而优先选择容易解释和调试的技术。前端使用 Streamlit；文献清单和证据片段使用可阅读的 JSON；学习数据使用 SQLite；大模型通过 OpenAI 兼容接口接入；向量模型首次使用时才下载。任一外部服务不可用时，系统应保留本地能力，而不是让整个页面崩溃。"),
    ("p", "安全方面，密钥只从环境变量或 .env 读取，不写入源码和报告；用户 PDF 与公开检索元数据分目录、分集合保存；上传文件设置大小、页数和字符数限制；错误信息转换为界面可读提示。学术伦理方面，不爬取知网，不绕过付费墙，不生成不存在的 DOI、作者、页码或实验数据。"),
    ("h2", "2.4 设计原则"),
    ("bullets", [
        "真实性优先：题名、作者、期刊、DOI、页码和统计结果必须来自登记清单、公开接口、真实 PDF 或用户上传数据。",
        "证据可定位：重要结论和题目尽量关联 chunk_id、论文 ID、文件名、页码和原文，界面能够回到原文核对。",
        "模拟显式标注：没有模型密钥时，问答、评分和综述使用本地模板时必须写明模拟属性。",
        "渐进增强：规则规划、BM25、SQLite 和数据分析可离线运行；模型、Chroma、Semantic Scholar 与 Zotero 属于增强能力。",
        "相关性不等于因果：系统不会把横断面相关结果自动解释为因果，也不会把单篇论文表述为领域共识。",
    ]),
    ("h1", "3 系统总体架构"),
    ("h2", "3.1 分层架构"),
    ("p", "系统采用界面层、编排层、Agent 与领域服务层、检索与工具层、持久化层五层结构。界面层只负责输入、状态和展示；编排层决定任务顺序；Agent 层把阅读、争议分析、知识综合、组卷和辅导拆为独立职责；工具层封装外部 API、PDF 和检索；持久化层保存可复核数据。依赖方向保持从界面到编排、再到领域服务和工具，减少页面直接拼接底层逻辑。"),
    ("figure", {
        "path": ASSET_DIR / "architecture.png",
        "caption": "图 1  体研智脑 Agent 分层架构",
        "width_cm": 16.2,
    }),
    ("p", "这种分层并不追求复杂的自主协作。当前 ResearchMind 是顺序工作流，由程序固定调用 LiteratureCollector、PaperReader、AcademicDebateAnalyzer、KnowledgeSynthesizer、LearningAssessmentAgent 和 PersonalTutorAgent，便于初学者按阶段调试。LangGraph 被列为后续演进方向，而不是当前已实现能力。"),
    ("h2", "3.2 关键数据流"),
    ("p", "科研任务首先进入规划器，形成步骤、检索式和研究设计模板。公开检索只返回外部元数据，不自动写入用户论文库；用户上传 PDF 则经过文件校验、哈希、提取和页级切分，形成独立证据集合。查询时，系统先取候选片段，再按模式执行 BM25、语义或混合检索，最后把证据交给本地摘录或受约束的模型回答。学习闭环把题目、用户答案、得分和知识点写入 SQLite，再计算掌握度和下一轮题目。"),
    ("figure", {
        "path": ASSET_DIR / "workflow.png",
        "caption": "图 2  从任务意图到学习反馈的端到端数据流",
        "width_cm": 16.2,
    }),
    ("h2", "3.3 模块职责"),
    ("table", {
        "caption": "表 2  主要代码模块与职责边界",
        "headers": ["模块", "职责", "关键约束"],
        "widths_cm": [4.4, 7.1, 4.6],
        "rows": [
            ["app.py / researchmind/ui.py", "页面、导航、上传、状态和下载", "不直接伪造分析结果；状态由真实对象驱动"],
            ["agents/planner.py", "意图分类、研究步骤、选题模板和中文检索式", "输出是规划和建议，不是检索结果"],
            ["researchmind/workflow.py", "六阶段顺序分析、学习闭环和验收流程", "记录每个阶段的结果，不跳步宣称完成"],
            ["researchmind/agents.py", "论文阅读、争议分析、知识综合、组卷和辅导", "观点和题目必须关联证据；证据不足保持为空"],
            ["services/rag_service.py", "BM25、语义/混合融合和证据问答", "证据为空时拒绝回答；语义失败回退 BM25"],
            ["services/user_paper_rag.py", "用户 PDF 哈希隔离、分页证据和独立 RAG", "只读取用户主动上传的论文"],
            ["tools/semantic_scholar.py", "公开元数据检索、字段映射和去重", "不下载无开放许可的全文"],
            ["researchmind/storage.py", "SQLite 表、事务、答题和掌握度", "失败回滚；数据库异常转换为可读错误"],
        ],
    }),
    ("h2", "3.4 数据存储与检索"),
    ("p", "项目内置文献库登记 12 篇论文，证据片段索引包含 1095 条按页切分记录。JSON 是文献清单和原文证据的事实源；Chroma 是可重建的语义索引；SQLite 保存领域、论文、分析、题目、答题和掌握度；用户 PDF 位于 data/user_papers，并按内容 SHA-256 隔离。即使 Chroma 的 HNSW 派生索引损坏，原始 PDF、清单和证据 JSON 也不会被覆盖。"),
    ("p", "BM25 负责中英文关键词召回，使用 k1=1.5、b=0.75。语义检索默认使用 intfloat/multilingual-e5-small，文献片段加 passage: 前缀，查询加 query: 前缀。混合模式把候选 BM25 分数归一化后与语义相似度融合，权重分别为 0.45 和 0.55；当 Chroma、嵌入模型或语义查询失败时，系统自动回退到关键词检索。"),
    ("h1", "4 意图控制与 Prompt 策略"),
    ("h2", "4.1 上下文分层"),
    ("p", "系统没有把所有信息拼成一个超长提示词，而是把上下文分为五层。不同层承担不同责任，既减少提示词冲突，也便于定位错误来源。"),
    ("table", {
        "caption": "表 3  上下文分层与控制目标",
        "headers": ["上下文层", "主要内容", "控制目标"],
        "widths_cm": [3.2, 7.0, 5.9],
        "rows": [
            ["系统约束", "真实来源、证据优先、模拟标注、失败时停止推断", "防止角色漂移和无依据扩写"],
            ["任务上下文", "研究问题、对象、变量、方法和用户选定文献", "把生成限制在当前科研任务内"],
            ["工具上下文", "Semantic Scholar 工具定义、参数 schema、返回条数", "让模型只请求已实现的工具和合法参数"],
            ["证据上下文", "chunk_id、论文题名、文件名、页码和原文摘录", "把结论绑定到可核验片段"],
            ["输出契约", "题名、作者、年份、DOI、引用、评分细则与反馈字段", "把自由文本转换为可检查和可保存的结构"],
        ],
    }),
    ("h2", "4.2 意图识别与任务规划"),
    ("p", "规划器先对任务关键词做规则分类，区分选题、文献检索、研究设计、数据分析、论文写作和学习测评等意图，再生成有限步骤。研究问题会被补充研究对象、核心变量、结局指标和常用同义词，形成中文数据库检索式草稿与英文检索结构提示。规则规划可以离线运行，便于在演示现场解释，也避免把模型调用失败误认为应用失败。"),
    ("p", "当任务需要外部文献时，模型不直接回答“有哪些论文”，而是先优化检索关键词，再发起工具调用。工具执行后，真实结果通过 role=tool 消息回传，模型第二次生成综述和研究现状。未配置密钥时，系统仍可执行公开检索，但使用规则整理并标注“非大模型综述”，不会虚构一次并不存在的 Function Calling。"),
    ("h2", "4.3 Function Calling 结构"),
    ("p", "Semantic Scholar 工具接收 query 和 limit 两个参数，返回题名、作者、年份、摘要、DOI、期刊或会议、来源页面和开放获取 PDF 链接。工具层负责参数校验、超时、限流和按 DOI/题名去重。模型只负责提出检索参数和基于返回元数据组织文字，不能修改工具字段，也不能补造未返回的作者、年份或 DOI。"),
    ("p", "GB/T 7714 风格引用由格式化函数读取现有字段生成；缺失字段会省略或明确标为缺失，而不是猜测。Semantic Scholar 无结果或不可用时，系统可回退到 arXiv 官方 Atom API，并在界面标记来源。公开检索结果不会写入用户上传 PDF 的 Chroma 集合，从数据层保持版权与用途边界。"),
    ("h2", "4.4 证据约束与防幻觉"),
    ("p", "证据问答的提示词要求：只能依据提供的证据片段；关键结论标注证据编号；不得编造作者、题名、年份、期刊、数据或统计结果；证据不足或相互矛盾时必须明确说明；区分文献原文与方法学解释。检索结果为空时，程序直接返回“当前文献不足以支持结论”，不调用模型强行生成。"),
    ("p", "学术争议采用结构化对象保存 topic、viewpoints、supporting_papers、evidence、page_numbers、possible_reasons、limitations、evidence_strength 和 research_gaps。每个观点至少关联一篇论文和一条页码证据；证据不足时保留空值并显示“证据不足”，而不是用常识补齐。系统不会把相关性自动升级为因果关系，比较研究时同时提示样本、设计、指标和统计方法的差异。"),
    ("h2", "4.5 失败回退与用户提示"),
    ("bullets", [
        "没有 API Key：常规问答使用明确标注的本地模板；证据问答展示原文摘录；主观题使用本地评分细则。",
        "模型调用失败：允许模拟时回退模板，不允许时返回可读失败原因，不把错误伪装成成功回答。",
        "Chroma 或嵌入模型失败：自动使用 BM25；HNSW 索引损坏时先隔离派生目录，再从证据 JSON 重建。",
        "Semantic Scholar 无结果：尝试 arXiv；仍无结果时给出检索建议，不生成不存在的候选论文。",
        "PDF 无文本或损坏：拒绝建立空索引，提示文件可能损坏或为扫描版。",
    ]),
    ("h1", "5 人机协同开发与架构约束"),
    ("h2", "5.1 增量开发流程"),
    ("p", "项目开发没有把“完成整个系统”作为一个一次性提示，而是按可观察功能拆分任务：先建立项目骨架和基础测试，再加入 PDF、数据分析和问答；随后增加 Semantic Scholar、用户 PDF RAG、ResearchMind 六阶段流程和学习闭环；最后修复界面样式、向量索引、选项展示和评分证据问题。每次修改都要求说明影响模块、保持既有接口并补充可回归的测试。"),
    ("p", "这一流程的重点不是让 AI 多写代码，而是把人的判断放在四个节点：确定本轮任务边界，检查实现是否符合现有架构，要求对失败路径给出回退，使用测试证明修改没有破坏旧功能。报告中不声称不存在的 AI 对话记录；截图、Prompt 快照和录屏应按竞赛要求另行从真实会话中导出。"),
    ("h2", "5.2 对 AI 生成代码的约束"),
    ("bullets", [
        "模块边界：页面、编排、Agent、工具和存储分离，禁止把一个完整功能堆进 app.py。",
        "数据契约：论文、证据、题目、评分和掌握度使用类型化模型，界面不能自行拼造来源。",
        "真实资料：外部字段必须来自接口响应；本地结论必须来自 PDF 片段或用户数据。",
        "错误处理：外部接口、文件读取、数据库和向量库失败时返回可读提示，并保留本地能力。",
        "测试闭环：新增核心行为至少对应一个测试；修复缺陷时增加回归断言。",
        "文档一致：README、技术文档和报告不得把规划能力写成已实现能力。",
    ]),
    ("h2", "5.3 两个缺陷修复案例"),
    ("p", "案例一为 Chroma HNSW 索引损坏。界面曾出现 Error loading hnsw index，直接删除整个知识库会扩大损失。修复思路是只处理派生索引：先把损坏目录隔离为带时间戳的备份，再从原始 PDF 和 evidence_chunks.json 重新构建集合。测试同时验证超过 HNSW 阈值的重新打开和损坏索引重建，确保原始证据不被修改。"),
    ("p", "案例二为学习测评选择题错位、选项含网络地址或英文残片。修复时不再用字符串拆分选项，而是按选项索引映射用户答案；生成选项时使用证据标签，并过滤 URL、DOI 链接和不符合要求的外文展示。提交后界面同时显示用户答案、标准答案、得分、评分依据和证据位置，测试对中文选项、无网址和无错位进行断言。"),
    ("h1", "6 核心功能实现"),
    ("h2", "6.1 科研工作台与选题"),
    ("p", "科研工作台把研究问题、研究对象和研究方法组合成研究简报。它先给出任务类型、步骤、中文检索式和研究设计检查项，再在当前领域证据中执行检索。阶段状态包括规划、检索、研究设计、数据分析、证据核验和科研报告；没有证据时显示待检索或证据不足，没有上传数据时显示待操作，科研报告固定为待撰写。"),
    ("p", "选题模块根据方向生成研究背景、研究问题、研究对象和研究价值模板。这些内容用于启发选题，不是已完成的研究结论。模型密钥为空时，界面明确标注模拟建议；检索策略只生成表达式和数据库建议，不声称已经返回论文。"),
    ("h2", "6.2 公开文献检索、综述与引用"),
    ("p", "智能文献研究页面分为公开文献检索和个人论文 RAG 两个区域。公开区域使用 Semantic Scholar 与 arXiv，整理真实元数据、研究现状和 GB/T 7714 风格引用；个人区域只读取用户主动上传的 PDF，按 SHA-256 建立独立目录与 Chroma 集合。两套数据在存储、检索和界面提示上保持隔离，避免把外部摘要误写进用户论文库。"),
    ("h2", "6.3 PDF 解析与本地 RAG"),
    ("p", "PDF 导入依次执行扩展名与大小校验、文件内容识别、SHA-256 去重、PyMuPDF 文本提取、书目信息识别、安全文件名生成和清单写入。文本按页切分，默认片段目标 700 字符、重叠 120 字符，片段不跨页，因此页码可保持确定。空文本、损坏文件、疑似扫描版和超过页数或字符上限的文件会被拒绝。"),
    ("p", "用户论文 RAG 支持 BM25 与 Chroma 语义检索。回答展示文件名、页码和原文证据；没有证据时拒绝作答；向量失败时回退 BM25。公开检索只处理外部元数据，用户 PDF 只在本地处理，这种分工既符合开放学术接口的使用方式，也降低版权风险。"),
    ("h2", "6.4 ResearchMind 六阶段顺序工作流"),
    ("table", {
        "caption": "表 4  六个 Agent 的职责与输出约束",
        "headers": ["Agent", "主要职责", "输出约束"],
        "widths_cm": [4.1, 6.8, 5.2],
        "rows": [
            ["LiteratureCollector", "公开检索、去重、元数据和开放 PDF 整理", "只下载明确开放获取的 PDF；不爬取知网"],
            ["PaperReader", "提取目的、对象、方法、结果、结论和局限", "每个字段关联页码证据；不足时保持空值"],
            ["AcademicDebateAnalyzer", "识别相同观点、相反观点和结论差异", "观点必须关联论文和原文；区分相关与因果"],
            ["KnowledgeSynthesizer", "概念体系、理论框架、共识、分歧、局限、空白和创新方向", "单篇论文不升级为共识；记录证据数量"],
            ["LearningAssessmentAgent", "生成单选、多选、判断、简答、论述和文献对比题", "每题含答案、细则、标签、难度、论文和证据位置"],
            ["PersonalTutorAgent", "批改、错题分析、薄弱点、资料推荐和下一轮组卷", "反馈引用评分细则；按历史表现调整题型"],
        ],
    }),
    ("h2", "6.5 学习测评与掌握度"),
    ("p", "系统支持六种题型。单选、多选和判断题使用客观题自动评分；简答、论述和文献对比题优先使用模型按细则评分，模型不可用时使用本地关键词细则，并标记“本地评分细则（非大模型）”。每次提交保存用户答案、标准答案、得分、满分、反馈、评分模式和知识点标签。"),
    ("p", "五维掌握度初始值为概念记忆 15%、文献理解 20%、对比分析 25%、批判性思维 20%、学术应用 20%。每次学习循环使用 new_score = old_score × 0.72 + score_percentage × 0.28 平滑更新，并限制在 0 到 100。得分低于满分 60% 的记录进入错题本；下一轮根据低于 60 分的维度选择不同题型，形成可解释的复习闭环。"),
    ("h2", "6.6 数据分析与中文界面"),
    ("p", "数据分析支持 CSV 和 XLSX，显示数据预览、行列数、字段类型、缺失值、数值列均值和标准差，并可选择数值列生成直方图。统计结果只来自用户本次上传的数据，不生成模拟实验数据。界面支持日间白色与夜间深色主题，侧边导航采用精简一级功能入口，每个页面提供与当前页面上下文相关的智能问答辅助入口。"),
    ("h1", "7 可复现性论证"),
    ("h2", "7.1 验证环境"),
    ("p", "当前验证环境为 Windows，Python 3.12.16、Streamlit 1.66.1、Pandas 2.3.3、PyMuPDF 1.28.2、ChromaDB 1.5.9、Matplotlib 3.11.2。项目依赖在 requirements.txt 中声明，敏感配置通过环境变量模板提供。以下结果来自本次报告生成前的实际执行，不使用预设截图代替验证。"),
    ("table", {
        "caption": "表 5  当前可复现性验证结果",
        "headers": ["验证项", "执行方式", "结果"],
        "widths_cm": [4.2, 8.0, 3.9],
        "rows": [
            ["自动化测试", "python -m pytest -q", "74 passed，约 15 秒"],
            ["测试收集", "python -m pytest --collect-only -q", "74 tests collected"],
            ["语法检查", "python -m compileall -q ...", "COMPILE_OK"],
            ["文献库", "读取 manifest.json 与 evidence_chunks.json", "12 篇论文，1095 条片段"],
            ["服务健康", "GET /_stcore/health", "HTTP 200，响应 ok"],
            ["页面加载", "测试覆盖核心与 ResearchMind 页面", "全部页面无异常加载"],
        ],
    }),
    ("h2", "7.2 十分钟复现路径"),
    ("p", "复现时先进入项目目录，创建 Python 3.11 或 3.12 虚拟环境并安装 requirements.txt；复制项目根目录的环境变量模板，若只演示本地检索可保持密钥为空；再执行 streamlit run app.py。首次语义检索会下载嵌入模型，离线或下载失败时系统回退 BM25。随后运行 pytest -q 和健康检查，确认应用与文献库可用。"),
    ("p", "演示顺序建议为：首页说明项目定位；科研工作台输入一个真实研究问题并查看流程规划；文献证据问答展示题名、文件名和页码；智能文献研究展示公开检索入口与用户 PDF 隔离说明；学习测评展示中文选项、标准答案和评分细则；系统状态展示文献库与索引状态。负责人应提前准备一个有权使用的 PDF 和一份脱敏 CSV/XLSX，避免现场依赖网络。"),
    ("h2", "7.3 证据链与数据一致性"),
    ("p", "证据索引加载时校验清单哈希、片段数量、片段 ID 唯一性、文件名是否登记和页码范围。九维研究框架记录清单与索引哈希，数据变化或文件损坏时自动重建。SQLite 使用外键、WAL 和短连接事务；文献同步签名避免每次启动重复写入。外部服务不可用不会修改本地证据事实源。"),
    ("h1", "8 安全、版权与伦理边界"),
    ("p", "项目把外部公开元数据与用户上传全文分为两套数据路径。公开检索只使用 Semantic Scholar 与 arXiv 官方接口，不爬取知网，不绕过付费墙；只有明确标注开放获取的链接才允许下载。用户 PDF 由用户主动提供，保存在本地隔离目录，并用独立集合检索。报告和界面不展示真实 API Key，示例只使用占位符。"),
    ("p", "系统不生成不存在的论文、DOI、作者、引用、页码或实验数据。模拟问答、规则综述和本地评分必须标注其性质。相关性研究不能被自动解释为因果，单篇论文不能自动代表共识，论文阅读和争议分析仍需要研究者核对全文。上述边界不是产品缺陷，而是科研辅助工具可信使用的前提。"),
    ("h1", "9 局限性与后续工作"),
    ("p", "当前版本适合本机单用户演示，不是生产级多租户平台。主要限制包括：扫描版 PDF 尚无 OCR；复杂跨栏、表格和公式可能提取不完整；论文阅读和知识综合主要依赖本地规则与检索，不等同于系统综述；六阶段工作流是顺序调用，不是 LangGraph 图执行或自主协商；SQLite 不适合高并发写入；没有账号、权限和审计日志。"),
    ("p", "后续工作按风险和价值排序：第一，增加 OCR、跨页段落合并和表格提取，同时保持页码来源；第二，加入 DOI 回查、引用跳转和文献更新监控；第三，使用真实答题数据校准掌握度并增加评测集；第四，在保留现有接口的前提下评估 LangGraph、重排序和查询改写；第五，增加认证、租户隔离、上传保留策略和结构化日志。任何新增能力都应继续满足真实来源、模拟标注和未执行不虚报三条底线。"),
    ("h1", "10 结论"),
    ("p", "体研智脑 Agent 将体育科研中的任务规划、公开检索、PDF 阅读、页级证据、学术分析、数据分析和学习测评组织为可检查的本地流程。系统的主要价值不在于替用户写出一篇论文，而在于让每一步都能说明依据、回退和边界。当前实现已经具备可运行界面、真实文献库、混合检索、六阶段学习闭环和自动化测试，同时明确保留网络、OCR、自动综述和多用户部署方面的限制。"),
    ("p", "对参赛项目而言，技术报告、代码仓库、Prompt 演进记录和演示视频应共同证明能力不是静态页面。提交前仍需补齐公开仓库、细粒度 Git 历史、至少三组真实 Prompt 链和 5 至 8 分钟演示视频；本报告只覆盖技术实现与可复现性部分，不代替其他强制材料。"),
    ("h1", "参考资料"),
]


def _set_run_font(run, east_asia: str, size: float | None = None, bold: bool = False) -> None:
    """设置 Word 中西文字体，避免中文回退到不一致的默认字体。"""
    run.font.name = east_asia
    run._element.rPr.rFonts.set(qn("w:eastAsia"), east_asia)
    run.font.color.rgb = RGBColor(0, 0, 0)
    if size is not None:
        run.font.size = Pt(size)
    run.bold = bold


def _set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def _set_cell_margins(cell, top: int = 90, start: int = 100, bottom: int = 90, end: int = 100) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def _set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def _set_cell_borders(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), "D9D9D9")
        borders.append(element)


def _add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("第 ")
    _set_run_font(run, "等线", 8.5)
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = " PAGE "
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char1)
    run._r.append(instr_text)
    run._r.append(fld_char2)
    tail = paragraph.add_run(" 页")
    _set_run_font(tail, "等线", 8.5)


def _set_docx_styles(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Noto Serif SC"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Noto Serif SC")
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.line_spacing = 1.45
    normal.paragraph_format.space_after = Pt(3)

    for name, size, before, after in (
        ("Title", 22, 0, 12),
        ("Subtitle", 13, 0, 10),
        ("Heading 1", 14, 12, 6),
        ("Heading 2", 11.5, 8, 4),
    ):
        style = doc.styles[name]
        style.font.name = "Noto Sans SC"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Noto Sans SC")
        style.font.size = Pt(size)
        style.font.bold = name != "Subtitle"
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = name.startswith("Heading")


def _add_docx_paragraph(doc: Document, text: str, *, indent: bool = True) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    paragraph.paragraph_format.line_spacing = 1.45
    paragraph.paragraph_format.space_after = Pt(3)
    if indent:
        paragraph.paragraph_format.first_line_indent = Pt(21)
    run = paragraph.add_run(text)
    _set_run_font(run, "Noto Serif SC", 10.5)


def _add_docx_table(doc: Document, spec: dict[str, Any], table_number: int) -> None:
    caption = doc.add_paragraph()
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.keep_with_next = True
    caption.paragraph_format.space_before = Pt(5)
    caption.paragraph_format.space_after = Pt(3)
    run = caption.add_run(spec["caption"])
    _set_run_font(run, "Noto Sans SC", 9.2, bold=True)

    headers = spec["headers"]
    rows = spec["rows"]
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.allow_autofit = False
    _set_cell_borders(table)
    _set_repeat_table_header(table.rows[0])

    for idx, header in enumerate(headers):
        cell = table.rows[0].cells[idx]
        cell.width = Cm(spec["widths_cm"][idx])
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        _set_cell_shading(cell, NAVY)
        _set_cell_margins(cell)
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(0)
        run = paragraph.add_run(header)
        _set_run_font(run, "Noto Sans SC", 8.8, bold=True)
        run.font.color.rgb = RGBColor(255, 255, 255)

    for row_index, values in enumerate(rows):
        cells = table.add_row().cells
        for col_index, value in enumerate(values):
            cell = cells[col_index]
            cell.width = Cm(spec["widths_cm"][col_index])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            _set_cell_margins(cell)
            if row_index % 2 == 1:
                _set_cell_shading(cell, PALE_GRAY)
            paragraph = cell.paragraphs[0]
            paragraph.alignment = TA_LEFT if col_index > 0 else WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.line_spacing = 1.2
            paragraph.paragraph_format.space_after = Pt(0)
            run = paragraph.add_run(str(value))
            _set_run_font(run, "Noto Serif SC", 8.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def _add_docx_figure(doc: Document, spec: dict[str, Any]) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(5)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run()
    run.add_picture(str(spec["path"]), width=Cm(spec["width_cm"]))

    caption = doc.add_paragraph()
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.space_after = Pt(7)
    caption.paragraph_format.keep_with_next = True
    run = caption.add_run(spec["caption"])
    _set_run_font(run, "Noto Sans SC", 9.0)


def build_docx() -> None:
    """构建 Word 版技术报告。"""
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.1)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.25)
    section.right_margin = Cm(2.25)
    section.header_distance = Cm(1.0)
    section.footer_distance = Cm(1.0)
    _set_docx_styles(doc)

    doc.core_properties.title = TITLE
    doc.core_properties.subject = FULL_PROJECT_TITLE
    doc.core_properties.author = "体研智脑 Agent"
    doc.core_properties.comments = "大学生 AI 大模型创新应用挑战赛技术报告"

    footer = section.footer.paragraphs[0]
    _add_page_number(footer)

    # 封面
    for _ in range(4):
        doc.add_paragraph()
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title.add_run(TITLE)
    _set_run_font(title_run, "Noto Sans SC", 22, bold=True)

    subtitle = doc.add_paragraph(style="Subtitle")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle_run = subtitle.add_run(SUBTITLE)
    _set_run_font(subtitle_run, "Noto Sans SC", 13, bold=False)

    project = doc.add_paragraph()
    project.alignment = WD_ALIGN_PARAGRAPH.CENTER
    project.paragraph_format.space_before = Pt(8)
    project_run = project.add_run(FULL_PROJECT_TITLE)
    _set_run_font(project_run, "Noto Serif SC", 10.5)

    for _ in range(9):
        doc.add_paragraph()

    metadata = doc.add_table(rows=4, cols=2)
    metadata.alignment = WD_TABLE_ALIGNMENT.CENTER
    metadata.autofit = False
    _set_cell_borders(metadata)
    meta_rows = [
        ("文档类型", "参赛技术报告"),
        ("项目版本", "V1.0"),
        ("验证环境", "Python 3.12.16 / Streamlit 1.66.1"),
        ("完成日期", "2026 年 11 月 11 日"),
    ]
    for row_idx, (label, value) in enumerate(meta_rows):
        for col_idx, text in enumerate((label, value)):
            cell = metadata.cell(row_idx, col_idx)
            cell.width = Cm(3.4 if col_idx == 0 else 8.0)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            _set_cell_margins(cell, 110, 130, 110, 130)
            if col_idx == 0:
                _set_cell_shading(cell, PALE_BLUE)
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx == 0 else WD_ALIGN_PARAGRAPH.LEFT
            paragraph.paragraph_format.space_after = Pt(0)
            run = paragraph.add_run(text)
            _set_run_font(run, "Noto Sans SC" if col_idx == 0 else "Noto Serif SC", 9.2, bold=col_idx == 0)

    doc.add_page_break()

    # 摘要
    abstract_heading = doc.add_paragraph(style="Heading 1")
    abstract_heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    abstract_run = abstract_heading.add_run("摘要")
    _set_run_font(abstract_run, "Noto Sans SC", 14, bold=True)
    _add_docx_paragraph(doc, ABSTRACT)

    keywords = doc.add_paragraph()
    keywords.paragraph_format.space_before = Pt(5)
    keywords.paragraph_format.space_after = Pt(4)
    key_run = keywords.add_run("关键词：")
    _set_run_font(key_run, "Noto Sans SC", 10.2, bold=True)
    key_value = keywords.add_run(KEYWORDS)
    _set_run_font(key_value, "Noto Serif SC", 10.2)

    table_number = 0
    for kind, payload in CONTENT:
        if kind == "h1":
            if payload == "参考资料":
                doc.add_page_break()
            heading = doc.add_paragraph(style="Heading 1")
            run = heading.add_run(payload)
            _set_run_font(run, "Noto Sans SC", 14, bold=True)
        elif kind == "h2":
            heading = doc.add_paragraph(style="Heading 2")
            run = heading.add_run(payload)
            _set_run_font(run, "Noto Sans SC", 11.5, bold=True)
        elif kind == "p":
            _add_docx_paragraph(doc, payload)
        elif kind == "bullets":
            for item in payload:
                paragraph = doc.add_paragraph(style="List Bullet")
                paragraph.paragraph_format.left_indent = Pt(18)
                paragraph.paragraph_format.first_line_indent = Pt(-12)
                paragraph.paragraph_format.space_after = Pt(2)
                paragraph.paragraph_format.line_spacing = 1.35
                run = paragraph.add_run(item)
                _set_run_font(run, "Noto Serif SC", 10.2)
        elif kind == "table":
            table_number += 1
            _add_docx_table(doc, payload, table_number)
        elif kind == "figure":
            _add_docx_figure(doc, payload)

    for reference in REFERENCES:
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        paragraph.paragraph_format.left_indent = Pt(18)
        paragraph.paragraph_format.first_line_indent = Pt(-18)
        paragraph.paragraph_format.line_spacing = 1.25
        paragraph.paragraph_format.space_after = Pt(2)
        run = paragraph.add_run(reference)
        _set_run_font(run, "Noto Serif SC", 9.2)

    doc.save(DOCX_PATH)


def _draw_box(ax, x, y, w, h, text, *, fill="#FFFFFF", edge=BLUE, font_size=10, bold=False):
    """绘制统一风格的架构节点，并兼容不带井号的主题色常量。"""
    fill = f"#{fill}" if isinstance(fill, str) and not fill.startswith("#") else fill
    edge = f"#{edge}" if isinstance(edge, str) and not edge.startswith("#") else edge
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.1,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(box)
    fp = FontProperties(fname=str(FONT_SANS_SC), size=font_size, weight="bold" if bold else "normal")
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontproperties=fp, color="#1C2635", wrap=True)


def _arrow(ax, start, end, *, color="#64748B", width=1.2):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=width,
        color=color,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)


def generate_figures() -> None:
    """生成报告中的架构图和数据流图。"""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams["axes.unicode_minus"] = False

    # 图宽与报告正文可用宽度接近，避免图片插入 PDF 后被整体缩小到难以阅读。
    fig, ax = plt.subplots(figsize=(6.4, 3.45), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    layers = [
        (0.79, 0.13, "界面层", "Streamlit 页面｜上传下载｜状态展示｜双主题"),
        (0.62, 0.13, "编排层", "任务规划｜科研工作台｜顺序工作流｜学习闭环"),
        (0.45, 0.13, "Agent 与服务层", "论文阅读｜争议分析｜知识综合｜组卷辅导｜LLM"),
        (0.28, 0.13, "检索与工具层", "BM25｜Chroma｜PDF 解析｜Semantic Scholar｜arXiv"),
        (0.11, 0.13, "持久化层", "证据 JSON｜Chroma 索引｜SQLite｜用户 PDF"),
    ]
    for y, h, label, detail in layers:
        _draw_box(ax, 0.05, y, 0.15, h, label, fill="#EAF0FC", edge=BLUE, font_size=8.4, bold=True)
        _draw_box(ax, 0.22, y, 0.73, h, detail, fill="#FFFFFF", edge="#AAB8CA", font_size=7.7)

    for y1, y2 in ((0.24, 0.28), (0.41, 0.45), (0.58, 0.62), (0.75, 0.79)):
        _arrow(ax, (0.58, y1 + 0.005), (0.58, y2 - 0.005))

    ax.text(
        0.96,
        0.045,
        "依赖方向：界面 → 编排 → 领域服务 → 工具 → 持久化",
        ha="right",
        va="center",
        fontproperties=FontProperties(fname=str(FONT_SANS), size=6.0),
        color="#5C6979",
    )
    fig.tight_layout(pad=0.08)
    fig.savefig(ASSET_DIR / "architecture.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 2.55), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    nodes = [
        (0.03, 0.55, 0.14, 0.30, "研究任务\n与用户输入", "#EAF0FC"),
        (0.20, 0.55, 0.14, 0.30, "意图识别\n与步骤规划", "#FFFFFF"),
        (0.37, 0.55, 0.15, 0.30, "公开检索或\n本地证据检索", "#FFFFFF"),
        (0.55, 0.55, 0.14, 0.30, "页级证据\n与来源校验", "#FFFFFF"),
        (0.72, 0.55, 0.14, 0.30, "阅读、争议\n与知识综合", "#FFFFFF"),
        (0.88, 0.55, 0.10, 0.30, "输出与\n边界提示", "#EAF0FC"),
    ]
    for x, y, w, h, label, fill in nodes:
        _draw_box(ax, x, y, w, h, label, fill=fill, edge=BLUE if fill == "#EAF0FC" else "#AAB8CA", font_size=7.5)
    for i in range(len(nodes) - 1):
        x1 = nodes[i][0] + nodes[i][2]
        x2 = nodes[i + 1][0]
        _arrow(ax, (x1 + 0.006, 0.70), (x2 - 0.006, 0.70))

    _draw_box(
        ax,
        0.20,
        0.08,
        0.39,
        0.24,
        "六类题型｜自动评分\n主观题细则｜错题本｜五维掌握度",
        fill="#F5F7FA",
        edge="#94A3B8",
        font_size=7.1,
    )
    _draw_box(
        ax,
        0.63,
        0.08,
        0.35,
        0.24,
        "推荐论文｜针对性复习\n下一轮组卷｜学习成长报告",
        fill="#F5F7FA",
        edge="#94A3B8",
        font_size=7.1,
    )
    _arrow(ax, (0.79, 0.55), (0.79, 0.32), color="#64748B")
    _arrow(ax, (0.63, 0.20), (0.595, 0.20), color="#64748B")
    _arrow(ax, (0.29, 0.55), (0.29, 0.32), color="#64748B")

    fig.tight_layout(pad=0.08)
    fig.savefig(ASSET_DIR / "workflow.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _register_pdf_fonts() -> None:
    if not FONT_SANS.exists() or not FONT_SANS_BOLD.exists():
        raise FileNotFoundError("未找到中文字体 Deng.ttf / Dengb.ttf，无法生成 PDF。")
    pdfmetrics.registerFont(TTFont("Deng", str(FONT_SANS)))
    pdfmetrics.registerFont(TTFont("DengBold", str(FONT_SANS_BOLD)))


def _pdf_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "ChineseBody",
            parent=styles["BodyText"],
            fontName="Deng",
            fontSize=10.2,
            leading=16.0,
            alignment=TA_JUSTIFY,
            firstLineIndent=20.4,
            spaceAfter=3.2,
            textColor=colors.HexColor("#000000"),
            wordWrap="CJK",
        ),
        "body_no_indent": ParagraphStyle(
            "ChineseBodyNoIndent",
            parent=styles["BodyText"],
            fontName="Deng",
            fontSize=10.2,
            leading=16.0,
            alignment=TA_JUSTIFY,
            firstLineIndent=0,
            spaceAfter=3.2,
            textColor=colors.HexColor("#000000"),
            wordWrap="CJK",
        ),
        "bullet": ParagraphStyle(
            "ChineseBullet",
            parent=styles["BodyText"],
            fontName="Deng",
            fontSize=9.9,
            leading=14.8,
            alignment=TA_LEFT,
            leftIndent=15,
            firstLineIndent=-10,
            spaceAfter=2.4,
            bulletIndent=4,
            textColor=colors.HexColor("#000000"),
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "ChineseH1",
            parent=styles["Heading1"],
            fontName="DengBold",
            fontSize=14,
            leading=19,
            textColor=colors.black,
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "ChineseH2",
            parent=styles["Heading2"],
            fontName="DengBold",
            fontSize=11.5,
            leading=16,
            textColor=colors.black,
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "ChineseCaption",
            parent=styles["BodyText"],
            fontName="Deng",
            fontSize=8.8,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.black,
            spaceBefore=2,
            spaceAfter=6,
            wordWrap="CJK",
        ),
        "table": ParagraphStyle(
            "ChineseTable",
            parent=styles["BodyText"],
            fontName="Deng",
            fontSize=8.3,
            leading=11.4,
            textColor=colors.black,
            wordWrap="CJK",
        ),
        "table_header": ParagraphStyle(
            "ChineseTableHeader",
            parent=styles["BodyText"],
            fontName="DengBold",
            fontSize=8.5,
            leading=11.4,
            alignment=TA_CENTER,
            textColor=colors.white,
            wordWrap="CJK",
        ),
        "table_label": ParagraphStyle(
            "ChineseTableLabel",
            parent=styles["BodyText"],
            fontName="DengBold",
            fontSize=8.5,
            leading=11.4,
            alignment=TA_CENTER,
            textColor=colors.black,
            wordWrap="CJK",
        ),
        "reference": ParagraphStyle(
            "ChineseReference",
            parent=styles["BodyText"],
            fontName="Deng",
            fontSize=8.8,
            leading=12.8,
            alignment=TA_LEFT,
            leftIndent=17,
            firstLineIndent=-17,
            spaceAfter=2.2,
            wordWrap="CJK",
        ),
    }


def _pdf_footer(canvas, doc) -> None:
    if doc.page == 1:
        return
    canvas.saveState()
    canvas.setFont("Deng", 8.5)
    canvas.setFillColor(colors.HexColor("#5C6979"))
    canvas.drawCentredString(A4[0] / 2, 1.0 * cm, f"第 {doc.page} 页")
    canvas.restoreState()


def _pdf_table(spec: dict[str, Any], styles: dict[str, ParagraphStyle]) -> list[Any]:
    caption = Paragraph(spec["caption"], styles["caption"])
    data = [[Paragraph(str(cell), styles["table_header"]) for cell in spec["headers"]]]
    for row in spec["rows"]:
        data.append([Paragraph(str(cell), styles["table"]) for cell in row])

    usable_width = A4[0] - 4.5 * cm
    column_widths = [usable_width * width / sum(spec["widths_cm"]) for width in spec["widths_cm"]]
    table = Table(data, colWidths=column_widths, repeatRows=1, hAlign="CENTER")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#18324F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("ALIGN", (1, 1), (-1, -1), "LEFT"),
        ("ALIGN", (0, 1), (0, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D9D9D9")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
    ]
    for index in range(1, len(data)):
        if index % 2 == 0:
            commands.append(("BACKGROUND", (0, index), (-1, index), colors.HexColor("#F5F7FA")))
    table.setStyle(TableStyle(commands))
    return [caption, table, Spacer(1, 0.18 * cm)]


def build_pdf() -> None:
    """构建 PDF 版技术报告，并保留可编辑 Word 版本。"""
    _register_pdf_fonts()
    styles = _pdf_styles()
    doc = BaseDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=2.25 * cm,
        rightMargin=2.25 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
        title=TITLE,
        author="体研智脑 Agent",
        subject=FULL_PROJECT_TITLE,
    )
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        id="normal",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_pdf_footer)])

    story: list[Any] = []
    story.append(Spacer(1, 4.5 * cm))
    story.append(
        Paragraph(
            TITLE,
            ParagraphStyle(
                "PdfTitle",
                fontName="DengBold",
                fontSize=23,
                leading=32,
                alignment=TA_CENTER,
                textColor=colors.black,
                spaceAfter=18,
                wordWrap="CJK",
            ),
        )
    )
    story.append(
        Paragraph(
            SUBTITLE,
            ParagraphStyle(
                "PdfSubtitle",
                fontName="Deng",
                fontSize=13,
                leading=21,
                alignment=TA_CENTER,
                textColor=colors.black,
                spaceAfter=12,
                wordWrap="CJK",
            ),
        )
    )
    story.append(Spacer(1, 0.35 * cm))
    story.append(
        Paragraph(
            FULL_PROJECT_TITLE,
            ParagraphStyle(
                "PdfFullTitle",
                fontName="Deng",
                fontSize=10.5,
                leading=17,
                alignment=TA_CENTER,
                textColor=colors.HexColor("#38485C"),
                wordWrap="CJK",
            ),
        )
    )
    story.append(Spacer(1, 4.2 * cm))

    meta_data = [
        [Paragraph("文档类型", styles["table_label"]), Paragraph("参赛技术报告", styles["table"])],
        [Paragraph("项目版本", styles["table_label"]), Paragraph("V1.0", styles["table"])],
        [Paragraph("验证环境", styles["table_label"]), Paragraph("Python 3.12.16 / Streamlit 1.66.1", styles["table"])],
        [Paragraph("完成日期", styles["table_label"]), Paragraph("2026 年 11 月 11 日", styles["table"])],
    ]
    meta_table = Table(
        meta_data,
        colWidths=[4.0 * cm, 9.0 * cm],
        hAlign="CENTER",
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF3FB")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D9D9D9")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        ),
    )
    story.append(meta_table)
    story.append(PageBreak())

    story.append(
        Paragraph(
            "摘要",
            ParagraphStyle(
                "PdfAbstractHeading",
                fontName="DengBold",
                fontSize=14,
                leading=20,
                alignment=TA_CENTER,
                textColor=colors.black,
                spaceBefore=0,
                spaceAfter=9,
                wordWrap="CJK",
            ),
        )
    )
    story.append(Paragraph(ABSTRACT, styles["body_no_indent"]))
    story.append(Spacer(1, 0.18 * cm))
    story.append(
        Paragraph(
            f"<b>关键词：</b>{KEYWORDS}",
            styles["body_no_indent"],
        )
    )
    story.append(Spacer(1, 0.25 * cm))

    table_number = 0
    for kind, payload in CONTENT:
        if kind == "h1":
            if payload == "参考资料":
                story.append(PageBreak())
            story.append(Paragraph(payload, styles["h1"]))
        elif kind == "h2":
            story.append(Paragraph(payload, styles["h2"]))
        elif kind == "p":
            story.append(Paragraph(payload, styles["body"]))
        elif kind == "bullets":
            for item in payload:
                story.append(Paragraph(item, styles["bullet"], bulletText="•"))
            story.append(Spacer(1, 0.12 * cm))
        elif kind == "table":
            table_number += 1
            story.extend(_pdf_table(payload, styles))
        elif kind == "figure":
            image_path = Path(payload["path"])
            if not image_path.exists():
                raise FileNotFoundError(f"缺少报告图片：{image_path}")
            max_width = A4[0] - 4.5 * cm - 0.4 * cm
            target_width = min(payload["width_cm"] * cm, max_width)
            # ReportLab 只传 width 时不会自动按比例换算 drawHeight，
            # 因此读取原始像素尺寸后同时传入等比例宽高。
            probe = Image(str(image_path))
            target_height = target_width * probe.imageHeight / probe.imageWidth
            image = Image(
                str(image_path),
                width=target_width,
                height=target_height,
            )
            story.append(
                KeepTogether(
                    [
                        Spacer(1, 0.18 * cm),
                        image,
                        Paragraph(payload["caption"], styles["caption"]),
                    ]
                )
            )

    for reference in REFERENCES:
        story.append(Paragraph(reference, styles["reference"]))

    doc.build(story)


def validate_outputs() -> None:
    """执行文件存在性和页数的基础检查。"""
    if not DOCX_PATH.exists() or DOCX_PATH.stat().st_size < 10_000:
        raise RuntimeError("DOCX 未正常生成。")
    if not PDF_PATH.exists() or PDF_PATH.stat().st_size < 10_000:
        raise RuntimeError("PDF 未正常生成。")
    print(f"DOCX: {DOCX_PATH}")
    print(f"PDF:  {PDF_PATH}")
    print(f"DOCX bytes: {DOCX_PATH.stat().st_size}")
    print(f"PDF bytes:  {PDF_PATH.stat().st_size}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_figures()
    build_docx()
    build_pdf()
    validate_outputs()


if __name__ == "__main__":
    main()
