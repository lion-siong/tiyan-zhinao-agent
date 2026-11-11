# 体研智脑 Agent

**基于大语言模型与多智能体协作的体育科研全流程智能助手**

体研智脑 Agent 面向体育教育专业学生、体育教师和科研初学者，将科研选题、文献检索、PDF 阅读、证据溯源、数据分析、学习测评和复习反馈组织成可检查、可追溯的工作流。项目中的 ResearchMind 是 AI 学术知识发现与主动学习引擎。

项目追求的是“真实资料可核验、模拟内容有标注、未执行步骤不虚报”，而不是用一段流畅文字替代完整科研过程。

## 一、核心能力

- **科研工作台**：把研究问题、检索策略、真实文献证据、研究设计模板和数据分析结果汇总到同一页面，并支持导出 JSON 工作底稿。
- **科研任务规划**：根据用户任务生成研究步骤、选题模板和中文数据库检索表达式草稿。
- **真实文献库**：内置 12 篇已登记论文、1095 条按页切分证据，打开应用后可直接检索。
- **PDF 自动导入**：校验文件、计算 SHA-256、识别书目信息、去重、保存 PDF，并自动更新清单、证据索引和研究框架。
- **开放学术检索**：通过 OpenAlex 检索真实论文元数据，OpenAlex 不可用时回退到 Crossref；只下载明确标记为开放获取的 PDF。
- **Semantic Scholar 文献研究**：通过 Function Calling 让模型先优化研究问题中的检索词，再调用 `semantic_scholar_search` 获取题名、作者、年份、摘要、DOI 和开放 PDF 链接。
- **公开预印本回退**：Semantic Scholar 无结果或暂时不可用时，可从 arXiv 官方 API 读取公开预印本元数据；不爬取知网，不绕过付费墙。
- **综述与引用整理**：配置大模型后，Agent 只依据检索到的元数据生成文献综述与研究现状，并整理为可人工核验的 GB/T 7714 风格引用。
- **用户 PDF 隔离 RAG**：用户主动上传的 PDF 按内容 SHA-256 隔离保存，在独立 Chroma 集合中检索，并展示文件名、页码和原文证据。
- **混合证据检索**：同时使用 BM25 关键词检索与 Chroma 语义检索，语义服务不可用时自动回退到 BM25。
- **页码级溯源**：文献问答、观点比较、争议分析、知识体系和测评题目均保留论文题名、文件名和 PDF 页码。
- **九维研究框架**：将论文证据归入研究背景、研究对象、研究方法、个体、家庭、学校、社会、行为结果和干预对策九个维度。
- **ResearchMind 六阶段工作流**：依次完成文献采集、论文阅读、学术争议分析、知识体系构建、自动组卷和个性化辅导。
- **学习闭环**：生成六种题型，自动评分或按评分细则评分，记录错题本、五维掌握度、推荐论文和下一轮复习题目。
- **数据分析**：读取 CSV/XLSX，完成数据预览、字段类型、缺失值、描述统计和数值列直方图。
- **可选大模型接入**：支持 OpenAI 兼容的 `/chat/completions` 接口；没有 API Key 时使用明确标注的本地模拟回答。
- **可选 Zotero 同步**：配置 Zotero Web API 后，可同步论文元数据和本地 PDF 附件。

## 二、界面结构

应用当前共有 20 个页面。

**核心科研工作区（10 个）**

1. 首页
2. 科研工作台
3. 智能科研问答
4. 科研选题
5. 文献阅读
6. 智能文献研究
7. 文献证据问答
8. 文献框架
9. 数据分析
10. 系统状态

**ResearchMind 学术与学习引擎（10 个）**

1. 学术领域管理
2. 文献自动采集
3. 我的论文库
4. 论文智能阅读
5. 学术观点对比
6. 学术分歧分析
7. 知识图谱
8. AI 学习测评
9. 错题本
10. 学习成长报告

**双主题配色**

页面支持 Streamlit 右上角菜单中的 Light / Dark 外观切换：

| 用途 | 主色 | 页面背景 | 面板背景 | 正文 | 边框 |
| --- | --- | --- | --- | --- | --- |
| 日间白色 | `#2457D6` 钴蓝 | `#F6F8FB` | `#FFFFFF` | `#182235` | `#D7E0EA` |
| 夜间深色 | `#3B68EE` 星蓝 | `#0B111B` | `#121B2A` | `#E7EEF9` | `#26354D` |

两套主题共用琥珀色提醒、绿色成功状态和红色错误状态，自定义首页卡片、侧边栏、
表单和论文框架组件会随主题同步切换。

## 三、技术栈

| 类别 | 技术 | 用途 |
| --- | --- | --- |
| 运行环境 | Python 3.11 / 3.12 | 应用运行与测试 |
| 界面 | Streamlit | 中文交互页面、上传、下载和状态展示 |
| 数据计算 | Pandas、openpyxl | CSV/XLSX 读取和描述统计 |
| PDF | PyMuPDF | PDF 文本提取、页级切分 |
| 可视化 | Matplotlib | 数值列直方图 |
| 检索 | 本地 BM25 | 中英文关键词检索与无网络回退 |
| 语义检索 | sentence-transformers、Chroma | 多语言向量生成与本地持久化检索 |
| 持久化 | SQLite、JSON | 论文库、学习记录、证据索引和研究框架 |
| 配置 | python-dotenv | 从 `.env` 读取密钥和运行参数 |
| 外部接口 | OpenAI 兼容接口、Semantic Scholar、arXiv、OpenAlex、Crossref、Zotero Web API | 问答、Function Calling、文献元数据与同步 |
| 测试 | Pytest | 74 个自动化测试 |

## 四、快速开始

以下命令以 Windows PowerShell 为例。

### 1. 环境要求

- Windows 10/11、Linux 或 macOS
- Python 3.11 或 3.12
- 首次构建语义向量库时需要网络下载嵌入模型
- Semantic Scholar、arXiv、OpenAlex、Crossref、Zotero 和真实大模型调用需要可访问对应网络服务

### 2. 进入项目目录

```powershell
cd "F:\体研智脑agant"
```

### 3. 创建虚拟环境

```powershell
py -3.12 -m venv .venv
```

如果系统没有 `py` 启动器：

```powershell
python -m venv .venv
```

### 4. 安装依赖

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 5. 配置环境变量

不配置 API Key 也可以运行本地文献库、证据摘录、规则规划和数据分析：

```powershell
Copy-Item .env.example .env
```

如需调用真实大模型，在 `.env` 中填写：

```dotenv
OPENAI_API_KEY=你的密钥
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

例如使用 DeepSeek 的 OpenAI 兼容接口：

```dotenv
OPENAI_API_KEY=你的DeepSeek密钥
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
```

如需 Zotero 同步：

```dotenv
ZOTERO_API_KEY=你的Zotero密钥
ZOTERO_LIBRARY_ID=你的用户或群组数字ID
ZOTERO_LIBRARY_TYPE=user
```

如需提高 Semantic Scholar 调用额度，可另行配置无需必填的公开 API Key：

```dotenv
SEMANTIC_SCHOLAR_API_KEY=你的SemanticScholar密钥
SEMANTIC_SCHOLAR_TIMEOUT_SECONDS=25
SEMANTIC_SCHOLAR_MAX_RESULTS=20
ARXIV_API_URL=https://export.arxiv.org/api/query
```

真实密钥只放在 `.env` 中，不要提交到代码仓库、截图或聊天记录。

### 6. 启动应用

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器通常会自动打开 `http://localhost:8501`。停止服务时，在终端按 `Ctrl+C`。

用于临时公网演示时，应显式清空密钥并把 ResearchMind 数据库指向隔离位置：

```powershell
$env:OPENAI_API_KEY=""
$env:ZOTERO_API_KEY=""
$env:ZOTERO_LIBRARY_ID=""
$env:RESEARCHMIND_DB_PATH="$env:TEMP\researchmind-public-demo.sqlite3"
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true
```

该应用尚无账号和权限控制，不要将真实用户数据或生产数据库直接暴露到公网。

### 7. 可选：构建语义向量库

应用会在需要时自动尝试同步向量库。也可以手动构建：

```powershell
.\.venv\Scripts\python.exe -m services.vector_service --build
```

首次构建会下载 `intfloat/multilingual-e5-small` 模型，缓存目录为 `data/models`。如果模型或 Chroma 不可用，系统会自动使用 BM25。

如果界面提示 `Error loading hnsw index`，说明 Chroma 的 HNSW 派生索引
文件损坏。原始 PDF、`manifest.json` 和证据片段索引不会被修改。再次点击
“构建或更新本地向量库”时，系统会先把损坏目录隔离为
`knowledge_base/chroma.corrupt-时间戳`，再从原始证据自动重建。只有在确认
备份不再需要后，才应手动清理该备份目录。

### 8. 运行测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

仅收集测试用例：

```powershell
.\.venv\Scripts\python.exe -m pytest --collect-only -q
```

### 9. 10 分钟极速复现

评审或另一台电脑复现时，按以下顺序执行：

```powershell
cd "F:\体研智脑agant"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app.py
```

验收时依次确认：

1. 首页可以打开，侧边导航为精简一级功能入口。
2. 科研工作台可以生成工作底稿，且不会把未执行步骤标记为完成。
3. 文献证据问答可以显示论文题名、文件名、页码和原文。
4. 未配置 API Key 时页面不崩溃，并明确标注模拟回答或本地规则模式。
5. AI 学习测评提交后同时显示用户答案、标准答案、得分和评分依据。
6. 执行 `pytest -q` 后得到 `74 passed`。

首次语义检索需要联网下载 `intfloat/multilingual-e5-small`。下载失败时系统自动回退 BM25，不阻断本地演示。若只做离线演示，可提前运行一次向量构建，并在演示前确认 `knowledge_base/chroma/`、论文清单和证据索引均可读。

### 10. 复现性声明

本项目当前可核验结果为：74 项自动化测试通过、Python 语法检查通过、Streamlit 健康检查返回 `ok`、12 篇登记论文与 1095 条页级证据可读。上述结果只说明当前工作区的软件功能状态，不代表已经完成公开仓库、真实 Prompt 链或现场演示视频等参赛流程材料。

## 五、推荐使用流程

### 科研工作流演示

1. 在“科研工作台”输入研究问题、研究对象和研究方法。
2. 查看规则规划、检索策略和研究设计模板。
3. 检索当前领域中的真实论文证据，核对文件名和页码。
4. 到“数据分析”上传真实 CSV/XLSX 并完成描述统计。
5. 回到工作台，查看各阶段状态并导出 JSON 工作底稿。
6. 点击“科研报告”仍保持“待撰写”，由研究者依据证据完成报告。

### 文献导入与研究框架

1. 在“文献自动采集”中检索真实论文元数据。
2. 仅对明确开放获取的候选论文下载 PDF。
3. 系统校验 PDF、计算 SHA-256、识别书目信息并去重。
4. PDF 保存到 `knowledge_base/pdfs/`。
5. `manifest.json`、证据索引、向量库和 `framework.json` 自动更新。
6. 新论文同步到 ResearchMind 的 SQLite 论文库。

也可以把现有 PDF 直接放入 `knowledge_base/pdfs/`，然后在“文献框架”或“我的论文库”中执行扫描和同步。

### ResearchMind 学习闭环

1. 在“学术领域管理”中创建或选择研究领域。
2. 同步项目论文库。
3. 在“论文智能阅读”中查看目的、对象、方法、结果、结论和局限。
4. 查看观点对比、学术分歧和知识图谱。
5. 在“AI 学习测评”中完成六类题目。
6. 在“错题本”和“学习成长报告”中查看评分细则、薄弱点、推荐论文和下一轮题目。

### 公开文献检索与用户 PDF RAG

1. 进入“智能文献研究”，默认打开“公开文献检索”。
2. 输入研究问题并选择返回条数。
3. 配置模型后，Agent 会先优化检索词并调用 `semantic_scholar_search`；未配置模型时仍执行真实公开检索，但明确标注为规则整理。
4. 查看论文元数据、研究现状、检索提示和 GB/T 7714 风格引用，并导出 JSON 结果。
5. 切换到“个人论文 RAG”，上传自己有权使用的 PDF。
6. 系统保存分页证据并建立独立 Chroma 索引，再基于所选论文原文回答并展示页码证据。

公开检索只处理外部元数据；用户 PDF 库只处理用户主动上传的文件，两套数据不会互相写入。

## 六、命令行入口

`main.py` 用于快速检查规划器和问答服务：

```powershell
.\.venv\Scripts\python.exe main.py --task "研究大学生久坐行为与体质健康的关系"
.\.venv\Scripts\python.exe main.py --topic "体育课堂即时反馈" --count 3
.\.venv\Scripts\python.exe main.py --ask "准实验研究需要哪些基本步骤？"
```

常用检查命令：

```powershell
.\.venv\Scripts\python.exe -m compileall app.py main.py config.py agents tools services researchmind tests
.\.venv\Scripts\python.exe -m tools.knowledge_base
```

重建论文库索引和研究框架：

```powershell
.\.venv\Scripts\python.exe -c "from tools.library_ingest import rebuild_library; rebuild_library()"
```

## 七、目录结构

```text
体研智脑agant/
├── app.py                         Streamlit 主入口与核心页面
├── main.py                        命令行入口
├── config.py                      环境变量与运行参数
├── requirements.txt               依赖清单
├── .env.example                   配置模板
├── README.md                      项目说明
├── 技术文档.md                    技术设计与维护文档
├── .streamlit/
│   └── config.toml                主题配置
├── agents/
│   ├── planner.py                 任务规划、选题模板与检索策略
│   └── literature_research_agent.py
│                                    Function Calling 文献检索与综述
├── services/
│   ├── llm_service.py             OpenAI 兼容接口与模拟回答
│   ├── rag_service.py             BM25、混合检索与证据问答
│   ├── vector_service.py          Chroma 与本地嵌入模型
│   ├── prompt_service.py          证据约束提示词
│   └── user_paper_rag.py          用户 PDF 隔离存储与独立 RAG
├── researchmind/
│   ├── models.py                  强类型数据模型
│   ├── storage.py                 SQLite 持久化
│   ├── agents.py                  五个学术与学习 Agent
│   ├── collectors.py              OpenAlex/Crossref 与开放 PDF
│   ├── workflow.py                六阶段顺序编排
│   ├── workbench.py               科研工作台逻辑与导出
│   ├── ui.py                      ResearchMind 十个页面
│   ├── literature_research_ui.py  公开文献检索与用户 PDF RAG 页面
│   └── zotero_service.py          Zotero Web API 同步
├── tools/
│   ├── pdf_reader.py              PDF 文本提取
│   ├── pdf_metadata.py            PDF 书目识别
│   ├── document_chunker.py        页级证据切分
│   ├── knowledge_base.py          文献清单与索引校验
│   ├── library_ingest.py          论文导入、去重与重建
│   ├── literature_framework.py    九维研究框架
│   ├── literature_search.py       检索表达式生成
│   ├── semantic_scholar.py        Semantic Scholar 公开元数据检索
│   ├── arxiv_search.py            arXiv 公开预印本回退
│   ├── citation_formatter.py      GB/T 7714 风格引用整理
│   └── data_analysis.py           CSV/XLSX 分析
├── knowledge_base/
│   ├── manifest.json              12 篇论文书目清单
│   ├── framework.json             九维研究框架
│   ├── pdfs/                      内置论文 PDF
│   ├── index/
│   │   └── evidence_chunks.json   1095 条页级证据
│   └── chroma/                    Chroma 持久化向量库
├── data/
│   ├── researchmind.sqlite3       ResearchMind 数据库
│   ├── user_papers/               用户主动上传的 PDF 与分页证据
│   └── models/                    嵌入模型缓存
├── outputs/                       日志和输出文件
└── tests/                         74 个自动化测试
```

## 八、主要配置项

完整配置见 [技术文档.md](技术文档.md) 和 `.env.example`。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 空 | OpenAI 或兼容服务密钥；为空时使用模拟回答 |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | 兼容接口地址 |
| `OPENAI_MODEL` | `gpt-4o-mini` | 模型名称 |
| `LLM_TIMEOUT_SECONDS` | `60` | 外部接口超时秒数 |
| `SEMANTIC_SCHOLAR_API_KEY` | 空 | Semantic Scholar 可选密钥；匿名也可检索公开元数据 |
| `SEMANTIC_SCHOLAR_BASE_URL` | `https://api.semanticscholar.org/graph/v1` | Semantic Scholar Graph API 地址 |
| `SEMANTIC_SCHOLAR_TIMEOUT_SECONDS` | `25` | Semantic Scholar 请求超时秒数 |
| `SEMANTIC_SCHOLAR_MAX_RESULTS` | `20` | 单次公开检索最大返回条数 |
| `ARXIV_API_URL` | `https://export.arxiv.org/api/query` | arXiv 官方 Atom API 地址 |
| `MAX_PDF_SIZE_MB` | `20` | 单篇 PDF 大小限制 |
| `MAX_UPLOAD_SIZE_MB` | `30` | 单次上传大小限制 |
| `MAX_PDF_PAGES` | `80` | 单篇 PDF 最大处理页数 |
| `MAX_PDF_CHARS` | `40000` | 单篇 PDF 最大提取字符数 |
| `RAG_CHUNK_CHARS` | `700` | 证据片段目标长度 |
| `RAG_CHUNK_OVERLAP` | `120` | 片段重叠字符数 |
| `RAG_TOP_K` | `5` | 默认召回证据数 |
| `RAG_MAX_DOCUMENTS` | `5` | 临时上传 PDF 数量上限 |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | 本地嵌入模型 |
| `CHROMA_COLLECTION_NAME` | `evidence_chunks` | Chroma 集合名称 |
| `RAG_RETRIEVAL_MODE` | `hybrid` | `hybrid`、`semantic` 或 `bm25` |
| `RESEARCHMIND_DATE_FROM` | `2023-01-01` | 文献检索起始日期 |
| `RESEARCHMIND_DB_PATH` | `data/researchmind.sqlite3` | ResearchMind SQLite 路径；演示时可隔离数据 |
| `OPENALEX_MAILTO` | 空 | OpenAlex 礼貌联系邮箱 |
| `ZOTERO_API_KEY` | 空 | Zotero API 密钥 |
| `ZOTERO_LIBRARY_ID` | 空 | Zotero 用户或群组 ID |
| `ZOTERO_LIBRARY_TYPE` | `user` | `user` 或 `group` |

## 九、数据与安全

- `.env` 已被 `.gitignore` 忽略，密钥不会写入公开文档。
- `data/` 下的 SQLite 数据库、模型缓存和用户数据默认不纳入版本管理。
- 用户上传 PDF 保存在 `data/user_papers/<SHA-256>/`，分页证据与 PDF 同目录，Chroma 位于 `data/user_papers/chroma/`。
- 用户论文库使用独立集合 `user_uploaded_papers`；Semantic Scholar 和 arXiv 元数据不会写入该集合。
- `knowledge_base/pdfs/` 当前包含已登记的 12 篇真实论文，引用时仍应回到原文核对。
- 上传 PDF 和 CSV/XLSX 前，应移除身份证号、手机号、学号等不必要的个人信息。
- 系统不会自动生成实验数据、DOI、作者、页码或统计结果。
- 只有明确开放获取的 PDF 才会由自动采集功能下载，系统不绕过付费墙。
- 本项目不爬取知网；公开检索仅使用 Semantic Scholar 与 arXiv 官方接口。
- Semantic Scholar/arXiv 的摘要可能缺失或截断，综述和研究现状必须回到原文核验。
- 本应用默认是本机单用户工具，尚未实现账号、租户、权限和审计日志。

## 十、能力边界

- 未配置 API Key 时，科研问答和证据综合使用本地模板或原文摘录，并明确标注模拟属性。
- 选题模板、任务规划和设计模板是规则化建议，不是已经完成的研究成果。
- 论文阅读、九维归类、观点比较和知识图谱主要使用本地规则与检索，不是自动综述。
- 相关性研究不能被直接解释为因果关系，单篇研究也不能被表述为普遍共识。
- PDF 扫描件需要 OCR，当前版本未实现 OCR。
- 数据分析提供描述统计，不替代统计检验、研究设计和专业判断。
- ResearchMind 目前是顺序工作流，不是 LangGraph 图执行，也不是自主多 Agent 协商。
- Semantic Scholar、arXiv、OpenAlex、Crossref、Zotero、模型接口和向量模型下载依赖网络服务可用性。
- 项目定位为竞赛演示和本地科研辅助工具，不应直接作为生产级多用户系统部署。

## 十一、文档

- 项目使用与快速开始：`README.md`
- 架构、数据模型、接口、测试与部署：`技术文档.md`

技术文档包含模块职责、数据流、SQLite 表结构、BM25 与混合检索参数、六 Agent 工作流、配置项、测试覆盖、部署建议和后续扩展路线。
