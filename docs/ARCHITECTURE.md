# 体研智脑 Agent 系统架构

本文档描述当前代码中已经实现的架构，不把规划中的能力写成已完成能力。

## 1. 架构目标

系统面向本机单用户的科研辅助与竞赛演示，优先保证四件事：

1. 外部文献只读取公开元数据或明确开放获取的 PDF。
2. 论文回答、观点比较和测评题目尽量保留文件名、题名和页码证据。
3. 外部模型、检索接口或向量库不可用时，页面不崩溃，并显示可执行的提示。
4. 模拟回答、规则模板和真实模型输出必须明确区分。

当前没有账号、租户隔离、分布式任务队列和自主多 Agent 协商。

## 2. 分层结构

```text
Streamlit 页面层
├── app.py
├── researchmind/ui.py
└── researchmind/literature_research_ui.py
        |
业务编排层
├── agents/planner.py
├── agents/literature_research_agent.py
├── researchmind/workflow.py
└── researchmind/workbench.py
        |
领域服务与 Agent 层
├── researchmind/collectors.py
├── researchmind/agents.py
├── services/llm_service.py
├── services/rag_service.py
├── services/user_paper_rag.py
└── services/vector_service.py
        |
工具与持久化层
├── tools/pdf_reader.py
├── tools/document_chunker.py
├── tools/library_ingest.py
├── tools/literature_framework.py
├── tools/semantic_scholar.py
├── tools/arxiv_search.py
├── researchmind/storage.py
└── knowledge_base/、data/
```

页面层不直接拼接数据库 SQL。科研流程通过业务层调用 Agent、工具和存储类，便于逐模块测试。

## 3. 主要运行路径

### 3.1 科研工作台

```text
用户研究问题
  -> 规则任务规划
  -> 检索词与中文数据库表达式草稿
  -> 本地证据检索
  -> 研究设计模板和阶段状态
  -> JSON 工作底稿导出
```

科研报告不会被系统自动标记为完成。没有真实执行的研究步骤仍显示为“待完成”或模板状态。

### 3.2 论文导入与知识框架

```text
PDF 或开放获取下载
  -> 文件类型、大小和可读性校验
  -> SHA-256 去重
  -> PyMuPDF 提取文本
  -> 书目信息识别
  -> 页级证据切分
  -> manifest.json / evidence_chunks.json
  -> 九维研究框架
  -> BM25 与 Chroma 索引
  -> ResearchMind SQLite 论文库
```

`knowledge_base/pdfs/` 保存已登记论文，`knowledge_base/index/evidence_chunks.json` 保存页级证据，`knowledge_base/framework.json` 保存九维归类。Chroma 是可从证据文件重建的派生索引。

### 3.3 公开文献检索与综述

```text
研究问题
  -> 配置模型时先通过 Function Calling 优化检索词
  -> semantic_scholar_search(query, limit)
  -> Semantic Scholar 元数据
  -> 无结果或服务不可用时尝试 arXiv
  -> 只基于返回元数据整理研究现状
  -> GB/T 7714 风格引用草稿
```

在当前开发网络曾观察到 Semantic Scholar 返回 HTTP 403。代码已提供错误转换、arXiv 回退和界面提示，但正式参赛前必须在评审网络重新执行真实检索，不能仅根据单元测试宣称外部服务可用。

公开检索只处理题名、作者、年份、摘要、DOI 和开放 PDF 链接，不写入用户 PDF 向量库。

### 3.4 用户 PDF 隔离 RAG

```text
用户主动上传 PDF
  -> 文件校验与 SHA-256
  -> data/user_papers/<SHA-256>/
  -> 页级文本与证据文件
  -> 独立 Chroma 集合 user_uploaded_papers
  -> 返回所选文件、页码和原文证据
```

外部检索元数据与用户上传全文使用不同存储路径和不同 Chroma 集合，避免把公开元数据误当成用户全文证据。

### 3.5 ResearchMind 六阶段工作流

```text
LiteratureCollector
  -> PaperReader
  -> AcademicDebateAnalyzer
  -> KnowledgeSynthesizer
  -> LearningAssessmentAgent
  -> PersonalTutorAgent
```

职责对应关系：

| 阶段 | 当前实现 | 主要输出 |
| --- | --- | --- |
| 文献采集 | `researchmind/collectors.py` | OpenAlex/Crossref 元数据、开放 PDF 候选 |
| 论文阅读 | `researchmind/agents.py::PaperReader` | 目的、对象、方法、结果、结论、局限和证据 |
| 争议分析 | `AcademicDebateAnalyzer` | 观点、论文、原文、页码、方法差异和证据强度 |
| 知识综合 | `KnowledgeSynthesizer` | 概念、理论框架、共识、分歧、局限与空白 |
| 学习测评 | `LearningAssessmentAgent` | 六种题型、标准答案、评分细则和知识点 |
| 个性化辅导 | `PersonalTutorAgent` | 批改记录、薄弱点、推荐论文和下一轮计划 |

`researchmind/workflow.py` 是显式顺序编排，不是 LangGraph 图执行。保留这一层是为了未来在不改写 Agent 和存储接口的前提下接入 LangGraph。

## 4. 检索与回退

检索模式由 `RAG_RETRIEVAL_MODE` 控制：

| 模式 | 行为 |
| --- | --- |
| `hybrid` | 合并 BM25 与 Chroma 语义结果 |
| `semantic` | 优先 Chroma，不可用时回退 BM25 |
| `bm25` | 只使用本地关键词检索 |

Chroma 使用 `sentence-transformers` 生成向量。模型下载失败、HNSW 派生索引损坏或集合不可打开时，系统给出提示并保留 BM25 可用路径。发现 HNSW 损坏时，服务会尝试把旧目录隔离为 `knowledge_base/chroma.corrupt-时间戳` 后重建。

## 5. 数据与隐私边界

| 数据 | 默认位置 | Git/公开导出 |
| --- | --- | --- |
| 环境变量与密钥 | `.env` | 禁止 |
| 学习记录与答题历史 | `data/researchmind.sqlite3` | 禁止 |
| 用户上传 PDF 与证据 | `data/user_papers/` | 禁止 |
| 论文全文和派生索引 | `knowledge_base/pdfs/`、`index/`、`chroma/` | 默认禁止，逐篇许可审核 |
| 浏览器资料与日志 | `.tmp/`、`outputs/` | 禁止 |
| 示例配置 | `.env.example` | 允许，只能有变量名和空占位符 |

公开部署前必须增加账号隔离、上传隔离、请求限流、操作审计和密钥管理；当前版本不应直接作为生产级多用户服务使用。

## 6. 外部接口与错误处理

| 接口 | 用途 | 失败时的当前行为 |
| --- | --- | --- |
| OpenAI 兼容 `/chat/completions` | 问答、Function Calling、主观题评分 | 无 Key 或调用失败时显示提示；支持能力回退到规则或抽取式回答 |
| Semantic Scholar Graph API | 公开论文元数据 | 转换为中文错误，尝试 arXiv |
| arXiv Atom API | 公开预印本元数据 | 返回空结果并提示 |
| OpenAlex | 文献自动采集 | 尝试 Crossref |
| Crossref | 文献自动采集回退 | 返回可展示错误 |
| Zotero Web API | 元数据与附件同步 | 未配置或失败时不阻断本地功能 |

所有真实 Key 必须从环境变量读取。日志、异常消息和导出材料不得包含 Authorization 头或 Key 原文。

## 7. 测试与验收层级

1. 单元测试：数据模型、检索参数、PDF 导入、评分、安全审计。
2. 工作流测试：论文库同步、争议分析、知识综合、组卷和错题推荐。
3. 页面测试：Streamlit 页面可导入和关键交互函数可调用。
4. 外部接口测试：只记录真实执行结果，不用 Mock 结果宣称线上服务可用。
5. 提交包审计：文件名、压缩包成员与文本内容同时检查，PDF 单独人工复核许可。

完整执行结果应以当前 `pytest`、语法检查、`pip check`、Streamlit 健康检查和导出包审计的实际输出为准。
