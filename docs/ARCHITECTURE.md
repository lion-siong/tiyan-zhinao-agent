# 体研智脑 Agent 技术架构

## 1. 架构目标

系统面向本机单用户的体育科研辅助，不以多租户生产平台为目标。设计优先级为：

1. 真实题录、PDF、页码和原文可以追溯。
2. 模型只处理语义任务，不拥有事实数据的最终写入权。
3. 外部检索、向量库和模型失败时保留可解释降级路径。
4. 论文库变化后，旧分析、图谱和试题不能继续冒充当前结果。
5. 核心工作流可测试、可恢复、可在 Windows 本机复现。

## 2. 分层结构

```text
页面层
├── app.py
├── researchmind/ui.py
├── researchmind/literature_research_ui.py
└── researchmind/knowledge_map_ui.py

业务编排层
├── agents/planner.py
├── agents/literature_research_agent.py
├── researchmind/workflow.py
├── researchmind/workbench.py
└── researchmind/library_state.py

领域服务与 Agent 型角色
├── researchmind/collectors.py
├── researchmind/literature_retrieval.py
├── researchmind/agents.py
├── researchmind/knowledge_map.py
├── services/rag_service.py
└── services/user_paper_rag.py

工具与持久化
├── tools/pdf_reader.py
├── tools/document_chunker.py
├── tools/library_ingest.py
├── researchmind/storage.py
└── services/vector_service.py
```

页面层不直接执行 SQL，也不直接拼装科研结论。业务层通过领域组件和工具服务读取
论文、证据、分析、题库与状态。

## 3. Agent 工作流

### 3.1 任务规划

`ResearchPlannerAgent` 采用规则识别研究类型并生成：

- 研究步骤；
- 工具路线；
- 中文数据库检索式草案；
- 选题模板；
- 研究设计检查项。

这是可复现的规则规划，不是大模型自主规划。

### 3.2 Function Calling 文献检索

```text
研究问题
  -> LLM 生成 semantic_scholar_search(query, limit)
  -> MultiSourceRetriever
     -> Semantic Scholar
     -> arXiv
     -> OpenAlex
     -> Crossref
     -> PubMed
  -> 真实元数据、去重、字段完整性分级
  -> LLM 基于返回元数据整理综述，或规则整理
```

Function Calling 只负责选择检索工具和参数；事实结果来自真实公开 API。
单源失败被隔离，429/403/5xx 按配置退避重试，网络中断可恢复。

### 3.3 ResearchMind 顺序工作流

```text
LiteratureCollector / Zotero Local API
  -> PaperReader
  -> AcademicDebateAnalyzer
  -> KnowledgeSynthesizer
  -> LearningAssessmentAgent
  -> PersonalTutorAgent
  -> 错题、掌握度与新检索需求
```

`ResearchMindWorkflow` 使用显式顺序编排，并保存论文库指纹、分析范围和状态。
两个后台线程最多处理两个分析任务；相同数据库、领域和指纹的任务会合并。

## 4. 证据模型

`EvidenceRef` 是全局事实引用对象：

| 字段 | 含义 |
| --- | --- |
| `chunk_id` | 证据片段稳定 ID |
| `document_id` | 论文 ID |
| `document_title` | 论文题名 |
| `file_name` | PDF 文件名 |
| `page_number` | 页码 |
| `quote` | 原文片段 |
| `source_type` | 全文、摘要或元数据 |
| `section_hint` | 方法、结果、结论等段落提示 |

阅读、RAG、争议、图谱和测评复用同一证据模型。摘要级证据明确标记，不伪装成
全文页码证据。

## 5. RAG 架构

### 5.1 检索

- BM25：中文双字与英文连续词项；
- Chroma：`intfloat/multilingual-e5-small`；
- 混合检索：关键词候选与语义候选融合；
- 语义失败或 HNSW 损坏：隔离旧索引并从证据文件重建，期间可回退 BM25。

### 5.2 回答

| 路径 | 行为 |
| --- | --- |
| 证据约束模型 | 只接收检索片段，关键结论标证据编号 |
| 本地摘录 | 不调用模型，展示命中原文并明确标记 |
| 证据不足 | 不调用模型，不补写结论 |

## 6. 数据事实源

| 数据 | 位置 | 事实地位 |
| --- | --- | --- |
| PDF | `knowledge_base*/pdfs/`、Zotero 路径、用户目录 | 原文文件 |
| 页级证据 | SQLite `paper_evidence`、JSON 索引 | 页码级事实源 |
| 论文与分析 | SQLite | 当前科研状态 |
| Chroma | 知识库 `chroma/` | 可重建派生索引 |
| 用户上传 PDF | `data/user_papers/` | 隔离私有空间 |
| 旧库 | `knowledge_base/` | 默认只读，不自动迁移 |

## 7. 一致性与恢复

- 论文元数据、证据、分类和题名规范化结果生成 SHA-256 论文库指纹。
- 指纹变化后，旧争议、综合、图谱和测评进入失效或待更新状态。
- 软删除保留论文、证据和 PDF；恢复后重新进入查询。
- 永久删除要求先进入回收站并生成备份。
- 共享 PDF 不会因删除单篇论文而被删除。
- SQLite 写入使用事务；PDF 入库失败时恢复清单、索引和框架。

## 8. 异常与降级

| 失败 | 行为 |
| --- | --- |
| 模型未配置 | 明示模拟回答或原文摘录，不伪造引用 |
| Function Calling 失败 | 改用规则检索，保留失败警告 |
| 单源限流/超时 | 重试、退避、切到其他公开来源 |
| 向量库失败 | BM25 回退或隔离重建 |
| 无证据 | 返回证据不足 |
| PDF 损坏/加密/扫描版 | 拒绝或提示无法建立文字索引 |
| 数据库变化 | 旧分析冻结并后台重算 |
| 组卷字段不足 | 跳过无法生成题型并说明字段缺口 |

## 9. 真实性与边界

- 六个角色由同一工作流顺序调用，不是六个独立大模型。
- Function Calling、RAG、主观题评分会调用模型；其余科研判断以规则和证据为主。
- 自动共识/争议是初筛，不是正式系统综述。
- 当前没有完整事件日志、多租户权限、OCR 和 500 页实测。
