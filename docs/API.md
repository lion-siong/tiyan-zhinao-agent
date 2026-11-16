# API 与核心接口说明

项目没有独立 HTTP API 服务，主要交互入口是 Streamlit。以下列出后续二开最常用的
Python 接口和输入输出。

## 1. 规划

```python
ResearchPlannerAgent.create_plan(task: str) -> ResearchPlan
ResearchPlannerAgent.generate_topic_suggestions(
    direction: str,
    count: int = 3,
) -> tuple[TopicSuggestion, ...]
```

规划为规则模式，不调用模型；返回步骤、工具路线和选题模板。

## 2. 文献研究与多源检索

```python
LiteratureResearchAgent.run(
    research_question: str,
    limit: int = 8,
) -> LiteratureResearchResult
```

输入研究问题，输出题录、综述、引用、来源报告和警告。`LiteratureResearchResult`
包含 `papers`、`references`、`report`、`candidates`、`mode`。

```python
MultiSourceRetriever.search(
    query: str,
    limit: int = 10,
    date_from: str | None = None,
    date_to: str | None = None,
) -> RetrievalReport
```

按 Semantic Scholar、arXiv、OpenAlex、Crossref、PubMed 汇总，单源失败隔离并
返回每个来源的状态、数量、耗时、重试次数和错误建议。

## 3. 工作流

```python
ResearchMindWorkflow.ensure_default_domain() -> ResearchDomain
ResearchMindWorkflow.run_analysis_pipeline(
    domain_id: str,
    topic: str = "青少年体育锻炼干预与体质健康水平的关系",
    generate_questions: bool = True,
) -> PipelineResult
```

输出包括 `analyses`、`debate`、`synthesis` 和 `questions`。

```python
ResearchMindWorkflow.grade_learning_cycle(
    domain_id: str,
    answers: Mapping[str, str],
    topic: str,
    generate_next: bool = True,
    assessment_modules: Iterable[str] | None = None,
) -> LearningCycleResult
```

批改答题、更新五维掌握度、记录错题并生成下一轮题目。

## 4. Zotero

```python
probe_zotero_local(base_url: str) -> dict[str, Any]
sync_zotero_local_library(workflow, base_url: str) -> ZoteroLocalSyncResult
```

同步返回活动题录数、新增、更新、仅题录、PDF 解析数、证据数和失效数。
`sync_zotero_local_library()` 不修改 Zotero，只读取活动库。

## 5. RAG

```python
EvidenceRAGService.answer_question(
    question: str,
    top_k: int = 5,
    allow_simulation: bool = True,
) -> EvidenceAnswer
```

`EvidenceAnswer` 包含回答、证据、模式、警告和成功状态。无证据时不会调用模型补写。

```python
UserPaperRAGService.ingest_pdf(...)
UserPaperRAGService.list_papers()
UserPaperRAGService.answer(...)
```

用户 PDF 使用独立目录和独立 Chroma 集合，与专题论文库隔离。

## 6. 知识图谱

```python
build_knowledge_map(
    domain_name,
    papers,
    analyses,
    evidence_by_paper,
    debate,
    synthesis,
) -> KnowledgeMap
```

输出五层节点、关系、学术发现、统计和科研洞察。节点和关系均保留论文/证据来源。

## 7. 测评

```python
LearningAssessmentAgent.generate_question_set(
    domain_id,
    question_types=None,
    assessment_modules=None,
    debate_topic="体育锻炼行为与体质健康",
) -> tuple[AssessmentQuestion, ...]

LearningAssessmentAgent.score(
    question: AssessmentQuestion,
    user_answer: str,
) -> ScoreResult
```

题目必须关联活动论文和原文证据；客观题标准判分，主观题五维细则评分。

## 8. 错误对象

- `LiteratureCollectionError`
- `RetrievalSourceError`
- `VectorStoreError`
- `KnowledgeBaseError`
- `ResearchMindStorageError`
- `AssessmentGenerationError`
- `ZoteroError` / `ZoteroLocalError`

错误信息应显示可操作原因，不把内部异常直接作为无解释页面错误。
