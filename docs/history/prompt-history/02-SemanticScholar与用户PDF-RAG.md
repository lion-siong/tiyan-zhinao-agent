# Prompt 链记录二：公开文献检索与用户 PDF RAG

## 1. 记录说明

- 记录性质：根据当前对话中真实出现的需求整理，不是原始聊天逐字稿。
- 对应功能：Semantic Scholar 工具调用、arXiv 回退、GB/T 7714 引用草稿、用户 PDF 隔离 RAG。
- 当前状态：代码和单元测试已存在；Semantic Scholar 在当前开发网络曾出现 HTTP 403，需在评审网络复测。

## 2. 真实用户 Prompt

> 帮我写一个Python的大模型Agent，集成Semantic Scholar论文检索功能。
> 要求：
> 1. 封装一个工具函数 semantic_scholar_search，参数：query, limit（返回论文条数）
> 2. 调用Semantic Scholar公开API，返回论文标题、作者、发表年份、摘要、DOI、pdf链接
> 3. 用function call模式：Agent先理解用户的研究问题，自动优化检索关键词，再调用这个工具检索论文
> 4. 拿到结果之后，自动整理文献综述、归纳研究现状，生成GB/T 7714引用格式
> 5. 额外支持用户上传PDF论文，做本地RAG向量检索（Chroma），可以基于上传的论文原文问答
> 6. 不要爬虫知网，只用Semantic Scholar + arXiv公开预印本；代码加注释，适合参赛项目（传智杯）
> 7. 区分：对外检索文献元信息，对内RAG只读取用户自己上传的PDF，规避版权风险

## 3. 人工约束

1. 不爬取知网，不绕过付费墙。
2. 公开检索只使用 Semantic Scholar 与 arXiv 官方接口。
3. 综述只能依据本次检索返回的元数据，不能补写不存在的结论。
4. GB/T 7714 引用只格式化已有字段，缺失作者、卷期、页码或 DOI 时留空。
5. 用户 PDF 必须与公开元数据隔离，问答必须显示文件名、页码和原文证据。
6. 外部接口失败时给出中文错误和回退路径，不把失败包装成成功。

## 4. 设计决策

```text
用户研究问题
  -> 大模型优化检索词
  -> semantic_scholar_search(query, limit)
  -> 元数据工具结果
  -> 第二次模型调用生成研究现状
  -> GB/T 7714 风格引用草稿
  -> Semantic Scholar 失败时尝试 arXiv 元数据
```

用户 PDF 使用独立路径：

```text
用户上传 PDF
  -> 文件校验与 SHA-256
  -> data/user_papers/<SHA-256>/
  -> 独立 Chroma 集合 user_uploaded_papers
  -> 文件名 + 页码 + 原文证据
```

没有 API Key 时，公开检索仍可执行，但页面明确显示“规则整理，不是大模型 Function Calling”。

## 5. 对应实现

- `tools/semantic_scholar.py`
- `tools/arxiv_search.py`
- `tools/citation_formatter.py`
- `agents/literature_research_agent.py`
- `services/user_paper_rag.py`
- `researchmind/literature_research_ui.py`

## 6. 已执行的自动化验证

相关测试位于 `tests/test_literature_research.py`，覆盖：

- 工具参数校验、字段映射、DOI/题名去重、限流和超时。
- Agent 先优化检索词，再调用工具并整理综述。
- Semantic Scholar 失败时回退 arXiv。
- 无 Key 时不伪装 Function Calling。
- 用户 PDF 按内容去重、独立 Chroma 集合、页码证据和删除。
- 向量检索失败时回退 BM25。

## 7. 外部服务真实测试计划

1. 在可访问 Semantic Scholar 的网络中执行匿名检索，记录日期、查询词、HTTP 状态、耗时和返回条数。
2. 若返回 403 或 429，记录真实状态和请求时间，不把 arXiv 回退结果写成 Semantic Scholar 成功。
3. 配置 DeepSeek 或其他 OpenAI 兼容模型，执行一次真实 Function Calling，保存脱敏后的工具参数和响应结构。
4. 检查日志中不包含 Authorization 头、API Key 或完整用户问题中的隐私。
5. 用两篇有权使用的 PDF 演示本地 RAG，确认回答证据来自用户文件。

当前未完成项：评审网络真实 Semantic Scholar 成功响应、原始对话导出、RAG 页面录屏。

## 8. 验收命令

```powershell
cd "F:\体研智脑agant"
.\.venv\Scripts\python.exe -m pytest tests/test_literature_research.py -q
.\.venv\Scripts\python.exe -m streamlit run app.py
```
