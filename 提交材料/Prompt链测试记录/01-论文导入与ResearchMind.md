# Prompt 链记录一：论文导入与研究框架

## 1. 记录说明

- 记录性质：根据当前对话中真实出现的用户需求整理，不是原始聊天逐字稿。
- 对应功能：PDF 导入、去重、页级证据、九维研究框架、ResearchMind 论文库同步。
- 当前状态：代码和自动化测试已存在；原始对话截图或导出文件尚未归档。

## 2. 真实用户 Prompt

第一段：

> 帮我根据论文内容导入到项目里

第二段：

> 我需要通过论文创立一个框架加入到程序里并且以后添加新的论文都会自动的添加到框架中

随后用户进一步提出 ResearchMind 的六 Agent 学术知识发现与主动学习需求，其中包括论文采集、结构化阅读、观点比较、知识综合、测评和个性化辅导。

## 3. 人工约束

1. 不得编造论文、DOI、作者、页码或实验数据。
2. 每个观点必须尽量关联论文、原文证据和页码。
3. 新论文应通过统一的导入流程自动更新清单、证据索引、研究框架和论文库。
4. 外部来源只使用公开元数据或明确开放获取的 PDF。
5. 用户上传 PDF 只做本地 RAG，不冒充公开检索结果。
6. 没有 API Key 时仍需运行本地规则能力，并明确显示模拟或规则模式。
7. 不重写已经稳定的模块，先做兼容扩展。

## 4. 设计决策

- 使用 SHA-256 标识文件内容，避免同名覆盖和重复入库。
- 使用 PyMuPDF 提取页级文本，片段保留文件名和页码。
- `manifest.json` 保存书目清单，`evidence_chunks.json` 保存页级证据，`framework.json` 保存九维归类。
- ResearchMind 使用独立 SQLite 表保存领域、论文、证据、分析、试题、答题和掌握度。
- 采用显式顺序工作流，保留未来替换为 LangGraph 的接口边界。
- Chroma 只作为可重建的派生索引，损坏时不修改原始 PDF 和证据文件。

## 5. 对应实现

- `tools/pdf_reader.py`
- `tools/document_chunker.py`
- `tools/library_ingest.py`
- `tools/literature_framework.py`
- `researchmind/storage.py`
- `researchmind/workflow.py`
- `researchmind/agents.py`

## 6. 已执行的自动化验证

相关测试位于：

- `tests/test_basic.py`
- `tests/test_researchmind.py`

覆盖内容包括：

- 新 PDF 目录条目可以同步到 ResearchMind。
- 六阶段分析和测评结果具有可追溯来源。
- PDF 页码在切分后保持。
- 损坏或空 PDF 不会生成伪造证据。
- 论文库同步与 SQLite 持久化可重复执行。

最终整体测试结果以提交时实际执行的 `pytest` 输出为准。

## 7. 待补的真实测试证据

- [ ] 导出本组需求对应的原始 AI 对话。
- [ ] 用一份有权使用的 PDF 录制从上传到进入框架的完整页面过程。
- [ ] 记录导入前后论文数、证据数和框架条数。
- [ ] 截图展示新论文的题名、文件名和页码证据。

## 8. 验收命令

```powershell
cd "F:\体研智脑agant"
.\.venv\Scripts\python.exe -m pytest tests/test_basic.py tests/test_researchmind.py -q
.\.venv\Scripts\python.exe -m streamlit run app.py
```

页面验收时，不使用尚未获得公开许可的 PDF 作为公开演示素材。
