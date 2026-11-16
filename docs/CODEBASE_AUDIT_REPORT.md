# 体研智脑 Agent 代码库审查报告

审查日期：2026-11-16

## 1. 审查结论

项目已具备可运行的完整科研辅助工作流，不是单一聊天页面。代码、配置、数据库、
测试和文档均已有较大规模实现。当前结构可以继续作为比赛提交基础，不需要重建项目。

审查时执行结果：

- Python 语法编译：通过。
- 自动化测试：181 passed，0 failed，7 warnings。
- `pip check`：无破损依赖。
- 项目虚拟环境启动：HTTP 200，健康检查返回 ok。
- Git 仓库：已初始化，保留 6 个已有提交。
- 敏感文件：`.env`、SQLite、PDF、虚拟环境和数据目录均被 Git 忽略；未发现被跟踪的真实密钥。

## 2. 实际技术栈与入口

| 类别 | 实际实现 |
| --- | --- |
| 前端 | Streamlit |
| 主入口 | `app.py` |
| 命令行入口 | `main.py` |
| 规划 Agent | `agents/planner.py` |
| Function Calling 文献 Agent | `agents/literature_research_agent.py` |
| 多源检索编排 | `researchmind/literature_retrieval.py` |
| ResearchMind 工作流 | `researchmind/workflow.py` |
| 阅读、争议、综合、测评、辅导 | `researchmind/agents.py` |
| SQLite 持久化 | `researchmind/storage.py` |
| PDF/证据工具 | `tools/pdf_reader.py`、`tools/document_chunker.py`、`tools/library_ingest.py` |
| RAG | `services/rag_service.py`、`services/vector_service.py`、`services/user_paper_rag.py` |
| Zotero | `researchmind/zotero_local_service.py`、`researchmind/zotero_service.py` |
| 知识图谱 | `researchmind/knowledge_map.py`、`knowledge_map_ui.py`、`knowledge_map_component.py` |

## 3. 真实调用关系

```text
Streamlit 页面
  -> ResearchMindWorkflow
    -> LiteratureCollector / MultiSourceRetriever
    -> PaperReader
    -> AcademicDebateAnalyzer
    -> KnowledgeSynthesizer
    -> LearningAssessmentAgent
    -> PersonalTutorAgent

公共问答
  -> LiteratureResearchAgent
    -> LLMService.chat_messages()
    -> MultiSourceRetriever
    -> Semantic Scholar / arXiv / OpenAlex / Crossref / PubMed

证据问答
  -> EvidenceRAGService
    -> BM25 + Chroma
    -> LLM 或抽取式摘录

论文库变化
  -> library_fingerprint()
  -> 派生分析失效
  -> 后台顺序重算
```

系统是“规则角色 + 确定性顺序编排 + 局部大模型工具调用”。除 Function Calling
文献检索、证据约束回答和主观题评分外，多数科研判断由可解释规则和持久化证据完成。
不得宣传为多个大模型自由协商或完全自主 Agent。

## 4. 已发现的问题

### P0：比赛提交阻断项

1. README 与文档曾使用旧快照；现已统一到当前 181 项测试、44 篇正式领域论文、
   3,296 条证据、138 节点和 1,220 条关系，旧数字仅作为历史差异说明保留。
2. 争议逻辑曾把“未达统计显著”归入反方；现已只在正向与明确负向同时出现时生成待审核候选。
3. 根目录安全审计会扫描本地 `.env`、测试临时数据库、PDF 和备份，不能直接对整个工作区导出；
   新打包脚本仅复制源码、文档、测试和许可允许再分发的内置测试语料。
4. 原提交包存在文档与代码分层问题；新版工具从同一根目录生成最终包，并要求在导出副本内复测。

### P1：工程规范问题

1. `.gitignore` 未覆盖 `.pytest-tmp-*`、`.sl-*.txt`、`research/test-temp/`、
   `research/backups/` 等运行产物。
2. README 篇幅过长，安装、真实能力、边界和比赛亮点不够集中。
3. 文档构建依赖 `reportlab`、`playwright` 未单独声明。
4. 缺少统一部署说明、API/模块接口说明、功能说明、演示脚本和比赛提交清单。
5. 核心文件较大，但调用链已可用；赛前贸然拆分模块会提高回归风险。

### P2：能力边界

1. 不支持扫描版 PDF OCR。
2. 未用 500 页以上真实 PDF 完成验收。
3. Zotero Web API 写入代码存在，但当前主路径是 Local API 读取/同步。
4. 只保存最近调度状态，不保存完整历史任务事件流。
5. 学术共识和争议是规则初筛，当前 7 题均待体育科研教师复核，不是正式 GRADE 或专家共识。
6. 当前研究关系图不是论文引用网络，版式联动翻译与完整写作 Copilot 尚未完成。
7. 当前为本机单用户架构，没有账号、租户、权限和公网限流。
8. 未配置模型时会使用明确标注的模拟回答；RAG 可退化为原文摘录。

## 5. 整改方案

1. 重写 README，以真实能力、安装、运行、数据结构、演示和边界为核心。
2. 保持现有模块路径，不移动已工作的业务代码。
3. 补充 `docs/DEPLOYMENT.md`、`docs/API.md`、`docs/FUNCTIONALITY.md`、
   `docs/ARCHITECTURE.md` 和 `demo/README.md`。
4. 增强 `.gitignore`，排除临时测试、服务器日志和证据生成中间文件。
5. 保留 `requirements.txt` 为最小运行依赖，新增可选文档/截图依赖清单。
6. 添加保守的竞赛审查许可证，不擅自授予商业或再分发权利。
7. 生成比赛提交检查清单和本审查报告。
8. 最终重新运行测试、依赖、导入、启动、安全和 README 命令检查。

## 6. 不在本轮执行

- 不移动 `researchmind/` 下现有模块。
- 不重构可运行的 Agent 工作流。
- 不修改论文数据库、PDF、测试记录和学习记录。
- 不创建虚假 Git 历史，不自动提交或推送远程仓库。
