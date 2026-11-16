# 体研智脑 Agent 部署说明

## 1. 操作系统与环境

已验证环境：

- Windows 11。
- Python 3.12.16。
- Streamlit 1.66.1。
- PyMuPDF 1.28.2。
- Chroma 1.5.9。

推荐 Python 3.11/3.12。不要让多个 Python 环境共用同一个虚拟环境目录。

## 2. 安装

```powershell
cd "<解压后的项目根目录>"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

可选文档与浏览器证据：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## 3. 环境变量

```powershell
Copy-Item .env.example .env
```

常用变量：

| 变量 | 作用 |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI 兼容模型密钥；空值进入明确降级模式 |
| `OPENAI_BASE_URL` | Chat Completions 兼容接口地址 |
| `OPENAI_MODEL` | 服务商模型名 |
| `RAG_RETRIEVAL_MODE` | `hybrid`、`semantic` 或 `bm25` |
| `MAX_PDF_SIZE_MB` | PDF 文件大小上限 |
| `MAX_PDF_PAGES` | 通用阅读页数上限 |
| `RESEARCHMIND_DB_PATH` | 旧库 SQLite 路径覆盖 |
| `ADOLESCENT_RESEARCHMIND_DB_PATH` | 青少年专题 SQLite 路径 |
| `ADOLESCENT_KNOWLEDGE_BASE_DIR` | 青少年专题知识库目录 |
| `ZOTERO_API_KEY` | 可选 Zotero Web API |
| `ZOTERO_LIBRARY_ID` | 可选 Zotero 用户/群组库 ID |

不要提交 `.env`，不要将密钥写入源码、截图或日志。

## 4. 数据库初始化

应用通过 SQLAlchemy 之外的原生 SQLite 仓储初始化数据库。当前实际实现为
`researchmind/storage.py`，首次创建领域时自动建表并执行增量迁移。不要手工删除
现有 SQLite 或 PDF。

默认路径：

```text
data/researchmind.sqlite3
data/adolescent_fitness/researchmind.sqlite3
knowledge_base/
knowledge_base_adolescent/
data/user_papers/
```

## 5. 启动

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

指定端口：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

健康检查：

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8502/_stcore/health
```

## 6. Zotero

### Local API

1. 启动 Zotero Desktop。
2. 启用 Local API，默认端点：`http://127.0.0.1:23119/api/users/0`。
3. 在“文献自动采集”中同步活动条目。
4. PDF 子附件可被页级解析；没有可解析 PDF 的记录标记为仅元数据。

### Web API

Web API 写入代码存在于 `researchmind/zotero_service.py`。当前比赛主流程只将
Local API 读取/同步作为已验证路径，Web API 写入需单独授权、测试和确认。

## 7. 向量索引

首次语义检索会加载 `intfloat/multilingual-e5-small`。网络不可用时：

- 文本和页级证据仍可用；
- 检索回退 BM25；
- 页面明确说明向量模式不可用；
- 不删除原始 PDF 或证据 JSON。

## 8. 常见问题

### 页面没有更新

确认浏览器打开的是当前 Streamlit 进程，必要时重启服务；不要同时启动多个旧版本。

### Zotero 连接失败

检查 Zotero 是否运行、Local API 是否开启、`127.0.0.1:23119` 是否可访问。

### 中文路径测试权限失败

使用 ASCII `--basetemp`：

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp C:\tmp\researchmind-tests
```

### 模型响应慢或失败

检查 `OPENAI_BASE_URL`、模型名和网络；页面会显示失败原因并回退到规则或原文摘录。
