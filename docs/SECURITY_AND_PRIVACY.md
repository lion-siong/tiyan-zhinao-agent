# 安全、隐私与参赛导出规范

本文档记录「体研智脑 Agent」的最低安全要求。目标不是把系统包装成生产级安全平台，而是确保竞赛仓库和导出包不泄露个人数据、密钥或未授权材料。

## 1. 当前风险结论

- 2026-11-16 仓库整理复核：`.env`、SQLite、PDF、用户数据、虚拟环境、日志和测试临时目录
  已加入 `.gitignore`；当前 Git 跟踪文件中未发现真实密钥。
- 项目过去曾在聊天中暴露过真实大模型 API Key。该 Key 必须立即在服务商控制台吊销并重新生成，不能只依赖删除本地文件。
- 根目录 `.env` 包含真实本地配置，只能保留在本机，不进入 Git、截图、录屏或提交压缩包。
- `data/researchmind.sqlite3` 含本机学习记录和答题历史，属于个人数据，默认不公开。
- `.tmp/` 与 `outputs/edge-profile-desktop/` 含浏览器缓存、Cookie 或登录资料的潜在风险，必须排除。
- `knowledge_base/chroma.corrupt-*` 是可重建的损坏索引备份，不属于参赛源码。
- `knowledge_base/pdfs/` 与 `knowledge_base/index/` 属于论文全文或派生证据。书目清单中记录了 CC BY 4.0 与 CC BY-NC-ND 4.0 等许可，但正式公开再分发前仍须逐篇核对版本、署名和许可条款。
- 当前工作区没有现成的参赛 ZIP/RAR 导出包；导出后必须用审计脚本重新检查。

## 2. 禁止进入 Git 与导出包的内容

- `.env`、`.streamlit/secrets.toml`、私钥、访问令牌和服务账号文件。
- SQLite、数据库日志、浏览器 Cookie、登录资料、历史记录和网页缓存。
- 用户上传 PDF、答题记录、学习报告、含个人信息的 CSV/XLSX。
- 真实 API 请求日志、包含 Authorization 头的错误日志和调试输出。
- 未确认公开许可的论文全文、受版权保护的批量摘录和证据索引。
- 临时截图、录屏草稿、浏览器 profile、模型缓存和 Chroma 派生数据。

## 3. 日常审计

对整个本地工作区执行审计会主动发现 `.env`、SQLite、PDF 和测试临时目录，因此
返回阻断是预期行为，不能为了“让审计变绿”而删除用户数据。正式提交时先生成不含
本地隐私数据的干净提交目录，再对该目录执行：

```powershell
.\.venv\Scripts\python.exe -m tools.security_audit ".\build\competition-source" --json
```

若需要把“需人工复核”也视为失败：

```powershell
.\.venv\Scripts\python.exe -m tools.security_audit . --fail-on-review
```

审计 ZIP 或 TAR：

```powershell
.\.venv\Scripts\python.exe -m tools.security_audit ".\提交材料\参赛导出包.zip"
```

工具不会打印密钥原文，只显示文件、规则和行号。输出中的 `BLOCK` 必须修复；`REVIEW` 必须由人工确认版权、隐私和公开范围。

## 4. Git 提交前检查

```powershell
git status --short
git diff --cached --name-only
git check-ignore -v .env data/researchmind.sqlite3 knowledge_base/pdfs/example.pdf
```

只有满足以下条件才能提交：

1. 暂存清单中没有 `.env`、数据库、日志、压缩包、浏览器资料和个人学习数据。
2. 新增 PDF 或证据文件已经完成许可审核，并记录来源与署名方式。
3. 示例配置只使用 `.env.example` 中的空占位符。
4. 测试和启动命令使用真实执行结果，不把未运行的检查写成“已通过”。

## 5. 密钥泄露处置

1. 在服务商控制台吊销旧 Key，不要继续使用。
2. 生成新 Key，只写入本机 `.env`。
3. 若 Key 曾进入 Git 历史，必须同时清理历史并再次轮换；只删除当前文件不足以消除风险。
4. 检索日志与导出包，确认没有 Authorization 头、Key 片段或请求截图。
5. 重新运行安全审计和完整测试。

## 6. 公网演示边界

当前应用没有账号、租户隔离和权限审计，默认适合本机单用户演示。若部署到公网，应使用隔离数据库、禁用真实用户数据、关闭本地密钥，并增加身份认证、请求限流、上传隔离和操作审计。
