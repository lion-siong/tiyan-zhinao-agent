# 体研智脑文献库

本目录保存用户提供的真实论文 PDF、书目信息、按页切分的证据索引，以及由论文
证据自动汇总出的研究框架。PDF 正文只作为科研资料使用，不会被当作程序指令执行。

## 当前研究框架

**社会生态视角下大学生体育参与与体质健康研究框架**

系统把当前文献证据组织为九个维度：

1. 研究背景与问题
2. 研究对象与场景
3. 研究设计与方法
4. 个体因素
5. 家庭因素
6. 学校因素
7. 社会因素
8. 行为与健康结果
9. 干预对策

每个维度都保留论文题名、文件名、PDF 页码、片段 ID、原文片段和匹配分数。
同一段原文可以同时服务多个维度，引用前必须回到论文核对上下文。

## 当前收录论文

当前共收录 12 篇真实论文，其中原有 2 篇、为“近三年文献补充”新增 10 篇。
收录年份为 2023-2026 年，均提供可核验 DOI；除王晨等人的论文采用
CC BY-NC-ND 4.0 外，其余论文采用 CC BY 4.0 许可。

| 年份 | 本地文件 | 论文 | 作者 | 来源 | 许可 | DOI |
| --- | --- | --- | --- | --- | --- | --- |
| 2023 | `pdfs/huang-2023-social-ecological.pdf` | Validation of Physical Activity Correlates Questionnaire from Social Ecological Model in College Students | Meirong Huang 等 | Journal of Clinical Medicine, 12(3): 777 | CC BY 4.0 | `10.3390/jcm12030777` |
| 2023 | `pdfs/sheng-2023-peer-support.pdf` | Exercise health belief model mediates the relationship between physical activity and peer support among Chinese college students: A cross-sectional survey | Jiazhi Sheng 等 | Frontiers in Psychology, 14: 1103109 | CC BY 4.0 | `10.3389/fpsyg.2023.1103109` |
| 2024 | `pdfs/zhang-2024-health-beliefs.pdf` | Effects of physical health beliefs on college students’ physical exercise behavior intention: mediating effects of exercise imagery | Li Zhang 等 | BMC Psychology, 12(1): 99 | CC BY 4.0 | `10.1186/s40359-024-01558-3` |
| 2024 | `pdfs/wang-2024-blended-pe.pdf` | Effects of blended learning in physical education on university students’ exercise attitudes and basketball skills: a cluster randomized controlled trial | Chen Wang 等 | BMC Public Health, 24(1): 3170 | CC BY-NC-ND 4.0 | `10.1186/s12889-024-20469-x` |
| 2024 | `pdfs/niu-2024-taichi-fitness.pdf` | Effects of Bafa Wubu and He-Style Tai Chi exercise training on physical fitness of overweight male university students: A randomized controlled trial | Yantao Niu 等 | PLoS ONE, 19(1): e0297117 | CC BY 4.0 | `10.1371/journal.pone.0297117` |
| 2025 | `pdfs/zhang-2025-predictors-pa.pdf` | Random forest-based identification and ranking of predictive factors for physical activity in Chinese college students | Ding-you Zhang 等 | Frontiers in Public Health, 13: 1699606 | CC BY 4.0 | `10.3389/fpubh.2025.1699606` |
| 2025 | `pdfs/sun-2025-aerobic-fitness.pdf` | Comparative effectiveness of morning and evening aerobic exercise on weight loss and physical fitness in Chinese college students with overweight and obesity | Jingyu Sun 等 | BMC Sports Science, Medicine and Rehabilitation, 17(1): 101 | CC BY 4.0 | `10.1186/s13102-025-01149-8` |
| 2025 | `pdfs/gao-2025-exercise-self-efficacy.pdf` | Correlational research on college students’ physical exercise behavior, academic engagement, and self-efficacy | Wei Gao 等 | Frontiers in Psychology, 16: 1428365 | CC BY 4.0 | `10.3389/fpsyg.2025.1428365` |
| 2025 | `pdfs/xiong-2025-social-support.pdf` | The role of social support and psychological capital in college students' physical exercise behavior: a mediation analysis | You Xiong 等 | Frontiers in Psychology, 16: 1601079 | CC BY 4.0 | `10.3389/fpsyg.2025.1601079` |
| 2026 | `pdfs/ass_2401349.pdf` | 大学生体育锻炼行为的影响因素研究 | 夏慧琴 | 社会科学前沿, 15(7): 533-546 | CC BY 4.0 | `10.12677/ass.2026.157601` |
| 2026 | `pdfs/aps_3081402.pdf` | 大学生体质健康影响因素与提升对策研究 | 彭玲、王俊 | 体育科学进展, 14(3): 464-471 | CC BY 4.0 | `10.12677/aps.2026.143067` |
| 2026 | `pdfs/liang-2026-app-pe.pdf` | App-supported versus conventional college physical education: Effects on standardized physical fitness scores and exercise motivation in Chinese university students | Xilin Liang、Zenan Wang | PLoS ONE, 21(3): e0345759 | CC BY 4.0 | `10.1371/journal.pone.0345759` |

系统只保存原始 PDF、经出版方页面核验的书目信息和原文片段，不自动编造摘要、
统计数据、参考文献或研究结论。文献进入项目后会自动重建证据索引和九维研究
框架；新增论文仍应检查自动识别的书目字段。

## 文件作用

```text
knowledge_base/
├── pdfs/                      原始论文 PDF
├── manifest.json              书目信息、PDF 哈希和页数
├── index/
│   └── evidence_chunks.json   按页切分的可检索原文证据
├── framework.json             九维研究框架和证据映射
└── README.md                  本说明
```

- `manifest.json`：记录文献题名、作者、期刊、年份、卷期、页码、DOI、页数和
  SHA-256。SHA-256 用于判断 PDF 是否重复或发生变化。
- `evidence_chunks.json`：保存从 PDF 提取的原文片段。每个片段都有文件名和页码，
  供证据问答和研究框架共同使用。
- `framework.json`：保存九个研究维度、研究逻辑链和每个维度下的真实原文证据。

这三个文件由系统维护，初学者不需要手工计算哈希或编辑 JSON。

## 添加新论文

### 方法一：在网页中上传（推荐）

1. 启动项目并打开“文献框架”页面。
2. 在“添加论文并自动更新框架”区域选择一篇或多篇 PDF。
3. 点击“导入论文并更新框架”。

系统会自动完成：

1. 检查空文件、文件大小、加密和损坏的 PDF；
2. 用 SHA-256 跳过重复论文；
3. 尝试识别题名、作者、期刊、年份、卷期、页码和 DOI；
4. 将 PDF 永久保存到 `pdfs/`；
5. 更新 `manifest.json` 并重建 `evidence_chunks.json`；
6. 根据当前全部论文重建 `framework.json`。

如果导入失败，系统会恢复导入前的文献清单、证据索引和研究框架，避免留下
半成品数据。

### 方法二：放入文献目录后扫描

1. 把 PDF 直接复制到 `knowledge_base/pdfs/`。
2. 打开“文献框架”页面。
3. 点击“扫描文献目录”。

页面第一次打开时也会自动扫描一次该目录。以后新增论文时，建议仍点击一次
“扫描文献目录”并确认导入结果。

## 自动归类的边界

- 当前归类使用固定关键词和本地 BM25 检索，属于**规则匹配**，不是大模型综述。
- 自动识别不到的字段会标记为“待核对”或“待补充”，系统不会猜测真实信息。
- 扫描版 PDF 没有可提取文字，当前版本不支持 OCR，会提示未导入。
- 新论文会进入后续生成的研究框架，但关键词匹配可能出现误归类，必须人工核对。
- 当前方案适合竞赛演示和小规模文献；文献明显增多后再考虑向量检索或文献管理平台。

只导入自己有权使用、非涉密且不含个人隐私的论文。不要把模拟数据或虚构文献
放入本目录。

## 命令行检查

在项目根目录执行：

```powershell
# 检查当前文献库
.\.venv\Scripts\python.exe -m tools.knowledge_base

# 重新读取已登记 PDF，重建证据索引
.\.venv\Scripts\python.exe -m tools.knowledge_base --build

# 从当前文献重新生成研究框架
.\.venv\Scripts\python.exe -c "from tools.library_ingest import rebuild_library; b, f = rebuild_library(); print(f'{f.documents_count} 篇论文，{len(f.dimensions)} 个维度，{f.evidence_count} 条证据')"
```

检查框架文件是否与当前清单和索引一致：

```powershell
.\.venv\Scripts\python.exe -c "from tools.literature_framework import load_literature_framework; f=load_literature_framework(); print(f.documents_count, len(f.dimensions), f.evidence_count)"
```
