# 可核验导出记录说明

本目录由 `../export_records.py` 从正式专题 SQLite 重新生成。

- `papers-44.csv`：正式专题领域的 44 篇活动论文。
- `evidence-3296-core-domain.csv`：正式专题领域的 3,296 条页级证据。
- `papers-46-all-database.csv`：当前 SQLite 中两个领域的 46 篇活动论文。
- `evidence-3311-all-database.csv`：当前整个 SQLite 的 3,311 条页级证据。
- `knowledge-graph-nodes.csv`：当前知识图谱全部节点。
- `knowledge-graph-edges.csv`：当前知识图谱全部关系。
- `knowledge-graph-statistics.json`：节点、关系、重复项和无证据关系统计。
- `export-manifest.json`：源数据库哈希、导出文件哈希、行数和各分类计数。

证据 CSV 保留原文哈希和短预览，不导出受版权保护的完整正文。核验时可在本地数据库按
`domain_id` 与 `evidence_id` 查询原文并重新计算 UTF-8 字节的 SHA-256。3,311 是数据库总计数，
其中正式 44 篇专题领域为 3,296 条，另 15 条属于独立的 2 篇论文领域。
