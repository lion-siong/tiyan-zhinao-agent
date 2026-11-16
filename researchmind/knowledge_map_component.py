"""ResearchMind 五层知识图谱的 Streamlit Custom Components v2 渲染器。

组件只使用 Python 已经整理好的真实节点、论文元数据和原文证据。层级展开
由浏览器完成，因此打开页面时不会把全部论文节点一次性渲染成密集关系网。
"""

from __future__ import annotations

from typing import Any

import streamlit as st


GRAPH_HTML = """
<section class="kg-shell" aria-label="五层知识图谱">
  <header class="kg-head">
    <div class="kg-title">
      <span class="kg-kicker">EVIDENCE MAP</span>
      <h2>五层知识图谱</h2>
      <p>研究领域 → 主题 → 变量与问题 → 论文证据 → 共识、争议与研究空白。</p>
    </div>
    <div class="kg-tools" role="toolbar" aria-label="知识地图工具">
      <label class="kg-field">
        <span>搜索</span>
        <input id="kg-search" type="search" placeholder="主题、研究内容或论文" autocomplete="off" />
      </label>
      <label class="kg-field">
        <span>主题筛选</span>
        <select id="kg-filter"></select>
      </label>
      <label class="kg-field kg-compact-field">
        <span>论文年份</span>
        <select id="kg-year"></select>
      </label>
      <label class="kg-field kg-compact-field">
        <span>证据类型</span>
        <select id="kg-evidence"></select>
      </label>
      <label class="kg-field kg-compact-field">
        <span>研究方法</span>
        <select id="kg-method"></select>
      </label>
      <div class="kg-modes" role="group" aria-label="图谱模式">
        <button type="button" data-mode="tree" class="active">层级树</button>
        <button type="button" data-mode="network">关系网络</button>
      </div>
      <label class="kg-relation" title="仅显示有原文证据的主题关系">
        <input id="kg-relations" type="checkbox" />
        <span>显示学术关系</span>
      </label>
      <div class="kg-type-toggles" aria-label="节点类型显示">
        <label><input id="kg-show-variables" type="checkbox" checked /><span>变量</span></label>
        <label><input id="kg-show-papers" type="checkbox" checked /><span>论文证据</span></label>
        <label><input id="kg-show-synthesis" type="checkbox" checked /><span>共识争议</span></label>
      </div>
      <div class="kg-actions">
        <button id="kg-overview" type="button" title="返回图谱总览">总览</button>
        <button id="kg-reset" type="button" title="重置筛选与视图">重置</button>
        <button id="kg-back" type="button" title="返回上一级">返回</button>
        <button id="kg-collapse" type="button" title="折叠全部分支">折叠</button>
        <button id="kg-fit" type="button" title="图谱居中">居中</button>
        <button id="kg-zoom-out" type="button" aria-label="缩小">−</button>
        <button id="kg-zoom-in" type="button" aria-label="放大">+</button>
        <button id="kg-fullscreen" type="button" title="全屏">全屏</button>
      </div>
    </div>
  </header>
  <div class="kg-stage" id="kg-stage">
    <svg id="kg-svg" role="img" aria-label="核心领域、主题、变量与问题、论文证据及共识争议的五层关系图"></svg>
    <div class="kg-breadcrumb" id="kg-breadcrumb" aria-live="polite"></div>
    <div class="kg-empty" id="kg-empty" hidden>
      <strong>没有匹配的图谱节点</strong>
      <span>请清除搜索词或选择其他研究主题。</span>
    </div>
    <aside class="kg-detail" id="kg-detail" hidden aria-live="polite">
      <div class="kg-detail-head">
        <div>
          <span class="kg-detail-type" id="kg-detail-type"></span>
          <h3 id="kg-detail-title"></h3>
        </div>
        <button id="kg-detail-close" type="button" aria-label="关闭详情">×</button>
      </div>
      <div class="kg-detail-body" id="kg-detail-body"></div>
    </aside>
  </div>
  <footer class="kg-foot">
    <div class="kg-legend">
      <span><i class="core"></i>核心方向</span>
      <span><i class="theme"></i>研究主题</span>
      <span><i class="direction"></i>研究方向</span>
      <span><i class="variable"></i>研究变量</span>
      <span><i class="paper"></i>论文证据</span>
      <span><i class="consensus"></i>共识</span>
      <span><i class="controversy"></i>争议</span>
      <span><i class="gap"></i>空白</span>
      <span><i class="relation"></i>有证据的研究关系</span>
      <span><i class="status-positive"></i>正向</span>
      <span><i class="status-negative"></i>负向</span>
      <span><i class="status-null"></i>无显著</span>
      <span><i class="status-inconsistent"></i>结果不一致</span>
    </div>
    <span id="kg-status" class="kg-status"></span>
  </footer>
</section>
"""


GRAPH_CSS = """
:host {
  color: var(--st-text-color, #172033);
  font-family: var(--st-font, "Microsoft YaHei", "Noto Sans SC", sans-serif);
}
* { box-sizing: border-box; }
.kg-shell {
  overflow: hidden;
  border: 1px solid var(--st-border-color, #dbe3ee);
  border-radius: 12px;
  background: var(--st-secondary-background-color, #ffffff);
}
.kg-head {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  justify-content: flex-start;
  gap: 22px;
  padding: 18px 20px 16px;
  border-bottom: 1px solid var(--st-border-color, #dbe3ee);
}
.kg-kicker {
  display: block;
  margin-bottom: 4px;
  color: var(--st-primary-color, #2457d6);
  font-size: 10px;
  font-weight: 760;
  letter-spacing: 0.14em;
}
.kg-title h2 {
  margin: 0;
  color: var(--st-text-color, #172033);
  font-size: 20px;
  line-height: 1.3;
}
.kg-title p {
  margin: 5px 0 0;
  color: var(--st-gray-color, #65758b);
  font-size: 12px;
  line-height: 1.55;
}
.kg-tools {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-end;
  justify-content: flex-start;
  gap: 8px;
}
.kg-field { display: grid; gap: 4px; min-width: 152px; }
.kg-compact-field { min-width: 108px; }
.kg-field span {
  color: var(--st-gray-color, #65758b);
  font-size: 10px;
  font-weight: 680;
}
.kg-field input,
.kg-field select,
.kg-actions button,
.kg-modes button,
.kg-relation {
  height: 34px;
  border: 1px solid var(--st-border-color, #dbe3ee);
  border-radius: 7px;
  background: var(--st-background-color, #f7f9fc);
  color: var(--st-text-color, #172033);
  font: inherit;
  font-size: 11px;
}
.kg-field input,
.kg-field select { width: 100%; padding: 0 10px; outline: none; }
.kg-field input:focus,
.kg-field select:focus {
  border-color: var(--st-primary-color, #2457d6);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--st-primary-color, #2457d6) 14%, transparent);
}
.kg-modes { display: inline-flex; gap: 3px; padding: 3px; border: 1px solid var(--st-border-color, #dbe3ee); border-radius: 8px; }
.kg-modes button { height: 28px; padding: 0 9px; border: 0; background: transparent; cursor: pointer; }
.kg-modes button.active { background: var(--st-primary-color, #2457d6); color: #fff; }
.kg-relation { display: inline-flex; align-items: center; gap: 6px; padding: 0 9px; cursor: pointer; }
.kg-relation input { accent-color: var(--st-primary-color, #2457d6); }
.kg-type-toggles {
  display: inline-flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}
.kg-type-toggles label {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  height: 34px;
  padding: 0 8px;
  border: 1px solid var(--st-border-color, #dbe3ee);
  border-radius: 7px;
  background: var(--st-background-color, #f7f9fc);
  color: var(--st-text-color, #172033);
  font-size: 10px;
  cursor: pointer;
}
.kg-type-toggles input { accent-color: var(--st-primary-color, #2457d6); }
.kg-actions { display: inline-flex; gap: 5px; }
.kg-actions button { min-width: 34px; padding: 0 9px; font-weight: 650; cursor: pointer; }
.kg-actions button:hover,
.kg-actions button:focus-visible,
.kg-modes button:focus-visible {
  border-color: var(--st-primary-color, #2457d6);
  color: var(--st-primary-color, #2457d6);
  outline: none;
}
.kg-stage {
  position: relative;
  min-height: 760px;
  overflow: hidden;
  background-color: var(--st-secondary-background-color, #fff);
  background-image: radial-gradient(circle at 1px 1px, color-mix(in srgb, var(--st-gray-color, #65758b) 18%, transparent) 1px, transparent 0);
  background-size: 22px 22px;
}
#kg-svg {
  display: block;
  width: 100%;
  height: 760px;
  touch-action: none;
  cursor: grab;
}
#kg-svg.dragging { cursor: grabbing; }
.kg-edge {
  fill: none;
  stroke: color-mix(in srgb, var(--st-gray-color, #65758b) 44%, transparent);
  stroke-width: 1.4;
}
.kg-edge.relation {
  stroke: color-mix(in srgb, var(--st-orange-color, #a55a0a) 66%, transparent);
  stroke-dasharray: 5 5;
}
.kg-edge.research_relation {
  stroke-width: 1.7;
  stroke-dasharray: 8 5;
}
.kg-edge.question_variable {
  stroke: color-mix(in srgb, #6b4fc8 58%, transparent);
  stroke-dasharray: 4 4;
}
.kg-edge.synthesis_relation {
  stroke: #8b6f47;
  stroke-dasharray: 7 4;
  stroke-width: 1.5;
}
.kg-edge.evidence_synthesis {
  stroke: color-mix(in srgb, var(--st-green-color, #147d64) 54%, transparent);
  stroke-dasharray: 2 4;
}
.kg-edge.status-positive { stroke: color-mix(in srgb, var(--st-green-color, #147d64) 82%, transparent); }
.kg-edge.status-negative { stroke: #c2414b; }
.kg-edge.status-null { stroke: #6b7280; }
.kg-edge.status-conditional { stroke: #9a6700; }
.kg-edge.status-inconsistent { stroke: #c2410c; stroke-dasharray: 3 4; }
.kg-edge.status-insufficient { stroke: #8b95a5; stroke-dasharray: 2 5; }
.kg-edge.active {
  stroke: var(--st-primary-color, #2457d6);
  stroke-width: 2.6;
}
.kg-edge.muted { opacity: 0.22; }
.kg-node { cursor: pointer; outline: none; }
.kg-node rect {
  stroke-width: 1.25;
  filter: drop-shadow(0 4px 8px rgba(18, 31, 51, 0.07));
  transition: stroke-width 120ms ease, filter 120ms ease;
}
.kg-node:hover rect,
.kg-node:focus rect,
.kg-node.selected rect {
  stroke-width: 2.6;
  filter: drop-shadow(0 7px 13px rgba(18, 31, 51, 0.13));
}
.kg-node.level-1 rect { fill: var(--st-primary-color, #2457d6); stroke: var(--st-primary-color, #2457d6); }
.kg-node.level-2 rect { fill: color-mix(in srgb, var(--st-green-color, #147d64) 9%, var(--st-secondary-background-color, #fff)); stroke: color-mix(in srgb, var(--st-green-color, #147d64) 52%, transparent); }
.kg-node.level-3 rect { fill: var(--st-secondary-background-color, #fff); stroke: color-mix(in srgb, var(--st-gray-color, #65758b) 46%, transparent); }
.kg-node.node-direction rect { fill: var(--st-secondary-background-color, #fff); stroke: color-mix(in srgb, var(--st-primary-color, #2457d6) 42%, transparent); }
.kg-node.node-pending-direction rect { stroke-dasharray: 4 3; stroke: color-mix(in srgb, var(--st-gray-color, #65758b) 58%, transparent); }
.kg-node.node-variable rect { fill: color-mix(in srgb, #6b4fc8 10%, var(--st-secondary-background-color, #fff)); stroke: color-mix(in srgb, #6b4fc8 55%, transparent); }
.kg-node.node-paper rect { fill: color-mix(in srgb, var(--st-orange-color, #a55a0a) 8%, var(--st-secondary-background-color, #fff)); stroke: color-mix(in srgb, var(--st-orange-color, #a55a0a) 48%, transparent); }
.kg-node.node-consensus rect { fill: color-mix(in srgb, var(--st-green-color, #147d64) 12%, var(--st-secondary-background-color, #fff)); stroke: color-mix(in srgb, var(--st-green-color, #147d64) 62%, transparent); }
.kg-node.node-controversy rect { fill: color-mix(in srgb, #c2414b 10%, var(--st-secondary-background-color, #fff)); stroke: color-mix(in srgb, #c2414b 58%, transparent); }
.kg-node.node-gap rect { fill: color-mix(in srgb, #8b6f47 7%, var(--st-secondary-background-color, #fff)); stroke: #8b6f47; stroke-dasharray: 5 3; }
.kg-node.is-filtered-out { display: none; }
.kg-node .kg-foreign { overflow: hidden; pointer-events: none; }
.kg-node .kg-label {
  display: -webkit-box;
  overflow: hidden;
  color: var(--st-text-color, #172033);
  font-size: 12px;
  font-weight: 720;
  line-height: 1.35;
  text-overflow: ellipsis;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}
.kg-node .kg-meta {
  display: block;
  margin-top: 4px;
  color: var(--st-gray-color, #65758b);
  font-size: 9.5px;
  font-weight: 550;
  white-space: nowrap;
}
.kg-node.level-1 .kg-label { color: #fff; font-size: 14px; }
.kg-node.level-1 .kg-meta { color: rgba(255, 255, 255, 0.78); }
.kg-breadcrumb {
  position: absolute;
  top: 12px;
  left: 14px;
  z-index: 3;
  max-width: calc(100% - 28px);
  padding: 7px 10px;
  border: 1px solid var(--st-border-color, #dbe3ee);
  border-radius: 7px;
  background: color-mix(in srgb, var(--st-secondary-background-color, #fff) 94%, transparent);
  color: var(--st-gray-color, #65758b);
  font-size: 10px;
  line-height: 1.4;
  backdrop-filter: blur(8px);
}
.kg-empty {
  position: absolute;
  inset: 80px 24px 24px;
  display: grid;
  place-content: center;
  gap: 5px;
  border: 1px dashed var(--st-border-color, #dbe3ee);
  text-align: center;
}
.kg-empty[hidden] { display: none; }
.kg-empty strong { font-size: 14px; }
.kg-empty span { color: var(--st-gray-color, #65758b); font-size: 11px; }
.kg-detail {
  position: absolute;
  top: 14px;
  right: 14px;
  z-index: 5;
  width: min(420px, calc(100% - 28px));
  max-height: calc(100% - 28px);
  overflow: auto;
  padding: 16px;
  border: 1px solid var(--st-border-color, #dbe3ee);
  border-radius: 10px;
  background: var(--st-secondary-background-color, #fff);
  box-shadow: 0 18px 46px rgba(18, 31, 51, 0.2);
}
.kg-detail[hidden] { display: none; }
.kg-detail-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
.kg-detail-type { color: var(--st-primary-color, #2457d6); font-size: 9px; font-weight: 760; letter-spacing: 0.1em; }
.kg-detail h3 { margin: 4px 0 0; font-size: 15px; line-height: 1.45; }
.kg-detail-head button { width: 28px; height: 28px; border: 1px solid var(--st-border-color, #dbe3ee); border-radius: 7px; background: transparent; color: inherit; cursor: pointer; }
.kg-detail-body { display: grid; gap: 10px; margin-top: 13px; }
.kg-summary { margin: 0; color: var(--st-gray-color, #65758b); font-size: 11px; line-height: 1.65; }
.kg-facts { display: grid; gap: 6px; padding: 10px; border-radius: 8px; background: var(--st-background-color, #f7f9fc); }
.kg-fact { display: grid; grid-template-columns: 78px 1fr; gap: 8px; font-size: 10px; line-height: 1.55; }
.kg-fact b { color: var(--st-text-color, #172033); }
.kg-fact span { color: var(--st-gray-color, #65758b); overflow-wrap: anywhere; }
.kg-evidence { padding: 9px 10px; border-left: 3px solid color-mix(in srgb, var(--st-primary-color, #2457d6) 44%, transparent); background: var(--st-background-color, #f7f9fc); }
.kg-evidence strong { display: block; margin-bottom: 5px; font-size: 10px; }
.kg-evidence p { margin: 0; color: var(--st-text-color, #172033); font-size: 10px; line-height: 1.62; white-space: pre-wrap; }
.kg-evidence small { display: block; margin-top: 5px; color: var(--st-gray-color, #65758b); font-size: 9px; }
.kg-paper-card,
.kg-relation-card {
  display: grid;
  gap: 4px;
  padding: 9px 10px;
  border: 1px solid var(--st-border-color, #dbe3ee);
  border-radius: 8px;
  background: var(--st-background-color, #f7f9fc);
}
.kg-paper-card strong,
.kg-relation-card strong { color: var(--st-text-color, #172033); font-size: 10px; line-height: 1.5; }
.kg-paper-card span,
.kg-relation-card span { color: var(--st-gray-color, #65758b); font-size: 9px; line-height: 1.5; }
.kg-link { color: var(--st-primary-color, #2457d6); font-size: 9px; font-weight: 680; text-decoration: none; }
.kg-link:hover { text-decoration: underline; }
.kg-foot { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 9px 16px; border-top: 1px solid var(--st-border-color, #dbe3ee); background: var(--st-background-color, #f7f9fc); }
.kg-legend { display: flex; flex-wrap: wrap; gap: 12px; color: var(--st-gray-color, #65758b); font-size: 9px; }
.kg-legend span { display: inline-flex; align-items: center; gap: 5px; }
.kg-legend i { width: 7px; height: 7px; border-radius: 50%; }
.kg-legend .core { background: var(--st-primary-color, #2457d6); }
.kg-legend .theme { background: var(--st-green-color, #147d64); }
.kg-legend .content { background: var(--st-gray-color, #65758b); }
.kg-legend .direction { background: var(--st-primary-color, #2457d6); }
.kg-legend .variable { background: #6b4fc8; }
.kg-legend .paper { background: var(--st-orange-color, #a55a0a); }
.kg-legend .consensus { background: var(--st-green-color, #147d64); }
.kg-legend .controversy { background: #c2414b; }
.kg-legend .gap { background: #8b6f47; }
.kg-legend .relation { width: 13px; height: 2px; border-radius: 0; background: var(--st-orange-color, #a55a0a); }
.kg-legend .status-positive { background: var(--st-green-color, #147d64); }
.kg-legend .status-negative { background: #c2414b; }
.kg-legend .status-null { background: #6b7280; }
.kg-legend .status-inconsistent { background: #c2410c; }
.kg-status { flex: 0 0 auto; color: var(--st-gray-color, #65758b); font-size: 9px; }
.kg-shell:fullscreen {
  width: 100vw;
  height: 100vh;
  border-radius: 0;
  background: var(--st-secondary-background-color, #fff);
}
.kg-shell:fullscreen .kg-stage { height: calc(100vh - 150px); min-height: 0; }
.kg-shell:fullscreen #kg-svg { height: calc(100vh - 150px); min-height: 0; }
@media (max-width: 1050px) {
  .kg-head { align-items: stretch; flex-direction: column; }
  .kg-tools { justify-content: flex-start; }
  .kg-field { flex: 1 1 180px; }
}
"""


GRAPH_JS = r"""
export default function ({ data, parentElement }) {
  const shell = parentElement.querySelector(".kg-shell")
  if (!shell || !data) return

  const NS = "http://www.w3.org/2000/svg"
  const svg = parentElement.querySelector("#kg-svg")
  const stage = parentElement.querySelector("#kg-stage")
  const detail = parentElement.querySelector("#kg-detail")
  const detailType = parentElement.querySelector("#kg-detail-type")
  const detailTitle = parentElement.querySelector("#kg-detail-title")
  const detailBody = parentElement.querySelector("#kg-detail-body")
  const empty = parentElement.querySelector("#kg-empty")
  const breadcrumb = parentElement.querySelector("#kg-breadcrumb")
  const status = parentElement.querySelector("#kg-status")
  const searchInput = parentElement.querySelector("#kg-search")
  const themeFilter = parentElement.querySelector("#kg-filter")
  const yearFilter = parentElement.querySelector("#kg-year")
  const evidenceFilter = parentElement.querySelector("#kg-evidence")
  const methodFilter = parentElement.querySelector("#kg-method")
  const relationToggle = parentElement.querySelector("#kg-relations")
  const showVariables = parentElement.querySelector("#kg-show-variables")
  const showPapers = parentElement.querySelector("#kg-show-papers")
  const showSynthesis = parentElement.querySelector("#kg-show-synthesis")

  const allNodes = Array.isArray(data.all_nodes) ? data.all_nodes : (data.nodes || [])
  const allEdges = Array.isArray(data.all_edges) ? data.all_edges : (data.edges || [])
  const allPapers = Array.isArray(data.all_papers) ? data.all_papers : (data.papers || [])
  const themes = Array.isArray(data.themes) ? data.themes : []
  const nodeById = new Map(allNodes.map((node) => [node.id, node]))
  const paperByNodeId = new Map(allPapers.map((paper) => [paper.node_id, paper]))
  const themeById = new Map(themes.map((theme) => [theme.id, theme]))
  const children = new Map()
  for (const node of allNodes) {
    for (const parentId of node.parent_ids || []) {
      if (!children.has(parentId)) children.set(parentId, [])
      children.get(parentId).push(node)
    }
  }

  let expandedTheme = ""
  let expandedDirection = ""
  let focusedVariable = ""
  let focusedPaper = ""
  let selectedId = data.core_id
  let mode = "tree"
  let zoom = 1
  let panX = 0
  let panY = 0
  let dragging = false
  let dragStart = null
  let lastPositions = new Map()
  let currentVisible = new Set()

  const viewport = document.createElementNS(NS, "g")
  svg.appendChild(viewport)

  function create(tag, attrs = {}) {
    const element = document.createElementNS(NS, tag)
    for (const [key, value] of Object.entries(attrs)) {
      element.setAttribute(key, String(value))
    }
    return element
  }

  function option(value, label) {
    const item = document.createElement("option")
    item.value = value
    item.textContent = label
    return item
  }

  themeFilter.replaceChildren(option("all", "全部研究主题"))
  for (const theme of themes) {
    themeFilter.appendChild(option(theme.id, `${theme.label}（${theme.paper_count}）`))
  }
  yearFilter.replaceChildren(option("all", "全部年份"))
  for (const year of (data.years || [])) {
    yearFilter.appendChild(option(String(year), String(year)))
  }
  evidenceFilter.replaceChildren(option("all", "全部证据"))
  for (const evidenceType of (data.evidence_types || [])) {
    evidenceFilter.appendChild(option(evidenceType, evidenceType))
  }
  methodFilter.replaceChildren(option("all", "全部方法"))
  for (const method of (data.methods || [])) {
    methodFilter.appendChild(option(method, method))
  }

  function ancestors(nodeId) {
    const result = []
    const queue = [...(nodeById.get(nodeId)?.parent_ids || [])]
    const seen = new Set()
    while (queue.length) {
      const id = queue.shift()
      if (seen.has(id)) continue
      seen.add(id)
      const node = nodeById.get(id)
      if (!node) continue
      result.push(node.id)
      queue.push(...(node.parent_ids || []))
    }
    return result
  }

  function childrenOf(nodeId, level) {
    return (children.get(nodeId) || []).filter((node) => !level || node.level === level)
  }

  function paperMatchesFilters(paperId) {
    const paper = paperByNodeId.get(`paper:${paperId}`)
    if (!paper) return false
    const year = yearFilter.value
    const evidenceType = evidenceFilter.value
    const method = methodFilter.value
    if (year !== "all" && String(paper.year) !== year) return false
    if (evidenceType !== "all" && !(paper.evidence_types || []).includes(evidenceType)) return false
    if (method !== "all" && paper.method_category !== method) return false
    return true
  }

  function nodeMatchesFilters(node) {
    if (node.id === data.core_id) return true
    if (node.type === "研究变量" && !showVariables.checked) return false
    if (node.type === "论文" && !showPapers.checked) return false
    if (node.level === 5 && !showSynthesis.checked) return false
    const paperIds = node.paper_ids || []
    if (yearFilter.value !== "all" && !paperIds.some(paperMatchesFilters)) return false
    if (
      evidenceFilter.value !== "all"
      && !(node.evidence_types || []).includes(evidenceFilter.value)
    ) return false
    return true
  }

  function directionTheme(directionId) {
    return (nodeById.get(directionId)?.parent_ids || []).find(
      (id) => nodeById.get(id)?.level === 2
    ) || ""
  }

  function searchMatches(query) {
    const needle = query.trim().toLocaleLowerCase("zh-CN")
    if (!needle) return null
    let matchedNode = null
    for (const node of allNodes) {
      const text = `${node.label} ${node.description || ""} ${node.type || ""}`
        .toLocaleLowerCase("zh-CN")
      if (!text.includes(needle)) continue
      matchedNode = node
      break
    }
    if (!matchedNode) {
      for (const paper of allPapers) {
        const text = [
          paper.title,
          paper.authors,
          paper.year,
          paper.doi,
          paper.journal,
          paper.abstract,
          paper.method_summary,
          paper.conclusion_summary,
        ].join(" ").toLocaleLowerCase("zh-CN")
        if (!text.includes(needle)) continue
        matchedNode = nodeById.get(paper.node_id) || null
        if (matchedNode) break
      }
    }
    if (!matchedNode) return new Set()

    focusedVariable = ""
    focusedPaper = ""
    selectedId = matchedNode.id
    if (matchedNode.level === 4) {
      const directionId = (matchedNode.parent_ids || []).find(
        (id) => nodeById.get(id)?.level === 3
      ) || ""
      expandedTheme = directionTheme(directionId)
      expandedDirection = directionId
      focusedPaper = matchedNode.id
    } else if (matchedNode.level === 5) {
      const parentDirection = (matchedNode.parent_ids || []).find(
        (id) => nodeById.get(id)?.level === 3
      ) || ""
      expandedDirection = parentDirection
      expandedTheme = directionTheme(parentDirection)
      focusedPaper = ""
    } else if (matchedNode.level === 3) {
      if (matchedNode.type === "研究变量") {
        const directionId = (matchedNode.direction_ids || [])[0] || ""
        expandedTheme = (matchedNode.parent_ids || [])[0] || ""
        expandedDirection = directionId
        focusedVariable = matchedNode.id
      } else {
        expandedTheme = directionTheme(matchedNode.id)
        expandedDirection = matchedNode.id
      }
    } else if (matchedNode.level === 2) {
      expandedTheme = matchedNode.id
      expandedDirection = ""
    } else {
      expandedTheme = ""
      expandedDirection = ""
    }
    return new Set([matchedNode.id, ...ancestors(matchedNode.id)])
  }

  function relatedPaperNodeIds(variableId) {
    return new Set(
      allEdges
        .filter((edge) => edge.kind === "paper_variable" && edge.target === variableId)
        .map((edge) => edge.source)
        .filter((nodeId) => nodeById.get(nodeId)?.type === "论文")
    )
  }

  function synthesisAppliesToDirection(node, directionId) {
    if ((node.parent_ids || []).includes(directionId)) return true
    const direction = nodeById.get(directionId)
    if (!direction) return false
    if (
      node.label.includes(direction.label)
      || direction.label.includes(node.label)
    ) return true
    return (node.paper_ids || []).some((paperId) =>
      (direction.paper_ids || []).includes(paperId)
    )
  }

  function visibleNodeIds() {
    const queryMatches = searchMatches(searchInput.value || "")
    if (queryMatches) return queryMatches
    const visible = new Set([data.core_id])
    const filteredThemes = themes
      .map((theme) => nodeById.get(theme.id))
      .filter((node) => node && nodeMatchesFilters(node))

    if (!expandedTheme) {
      for (const node of filteredThemes) visible.add(node.id)
      return visible
    }

    const theme = nodeById.get(expandedTheme)
    if (!theme) return visible
    visible.add(theme.id)
    const thirdLayer = childrenOf(expandedTheme, 3)
    const directions = thirdLayer
      .filter(
        (node) => node.type === "研究方向" || node.type === "待细分方向"
      )
      .filter(nodeMatchesFilters)
    const shownDirections = expandedDirection
      ? directions.filter((node) => node.id === expandedDirection)
      : directions
    for (const node of shownDirections) visible.add(node.id)

    const directionVariableIds = new Set(
      allNodes
        .filter(
          (node) => node.type === "研究变量"
            && (!expandedDirection
              || (node.direction_ids || []).includes(expandedDirection))
            && (node.parent_ids || []).includes(expandedTheme)
            && nodeMatchesFilters(node)
        )
        .map((node) => node.id)
    )
    for (const variableId of directionVariableIds) visible.add(variableId)

    if (mode === "tree" && expandedDirection) {
      for (const paperNode of allNodes) {
        if (
          paperNode.type === "论文"
          && (paperNode.parent_ids || []).includes(expandedDirection)
          && nodeMatchesFilters(paperNode)
        ) visible.add(paperNode.id)
      }
      for (const synthesisNode of allNodes) {
        if (
          synthesisNode.level === 5
          && showSynthesis.checked
          && synthesisAppliesToDirection(synthesisNode, expandedDirection)
          && nodeMatchesFilters(synthesisNode)
        ) visible.add(synthesisNode.id)
      }
    }

    if (focusedVariable && mode === "tree") {
      visible.add(focusedVariable)
      for (const paperNodeId of relatedPaperNodeIds(focusedVariable)) {
        const paperNode = nodeById.get(paperNodeId)
        if (paperNode && nodeMatchesFilters(paperNode)) visible.add(paperNodeId)
      }
    }
    if (focusedPaper) {
      const paperNode = nodeById.get(focusedPaper)
      if (paperNode && nodeMatchesFilters(paperNode)) {
        visible.add(focusedPaper)
        for (const parentId of paperNode.parent_ids || []) {
          if (nodeById.get(parentId)?.level === 3) visible.add(parentId)
        }
      }
    }
    return visible
  }

  function nodeSize(node) {
    if (node?.type === "研究变量") return { width: 174, height: 62 }
    if (node?.type === "论文") return { width: 246, height: 76 }
    if (node?.level === 5) return { width: 214, height: 74 }
    if (node?.level === 1) return { width: 248, height: 70 }
    if (node?.level === 2) return { width: 222, height: 68 }
    if (node?.level === 3) return { width: 190, height: 64 }
    return { width: 220, height: 68 }
  }

  function placeCentered(items, level, y, width, gap) {
    const size = nodeSize(nodeById.get(items[0]?.id || data.core_id))
    const itemWidth = size.width
    const total = items.length * itemWidth + Math.max(0, items.length - 1) * gap
    const start = Math.max(35, (width - total) / 2)
    items.forEach((item, index) => {
      lastPositions.set(item.id, {
        x: start + index * (itemWidth + gap) + itemWidth / 2,
        y,
        ...nodeSize(item),
      })
    })
  }

  function gridLayout(items, level, startY, columns, width, gapX, gapY) {
    const sizes = items.map((item) => nodeSize(item))
    const itemWidth = Math.max(
      ...sizes.map((size) => size.width),
      level === 4 ? 174 : 180,
    )
    const itemHeight = Math.max(...sizes.map((size) => size.height), 62)
    const totalWidth = columns * itemWidth + Math.max(0, columns - 1) * gapX
    const startX = Math.max(35, (width - totalWidth) / 2)
    items.forEach((item, index) => {
      const row = Math.floor(index / columns)
      const column = index % columns
      lastPositions.set(item.id, {
        x: startX + column * (itemWidth + gapX) + itemWidth / 2,
        y: startY + row * (itemHeight + gapY),
        ...nodeSize(item),
      })
    })
    return Math.ceil(items.length / columns)
  }

  function buildLayout(visible) {
    lastPositions = new Map()
    const virtualWidth = 1180
    const core = nodeById.get(data.core_id)

    if (!expandedTheme) {
      const visibleThemes = themes.filter((theme) => visible.has(theme.id))
      const center = { x: virtualWidth / 2, y: 320 }
      if (core) {
        lastPositions.set(core.id, { ...center, ...nodeSize(core) })
      }
      const slots = [
        { x: 250, y: 145 },
        { x: 590, y: 110 },
        { x: 930, y: 145 },
        { x: 250, y: 520 },
        { x: 590, y: 560 },
        { x: 930, y: 520 },
      ]
      visibleThemes.forEach((theme, index) => {
        const slot = slots[index % slots.length]
        lastPositions.set(theme.id, { ...slot, ...nodeSize(theme) })
      })
      return { width: virtualWidth, height: 700 }
    }

    if (core) {
      lastPositions.set(core.id, {
        x: virtualWidth / 2,
        y: 78,
        ...nodeSize(core),
      })
    }
    const theme = nodeById.get(expandedTheme)
    if (theme && visible.has(theme.id)) {
      lastPositions.set(theme.id, {
        x: virtualWidth / 2,
        y: 185,
        ...nodeSize(theme),
      })
    }

    const questions = childrenOf(expandedTheme, 3)
      .filter(
        (node) => node.type === "研究方向" || node.type === "待细分方向"
      )
      .filter((node) => visible.has(node.id))
    const themeVariables = allNodes.filter(
      (node) => node.type === "研究变量"
        && visible.has(node.id)
        && (node.parent_ids || []).includes(expandedTheme)
    )
    if (!expandedDirection) {
      const questionRows = gridLayout(
        questions,
        3,
        320,
        mode === "network" ? 6 : 5,
        virtualWidth,
        20,
        62,
      )
      const variableRows = gridLayout(
        themeVariables,
        3,
        420 + questionRows * 72,
        6,
        virtualWidth,
        18,
        62,
      )
      return {
        width: virtualWidth,
        height: Math.max(
          760,
          510 + questionRows * 82 + variableRows * 76,
        ),
      }
    }

    const direction = nodeById.get(expandedDirection)
    if (direction && visible.has(direction.id)) {
      lastPositions.set(direction.id, {
        x: virtualWidth / 2,
        y: 285,
        ...nodeSize(direction),
      })
    }
    const variables = allNodes.filter(
      (node) => node.type === "研究变量"
        && visible.has(node.id)
        && (node.direction_ids || []).includes(expandedDirection)
    )
    const variableRows = gridLayout(
      variables,
      3,
      395,
      mode === "network" ? 5 : 4,
      virtualWidth,
      20,
      66,
    )
    const paperNodes = allNodes.filter(
      (node) => node.type === "论文" && visible.has(node.id)
    )
    const paperRows = gridLayout(
      paperNodes,
      4,
      510 + variableRows * 76,
      3,
      virtualWidth,
      24,
      72,
    )
    const synthesisNodes = allNodes.filter(
      (node) => node.level === 5 && visible.has(node.id)
    )
    const synthesisRows = gridLayout(
      synthesisNodes,
      5,
      650 + variableRows * 76 + paperRows * 82,
      3,
      virtualWidth,
      24,
      68,
    )
    return {
      width: virtualWidth,
      height: Math.max(
        820,
        700 + variableRows * 82 + paperRows * 82 + synthesisRows * 82,
      ),
    }
  }

  function drawEdge(edge, visible, activeIds) {
    const source = lastPositions.get(edge.source)
    const target = lastPositions.get(edge.target)
    if (!source || !target || !visible.has(edge.source) || !visible.has(edge.target)) return
    if (mode === "tree") {
      if (edge.kind === "research_relation") return
      if (
        edge.kind === "paper_variable"
        && (!focusedVariable || edge.target !== focusedVariable)
      ) return
    }
    if (mode === "network") {
      if (edge.kind === "paper_variable") return
      if (edge.kind === "research_relation" && !relationToggle.checked) return
    }
    const sameRow = Math.abs(source.y - target.y) < 36
    const pathData = sameRow
      ? `M ${source.x + (target.x > source.x ? source.width / 2 : -source.width / 2)} ${source.y} C ${source.x + (target.x > source.x ? 100 : -100)} ${source.y}, ${target.x + (target.x > source.x ? -100 : 100)} ${target.y}, ${target.x + (target.x > source.x ? -target.width / 2 : target.width / 2)} ${target.y}`
      : `M ${source.x} ${source.y + source.height / 2} C ${source.x} ${source.y + source.height / 2 + 42}, ${target.x} ${target.y - target.height / 2 - 42}, ${target.x} ${target.y - target.height / 2}`
    const statusClass = edge.relation_status === "正向关联"
      ? "status-positive"
      : edge.relation_status === "负向关联"
        ? "status-negative"
        : edge.relation_status === "无显著关联"
          ? "status-null"
          : edge.relation_status === "存在条件性关系"
            ? "status-conditional"
            : edge.relation_status === "研究结果不一致"
              ? "status-inconsistent"
              : edge.relation_status === "证据不足"
                ? "status-insufficient"
                : ""
    const path = create("path", {
      d: pathData,
      class: `kg-edge ${edge.kind} ${statusClass}${activeIds.has(edge.source) && activeIds.has(edge.target) ? " active" : ""}${selectedId && !activeIds.has(edge.source) && !activeIds.has(edge.target) ? " muted" : ""}`,
    })
    const title = create("title")
    title.textContent = `${edge.relation} · 论文 ${(edge.paper_ids || []).length} 篇 · 证据 ${edge.evidence_count || 0} 条`
    path.appendChild(title)
    viewport.appendChild(path)
  }

  function nodeMeta(node) {
    if (node.level === 1) return `${node.paper_count || 0} 篇范围内论文`
    if (node.level === 2) return `${node.paper_count || 0} 篇 · ${node.evidence_count || 0} 条证据`
    if (node.level === 5) {
      return `${node.variable_role || node.type} · ${node.paper_count || 0} 篇 · ${node.evidence_count || 0} 条证据`
    }
    if (node.level === 3) {
      const children = node.child_ids || []
      const variables = children.filter((id) => nodeById.get(id)?.type === "研究变量").length
      return `${node.paper_count || 0} 篇 · ${variables} 个变量`
    }
    if (node.type === "研究变量") return `${node.variable_role || "研究变量"} · ${node.paper_count || 0} 篇`
    const paper = paperByNodeId.get(node.id)
    return paper ? `${paper.first_author || "作者待核对"} · ${paper.year || "年份待核对"}` : "查看论文详情"
  }

  function drawNode(node, activeIds) {
    const pos = lastPositions.get(node.id)
    if (!pos) return
    const nodeClass = node.type === "论文"
      ? "node-paper"
      : node.type === "研究变量"
        ? "node-variable"
        : node.type === "学术共识"
          ? "node-consensus"
          : node.type === "学术争议"
            ? "node-controversy"
            : node.type === "研究空白"
              ? "node-gap"
        : node.type === "研究方向"
          ? "node-direction"
          : node.type === "待细分方向"
            ? "node-pending-direction"
            : ""
    const group = create("g", {
      class: `kg-node level-${node.level} ${nodeClass}${selectedId === node.id ? " selected" : ""}`,
      tabindex: "0",
      role: "button",
      "aria-label": `${node.type}：${node.label}`,
    })
    group.appendChild(create("rect", {
      x: pos.x - pos.width / 2,
      y: pos.y - pos.height / 2,
      width: pos.width,
      height: pos.height,
      rx: node.level === 1
        ? 9
        : node.type === "研究变量"
          ? 18
          : node.type === "论文"
            ? 5
            : 7,
    }))
    const foreign = create("foreignObject", {
      x: pos.x - pos.width / 2 + 12,
      y: pos.y - pos.height / 2 + 10,
      width: pos.width - 24,
      height: pos.height - 16,
      class: "kg-foreign",
    })
    const wrapper = document.createElement("div")
    wrapper.style.width = "100%"
    wrapper.style.height = "100%"
    const label = document.createElement("div")
    label.className = "kg-label"
    label.textContent = node.label
    const meta = document.createElement("span")
    meta.className = "kg-meta"
    meta.textContent = nodeMeta(node)
    wrapper.append(label, meta)
    foreign.appendChild(wrapper)
    group.appendChild(foreign)
    group.onclick = () => handleNodeClick(node)
    group.ondblclick = (event) => {
      event.stopPropagation()
      showDetail(node)
    }
    group.onkeydown = (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault()
        handleNodeClick(node)
      }
    }
    viewport.appendChild(group)
  }

  function handleNodeClick(node) {
    selectedId = node.id
    if (node.id === data.core_id) {
      expandedTheme = ""
      expandedDirection = ""
      focusedVariable = ""
      focusedPaper = ""
      showDetail(node)
    } else if (node.level === 2) {
      if (expandedTheme === node.id) {
        expandedTheme = ""
        expandedDirection = ""
      } else {
        expandedTheme = node.id
        expandedDirection = ""
      }
      focusedVariable = ""
      focusedPaper = ""
      hideDetail()
    } else if (node.type === "研究变量") {
      expandedTheme = (node.parent_ids || [])[0] || expandedTheme
      expandedDirection = (node.direction_ids || [])[0] || expandedDirection
      focusedVariable = node.id
      focusedPaper = ""
      showDetail(node)
    } else if (node.level === 3) {
      if (expandedDirection === node.id) {
        expandedDirection = ""
        hideDetail()
      } else {
        expandedDirection = node.id
        expandedTheme = directionTheme(node.id) || expandedTheme
        hideDetail()
      }
      focusedVariable = ""
      focusedPaper = ""
    } else if (node.type === "论文") {
      const directionId = (node.parent_ids || []).find(
        (id) => nodeById.get(id)?.level === 3
      ) || expandedDirection
      expandedDirection = directionId
      expandedTheme = directionTheme(directionId) || expandedTheme
      focusedPaper = node.id
      showDetail(node)
    } else {
      const directionId = (node.parent_ids || []).find(
        (id) => nodeById.get(id)?.level === 3
      ) || expandedDirection
      expandedDirection = directionId
      expandedTheme = directionTheme(directionId) || expandedTheme
      showDetail(node)
    }
    renderAndFit()
  }

  function applyViewport() {
    viewport.setAttribute("transform", `translate(${panX} ${panY}) scale(${zoom})`)
  }

  function fitView() {
    if (!lastPositions.size) return
    const values = [...lastPositions.values()]
    const minX = Math.min(...values.map((item) => item.x - item.width / 2))
    const maxX = Math.max(...values.map((item) => item.x + item.width / 2))
    const minY = Math.min(...values.map((item) => item.y - item.height / 2))
    const maxY = Math.max(...values.map((item) => item.y + item.height / 2))
    const width = Math.max(1, maxX - minX)
    const height = Math.max(1, maxY - minY)
    const panelReserve = detail.hidden
      ? 0
      : Math.min(450, Math.max(300, svg.clientWidth * 0.36))
    const availableWidth = Math.max(360, svg.clientWidth - panelReserve)
    const scaleX = (availableWidth - 80) / width
    const scaleY = (svg.clientHeight - 90) / height
    zoom = Math.max(0.32, Math.min(1.25, scaleX, scaleY))
    panX = (availableWidth - width * zoom) / 2 - minX * zoom
    panY = (svg.clientHeight - height * zoom) / 2 - minY * zoom
    applyViewport()
  }

  function activePath() {
    const active = new Set([selectedId])
    for (const id of ancestors(selectedId)) active.add(id)
    return active
  }

  function render() {
    currentVisible = visibleNodeIds()
    const layout = buildLayout(currentVisible)
    svg.setAttribute("viewBox", `0 0 ${layout.width} ${layout.height}`)
    svg.style.height = `${Math.max(760, Math.min(1600, layout.height + 100))}px`
    viewport.replaceChildren()
    const activeIds = activePath()
    for (const edge of allEdges) drawEdge(edge, currentVisible, activeIds)
    for (const node of allNodes) {
      if (currentVisible.has(node.id)) drawNode(node, activeIds)
    }
    empty.hidden = currentVisible.size > 0
    const modeLabel = mode === "tree" ? "层级树" : "关系网络"
    const visibleRelations = allEdges.filter(
      (edge) => currentVisible.has(edge.source)
        && currentVisible.has(edge.target)
        && (mode === "network" ? edge.kind === "research_relation" : edge.kind !== "research_relation")
    ).length
    status.textContent = `${modeLabel} · ${currentVisible.size} 个节点 · ${visibleRelations} 条可见关系`
    const names = [data.core_direction]
    if (expandedTheme) names.push(nodeById.get(expandedTheme)?.label || "")
    if (expandedDirection) names.push(nodeById.get(expandedDirection)?.label || "")
    if (focusedVariable) names.push(nodeById.get(focusedVariable)?.label || "")
    if (focusedPaper) names.push(nodeById.get(focusedPaper)?.label || "")
    breadcrumb.textContent = names.filter(Boolean).join("  /  ")
  }

  function renderAndFit() {
    try {
      render()
      requestAnimationFrame(() => {
        try {
          fitView()
        } catch (error) {
          status.textContent = `视图适配失败：${error?.message || error}`
        }
      })
    } catch (error) {
      status.textContent = `图层计算失败：${error?.message || error}`
      empty.hidden = false
      empty.querySelector("strong").textContent = "知识图谱图层计算失败"
      empty.querySelector("span").textContent =
        "请重置视图或检查筛选条件；原始论文数据未被修改。"
    }
  }

  function addFact(label, value) {
    if (!value) return
    const row = document.createElement("div")
    row.className = "kg-fact"
    const name = document.createElement("b")
    name.textContent = label
    const text = document.createElement("span")
    text.textContent = value
    row.append(name, text)
    detailBody.querySelector(".kg-facts")?.appendChild(row)
  }

  function addEvidence(title, evidence) {
    if (!evidence) return
    const block = document.createElement("div")
    block.className = "kg-evidence"
    const heading = document.createElement("strong")
    heading.textContent = title
    const quote = document.createElement("p")
    quote.textContent = evidence.quote
    const source = document.createElement("small")
    const location = evidence.source_type === "摘要"
      ? "摘要（未提供可检索全文）"
      : evidence.source_type === "元数据"
        ? "论文题录"
        : `第 ${evidence.page_number} 页`
    source.textContent = `${evidence.document_title} · ${evidence.source_type || "全文"} · ${location} · ${evidence.file_name}`
    block.append(heading, quote, source)
    detailBody.appendChild(block)
  }

  function addPaperCard(paper) {
    if (!paper) return
    const card = document.createElement("div")
    card.className = "kg-paper-card"
    const title = document.createElement("strong")
    title.textContent = paper.title
    const meta = document.createElement("span")
    meta.textContent = `${paper.first_author || "作者待核对"} · ${paper.year || "年份待核对"} · DOI：${paper.doi || "未提供"}`
    card.append(title, meta)
    if (paper.source_url && /^https?:\/\//i.test(paper.source_url)) {
      const link = document.createElement("a")
      link.className = "kg-link"
      link.href = paper.source_url
      link.target = "_blank"
      link.rel = "noopener noreferrer"
      link.textContent = "打开论文来源"
      card.appendChild(link)
    }
    detailBody.appendChild(card)
  }

  function addRelationCard(edge) {
    if (!edge) return
    const block = document.createElement("div")
    block.className = "kg-relation-card"
    const title = document.createElement("strong")
    title.textContent = edge.relation
    const meta = document.createElement("span")
    meta.textContent = `论文 ${(edge.paper_ids || []).length} 篇 · 证据 ${edge.evidence_count || 0} 条 · ${edge.relation_origin || "原文证据"}`
    block.append(title, meta)
    detailBody.appendChild(block)
  }

  function resetDetail() {
    detailType.textContent = ""
    detailTitle.textContent = ""
    detailBody.replaceChildren()
  }

  function showDetail(node) {
    selectedId = node.id
    resetDetail()
    detailType.textContent = node.type || "图谱节点"
    detailTitle.textContent = node.label
    const summary = document.createElement("p")
    summary.className = "kg-summary"
    summary.textContent = node.description || "暂无说明。"
    detailBody.appendChild(summary)

    const facts = document.createElement("div")
    facts.className = "kg-facts"
    detailBody.appendChild(facts)

    if (node.type === "论文") {
      const paper = paperByNodeId.get(node.id)
      if (paper) {
        addFact("第一作者", paper.first_author)
        addFact("发表年份", paper.year)
        addFact("期刊", paper.journal || "未提供")
        addFact("DOI", paper.doi || "未提供")
        addFact("研究方法", paper.method_summary)
        addFact("研究结论", paper.conclusion_summary)
        addFact("研究局限", paper.limitation_summary)
        addFact("分析状态", paper.status)
        addFact("摘要", (paper.abstract || "").slice(0, 700))
        const methodEvidence = paper.methods?.[0]
        const resultEvidence = paper.results?.[0]
        const conclusionEvidence = paper.conclusions?.[0]
        addEvidence("方法原文", methodEvidence)
        addEvidence("结果原文", resultEvidence)
        addEvidence("结论原文", conclusionEvidence)
        addEvidence("摘要证据", paper.abstract_evidence)
        addPaperCard(paper)
      }
    } else if (node.type === "研究变量") {
      addFact("变量角色", node.variable_role)
      addFact("相关论文", `${node.paper_count || 0} 篇`)
      addFact("可追溯证据", `${node.evidence_count || 0} 条`)
      const paperEdges = allEdges.filter(
        (edge) => edge.kind === "paper_variable" && edge.target === node.id
      )
      const paperIds = [...new Set(paperEdges.map((edge) => edge.source))]
      for (const paperNodeId of paperIds.slice(0, 8)) {
        addPaperCard(paperByNodeId.get(paperNodeId))
      }
      const relations = allEdges.filter(
        (edge) => edge.kind === "research_relation"
          && (edge.source === node.id || edge.target === node.id)
      )
      for (const edge of relations.slice(0, 8)) addRelationCard(edge)
      if (node.evidence?.[0]) addEvidence("变量原文证据", node.evidence[0])
    } else if (node.level === 5) {
      addFact("综合类型", node.type)
      addFact("当前状态", node.variable_role)
      addFact("涉及论文", `${node.paper_count || 0} 篇`)
      addFact("可追溯证据", `${node.evidence_count || 0} 条`)
      const paperNodeIds = allEdges
        .filter(
          (edge) => edge.kind === "evidence_synthesis"
            && edge.target === node.id
        )
        .map((edge) => edge.source)
      for (const paperNodeId of [...new Set(paperNodeIds)].slice(0, 8)) {
        addPaperCard(paperByNodeId.get(paperNodeId))
      }
      for (const evidence of (node.evidence || []).slice(0, 4)) {
        addEvidence("综合证据", evidence)
      }
    } else if (node.type === "研究方向" || node.type === "待细分方向") {
      addFact("论文数量", String(node.paper_count || 0))
      addFact("原文证据", String(node.evidence_count || 0))
      const variableNames = (node.child_ids || [])
        .map((id) => nodeById.get(id))
        .filter((item) => item?.type === "研究变量")
        .map((item) => item.label)
      addFact("相关变量", variableNames.slice(0, 12).join("、"))
      const relations = allEdges.filter(
        (edge) => edge.kind === "research_relation"
          && (edge.source === node.id || edge.target === node.id)
      )
      for (const edge of relations.slice(0, 8)) addRelationCard(edge)
      if (node.evidence?.[0]) addEvidence("方向归属原文", node.evidence[0])
    } else if (node.level === 2) {
      const theme = themeById.get(node.id)
      if (theme) {
        addFact("研究方向", String(theme.direction_count || 0))
        addFact("研究变量", String(theme.variable_count || 0))
        addFact("主要方法", theme.method_summary)
        addFact("研究对象", theme.population_summary)
        addFact("评价指标", theme.measure_summary)
        addFact("代表性结论", theme.conclusion_summary)
        addFact("相关争议", theme.controversy_summary)
        addFact("研究不足", theme.limitation_summary)
      }
    } else {
      addFact("论文数量", String(node.paper_count || 0))
      addFact("原文证据", String(node.evidence_count || 0))
      if (node.evidence?.[0]) addEvidence("代表性原文", node.evidence[0])
    }
    detail.hidden = false
    requestAnimationFrame(fitView)
  }

  function hideDetail() {
    detail.hidden = true
  }

  function focusTheme(themeId) {
    selectedId = themeId
    expandedTheme = themeId
    expandedDirection = ""
    focusedVariable = ""
    focusedPaper = ""
    hideDetail()
    renderAndFit()
  }

  themeFilter.onchange = () => {
    const value = themeFilter.value
    if (value === "all") {
      expandedTheme = ""
      expandedDirection = ""
      focusedVariable = ""
      focusedPaper = ""
      selectedId = data.core_id
      searchInput.value = ""
      renderAndFit()
      return
    }
    focusTheme(value)
  }

  let searchTimer = null
  searchInput.oninput = () => {
    window.clearTimeout(searchTimer)
    searchTimer = window.setTimeout(() => {
      if (!searchInput.value.trim()) {
        expandedTheme = ""
        expandedDirection = ""
        focusedVariable = ""
        focusedPaper = ""
        selectedId = data.core_id
      }
      renderAndFit()
    }, 90)
  }

  for (const button of parentElement.querySelectorAll(".kg-modes button")) {
    button.onclick = () => {
      mode = button.dataset.mode
      for (const item of parentElement.querySelectorAll(".kg-modes button")) {
        item.classList.toggle("active", item === button)
      }
      relationToggle.disabled = mode !== "network"
      relationToggle.checked = mode === "network"
      focusedVariable = ""
      focusedPaper = ""
      selectedId = expandedDirection || expandedTheme || data.core_id
      renderAndFit()
    }
  }
  relationToggle.disabled = true
  relationToggle.onchange = render
  yearFilter.onchange = renderAndFit
  evidenceFilter.onchange = renderAndFit
  methodFilter.onchange = renderAndFit
  for (const toggle of [showVariables, showPapers, showSynthesis]) {
    toggle.onchange = renderAndFit
  }

  const resetGraphView = () => {
    expandedTheme = ""
    expandedDirection = ""
    focusedVariable = ""
    focusedPaper = ""
    selectedId = data.core_id
    themeFilter.value = "all"
    yearFilter.value = "all"
    evidenceFilter.value = "all"
    methodFilter.value = "all"
    searchInput.value = ""
    showVariables.checked = true
    showPapers.checked = true
    showSynthesis.checked = true
    hideDetail()
    renderAndFit()
  }
  parentElement.querySelector("#kg-overview").onclick = resetGraphView
  parentElement.querySelector("#kg-reset").onclick = resetGraphView
  parentElement.querySelector("#kg-back").onclick = () => {
    if (focusedPaper) focusedPaper = ""
    else if (focusedVariable) focusedVariable = ""
    else if (expandedDirection) expandedDirection = ""
    else if (expandedTheme) expandedTheme = ""
    else selectedId = data.core_id
    hideDetail()
    renderAndFit()
  }
  parentElement.querySelector("#kg-collapse").onclick = () => {
    expandedTheme = ""
    expandedDirection = ""
    focusedVariable = ""
    focusedPaper = ""
    selectedId = data.core_id
    renderAndFit()
  }
  parentElement.querySelector("#kg-fit").onclick = fitView
  parentElement.querySelector("#kg-fullscreen").onclick = () => {
    if (!document.fullscreenElement) {
      shell.requestFullscreen?.().then(() => renderAndFit())
    } else {
      document.exitFullscreen?.().then(() => renderAndFit())
    }
  }
  parentElement.querySelector("#kg-detail-close").onclick = () => {
    hideDetail()
    requestAnimationFrame(fitView)
  }
  parentElement.querySelector("#kg-zoom-in").onclick = () => {
    zoom = Math.min(1.8, zoom + 0.12)
    applyViewport()
  }
  parentElement.querySelector("#kg-zoom-out").onclick = () => {
    zoom = Math.max(0.26, zoom - 0.12)
    applyViewport()
  }

  svg.onpointerdown = (event) => {
    if (event.button !== 0) return
    dragging = true
    dragStart = { x: event.clientX, y: event.clientY, panX, panY }
    svg.classList.add("dragging")
    svg.setPointerCapture(event.pointerId)
  }
  svg.onpointermove = (event) => {
    if (!dragging || !dragStart) return
    panX = dragStart.panX + event.clientX - dragStart.x
    panY = dragStart.panY + event.clientY - dragStart.y
    applyViewport()
  }
  svg.onpointerup = (event) => {
    dragging = false
    svg.classList.remove("dragging")
    if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId)
  }
  svg.onwheel = (event) => {
    event.preventDefault()
    zoom = Math.max(0.26, Math.min(1.8, zoom + (event.deltaY < 0 ? 0.08 : -0.08)))
    applyViewport()
  }

  document.onfullscreenchange = () => {
    window.setTimeout(renderAndFit, 80)
  }

  renderAndFit()
}
"""


_GRAPH_COMPONENT = st.components.v2.component(
    "researchmind_hierarchical_knowledge_map_v3",
    html=GRAPH_HTML,
    css=GRAPH_CSS,
    js=GRAPH_JS,
)


def render_knowledge_graph_component(
    payload: dict[str, Any],
    *,
    key: str,
    height: int = 760,
) -> Any:
    """挂载五层知识地图组件，供页面在失败时统一回退。"""

    return _GRAPH_COMPONENT(
        data=payload,
        key=key,
        height=height,
    )


__all__ = ["render_knowledge_graph_component"]
